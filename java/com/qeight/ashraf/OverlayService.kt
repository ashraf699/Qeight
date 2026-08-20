package com.ashraf.qeight

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.ServiceInfo
import android.graphics.Color
import android.graphics.PixelFormat
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.view.ContextThemeWrapper
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * OverlayService
 *
 * Foreground service that manages the Qeight billiards analysis overlay.
 * Acquires screen frames via the MediaProjection API (requires explicit
 * user consent at launch), feeds them through an on-device Vulkan/CPU
 * geometry pipeline, and renders a transparent informational overlay.
 *
 * No data is transmitted off-device. The user controls the overlay
 * entirely through the floating cluster UI and may stop the service
 * at any time.
 */
class OverlayService : Service() {

    companion object {
        private const val TAG = "OverlayService"
        private const val NOTIFICATION_CHANNEL_ID = "qeight_overlay"
        private const val NOTIFICATION_ID = 1001
        const val EXTRA_PROJECTION_DATA = "extra_projection_data"
        const val EXTRA_PROJECTION_RESULT_CODE = "extra_projection_result_code"
        private const val PREFS_NAME = "qeight_prefs"
        private const val KEY_ROI_CALIBRATION = "roi_calibration_json"

        /**
         * Master switch for verbose logging.
         * Set to [true] to enable all Log.d / Log.w / Log.e output.
         * Default is [false] (logging OFF) for production builds.
         */
        private const val LOGGING_ENABLED = false

        private fun logD(tag: String, msg: String) {
            if (LOGGING_ENABLED) Log.d(tag, msg)
        }
        private fun logW(tag: String, msg: String, tr: Throwable? = null) {
            if (!LOGGING_ENABLED) return
            if (tr != null) Log.w(tag, msg, tr) else Log.w(tag, msg)
        }
        private fun logE(tag: String, msg: String, tr: Throwable? = null) {
            if (!LOGGING_ENABLED) return
            if (tr != null) Log.e(tag, msg, tr) else Log.e(tag, msg)
        }
    }

    // ── Mode enum ─────────────────────────────────────────────────────────────
    // Single source of truth for which logical mode is active. The native gate
    // (QeightJNI.setScreenMode) is kept in sync whenever this field changes.
    // PIPELINE = normal overlay operation; INDIRECT = indirect-mode magnifier.
    private enum class ScreenMode { PIPELINE, INDIRECT }

    @Volatile private var screenMode: ScreenMode = ScreenMode.PIPELINE

    private lateinit var windowManager: WindowManager
    private lateinit var prefs: SharedPreferences

    // Vulkan render surface
    private var renderSurface: SurfaceView? = null
    private var renderSurfaceParams: WindowManager.LayoutParams? = null

    // Floating button
    private var floatingButton: View? = null
    private var floatingButtonParams: WindowManager.LayoutParams? = null

    // Floating panel
    private var floatingPanel: LinearLayout? = null
    private var floatingPanelParams: WindowManager.LayoutParams? = null
    private var isPanelVisible = false

    // MediaProjection and screen capture.
    //
    // IMPORTANT (Android 14+): MediaProjection#createVirtualDisplay() may be
    // called only ONCE per MediaProjection instance — a second call throws
    // SecurityException, and on some OEM builds this instead silently fires
    // MediaProjection.Callback#onStop() on the whole projection, which was
    // tearing down this entire service moments after entering indirect mode.
    // See: https://developer.android.com/about/versions/14/behavior-changes-14
    //
    // Previously, pipeline and indirect mode each owned a separate
    // ScreenCaptureManager (and therefore a separate VirtualDisplay built
    // from the same MediaProjection) — a second createVirtualDisplay() call
    // on the shared token. Since pipeline and indirect mode are mutually
    // exclusive (screenMode gates which one is active; they are never
    // consuming frames concurrently), a single shared ScreenCaptureManager
    // is safe: only one VirtualDisplay/ImageReader is ever created for the
    // entire service lifetime, and switching modes just pauses one loop and
    // resumes the other via pauseCapture()/resumeCapture() rather than
    // stopping/rebuilding the capture pipeline.
    private var mediaProjection: MediaProjection? = null
    private var screenCaptureManager: ScreenCaptureManager? = null
    private var indirectLoopThread: Thread? = null

    // CalibrationManager instance
    private var calibrationManager: CalibrationManager? = null

    // IndirectModeController instance
    private var indirectModeController: IndirectModeController? = null

    // Screen dimensions
    private var screenWidth  = 2400
    private var screenHeight = 1080

    // Capture-resolution dimensions
    private var captureScale  = 1.0f
    private var captureWidth  = 2400
    private var captureHeight = 1080

    // Pipeline parameter state
    private var cbcReflections      = 0
    private var tgtReflections      = 2
    private var lineThickness       = 4
    private var cueForceStat        = 11
    private var cueSpinStat         = 11
    private var trajectoryPowerPct  = 100
    private var cushionShotsEnabled = true
    private var overlayColorR       = 255
    private var overlayColorG       = 255
    private var overlayColorB       = 255
    private var overlayColorA       = 255
    private var indirectMagnifierZoom = 7

    // ROI
    private var roiX1 = 442
    private var roiY1 = 227
    private var roiX2 = 1958
    private var roiY2 = 1012

    @Volatile private var surfaceReady       = false
    @Volatile private var rendererDestroyed  = false

    @Volatile private var isPausedByPanel   = false
    @Volatile private var isPausedBySurface = false
    @Volatile private var isPausedByUser    = false
    @Volatile private var isPausedByCluster = false

    // Guide band fill state
    private var parallelFillEnabled = true
    private var parallelFillAlpha   = 0.25f

    // Pipeline capture loop — pipeline-only, no consumer branching
    private val isLoopRunning = AtomicBoolean(false)
    private var loopThread: Thread? = null

    private val mainHandler = Handler(Looper.getMainLooper())

    // Trajectory power button references
    private val trajectoryPowerButtons = arrayOfNulls<Button>(4)

    // Cluster button references
    private var eightBallButton:       Button? = null
    private var clusterPauseButton:    Button? = null
    private var clusterSettingsButton: Button? = null
    private var clusterStopButton:     Button? = null
    private var clusterIndirectButton: Button? = null

    // Floating button cluster state
    private var clusterExpanded = false
    private val autoCollapseRunnable = Runnable { collapseCluster() }

    // ──────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ──────────────────────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        logD(TAG, "onCreate: OverlayService starting")

        // g_screenMode is a process-global static in the native layer
        // (QeightJNI.cpp), not tied to this service instance's lifecycle.
        // If a previous OverlayService instance was torn down while still
        // in INDIRECT mode — via a path that skipped onIndirectModeExit()
        // (system-initiated stopService(), task kill, OEM background-kill,
        // etc.) — and the process itself was reused rather than killed,
        // that native static survives into this fresh instance even though
        // the Kotlin `screenMode` field below is freshly initialized to
        // PIPELINE. Left unaddressed, every processFrame()/renderOverlay()
        // call in this new session is silently suppressed by the native
        // mode gate, and the overlay never renders again until the process
        // is actually killed. Force the native gate back to a known state
        // here, before addRenderSurface()/surfaceCreated() can run any
        // pipeline work, rather than relying on every possible teardown
        // path to have called setScreenMode(0) cleanly.
        try {
            QeightJNI.setScreenMode(0)
        } catch (e: Exception) {
            logW(TAG, "onCreate: setScreenMode(0) reset failed (non-fatal)", e)
        }

        try {
            windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
            prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

            createNotificationChannel()

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    NOTIFICATION_ID,
                    buildForegroundNotification(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
                )
            } else {
                startForeground(NOTIFICATION_ID, buildForegroundNotification())
            }

            readScreenDimensions()
            loadRoiFromPrefs()
            addRenderSurface()
            addFloatingButton()

            try {
                QEightTileService.requestTileUpdate(this)
                logD(TAG, "onCreate: Tile update requested")
            } catch (e: Exception) {
                logW(TAG, "onCreate: requestTileUpdate failed (non-fatal)", e)
            }

        } catch (e: Exception) {
            logE(TAG, "onCreate: Fatal exception", e)
            Toast.makeText(this, "Qeight failed to start: ${e.message}",
                Toast.LENGTH_LONG).show()
            stopSelf()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            // Tile service stop action — mode-gated so it cannot interfere
            // with indirect mode's own teardown sequence.
            if (intent?.action == "STOP_SERVICE") {
                if (screenMode == ScreenMode.PIPELINE) {
                    stopSelf()
                } else {
                    logD(TAG, "onStartCommand: STOP_SERVICE ignored while INDIRECT mode active")
                }
                return START_STICKY
            }

            val resultCode = intent?.getIntExtra(EXTRA_PROJECTION_RESULT_CODE, 0) ?: 0

            val projectionData: Intent? =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent?.getParcelableExtra(EXTRA_PROJECTION_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent?.getParcelableExtra(EXTRA_PROJECTION_DATA)
                }

            if (resultCode == android.app.Activity.RESULT_OK && projectionData != null) {
                val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                        as MediaProjectionManager
                mediaProjection = mgr.getMediaProjection(resultCode, projectionData)
                startScreenCapture()
            }
        } catch (e: Exception) {
            logE(TAG, "onStartCommand: Exception", e)
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    /**
     * Fires when the user swipes this app off the recents list. This is a
     * more reliable cross-session teardown hook than onDestroy() alone —
     * many OEMs (this app has seen it on ColorOS specifically) kill a
     * foreground service's process without ever delivering onDestroy(), but
     * still deliver onTaskRemoved() first. Without this override, swiping
     * the app away while indirect mode is paused-but-alive
     * (ScreenCaptureManager.pauseCapture() keeps the VirtualDisplay/
     * ImageReader/HandlerThread running by design — see that class's
     * header) leaves those resources orphaned: the next session builds a
     * brand-new ScreenCaptureManager and VirtualDisplay while the old one
     * is still mirroring the screen and posting frames, producing the
     * cross-session render flicker this override exists to prevent.
     *
     * stopSelf() forces Android to run the normal onDestroy() path (which
     * already tears down indirect mode, the capture manager, and native
     * state defensively) before the process is allowed to go away.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        logD(TAG, "onTaskRemoved: app swiped from recents — forcing full teardown via stopSelf()")
        try {
            stopSelf()
        } catch (e: Exception) {
            logE(TAG, "onTaskRemoved: stopSelf() failed", e)
        }
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        super.onDestroy()
        logD(TAG, "onDestroy: Releasing all resources — screenMode=$screenMode " +
                "indirectModeControllerNonNull=${indirectModeController != null} " +
                "screenCaptureManagerNonNull=${screenCaptureManager != null}")
        logW(TAG, "onDestroy: this was triggered by the SYSTEM calling stopService()/handleStopService — " +
                "check ColorOS battery optimization / background-activity / autostart permissions for this app, " +
                "and confirm startForeground() + FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION are still valid " +
                "at the moment indirect mode begins capturing.")

        // ── Indirect mode teardown (in case destroyed while INDIRECT active) ──
        stopIndirectLoop()
        try { indirectModeController?.stop() } catch (e: Exception) {
            logE(TAG, "onDestroy: indirectModeController stop failed", e)
        }
        indirectModeController = null

        // ── Pipeline teardown ─────────────────────────────────────────────────
        stopCaptureLoop()

        try {
            calibrationManager?.dismiss()
        } catch (e: Exception) {
            logE(TAG, "onDestroy: calibrationManager dismiss failed", e)
        }
        calibrationManager = null

        // Only one ScreenCaptureManager exists now (shared between pipeline
        // and indirect mode), so it's stopped exactly once here, regardless
        // of which mode was active when the service was destroyed.
        try { screenCaptureManager?.stop() } catch (e: Exception) {
            logE(TAG, "onDestroy: screenCaptureManager stop failed", e)
        }
        screenCaptureManager = null
        try { QeightJNI.destroyPipeline() } catch (e: Exception) {
            logE(TAG, "onDestroy: destroyPipeline failed", e)
        }
        try { QeightJNI.destroyRenderer() } catch (e: Exception) {
            logE(TAG, "onDestroy: destroyRenderer failed", e)
        }
        try { QeightJNI.destroyVulkan() } catch (e: Exception) {
            logE(TAG, "onDestroy: destroyVulkan failed", e)
        }
        try { mediaProjection?.stop() } catch (e: Exception) {
            logE(TAG, "onDestroy: mediaProjection stop failed", e)
        }

        listOf(floatingPanel, floatingButton, renderSurface).forEach { view ->
            try { if (view?.parent != null) windowManager.removeView(view) }
            catch (e: Exception) { logE(TAG, "onDestroy: removeView failed for $view", e) }
        }

        // IMPORTANT: MainActivity registers its serviceStoppedReceiver via
        // LocalBroadcastManager (see MainActivity.inflateHomeScreen()). A
        // plain sendBroadcast() here is a GLOBAL broadcast and is never
        // delivered to a LocalBroadcastManager-registered receiver — the two
        // systems are entirely separate. That mismatch was the root cause of
        // the "stop from floating panel, then Start again -> stale error
        // toast with no re-prompt" bug: projectionData never got nulled out
        // on the MainActivity side, so the next START tap skipped the
        // permission re-request and replayed a dead MediaProjection token.
        // Broadcasting through LocalBroadcastManager (matching the receiver)
        // fixes delivery. sendBroadcast() is also still called as a harmless
        // no-op-if-unheard fallback in case anything else in the app ever
        // listens globally for this action.
        LocalBroadcastManager.getInstance(this)
            .sendBroadcast(Intent("com.ashraf.qeight.SERVICE_STOPPED"))
        sendBroadcast(Intent("com.ashraf.qeight.SERVICE_STOPPED"))

        try {
            QEightTileService.requestTileUpdate(this)
            logD(TAG, "onDestroy: Tile update requested")
        } catch (e: Exception) {
            logW(TAG, "onDestroy: requestTileUpdate failed (non-fatal)", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Pipeline capture loop — pipeline-only, unconditional dispatch
    // ──────────────────────────────────────────────────────────────────────────
    // This loop runs only while screenMode == PIPELINE. It dispatches every
    // acquired frame unconditionally to QeightJNI.processFrame /
    // QeightJNI.renderOverlay — there is no consumer branch here. Indirect
    // mode has its own consuming thread (indirectLoopThread) but reads from
    // the SAME shared screenCaptureManager instance; the two loops are never
    // running concurrently since screenMode gates which one is active.

    private fun startCaptureLoop() {
        isLoopRunning.set(true)
        loopThread = Thread({
            logD(TAG, "captureLoop: started")
            while (isLoopRunning.get()) {

                // Hard gate: surface mid-teardown — unsafe for anything render-related.
                if (isPausedBySurface) {
                    try { Thread.sleep(66L) } catch (e: InterruptedException) { break }
                    continue
                }

                // Pipeline pause gate: user pause, panel open, cluster open, or
                // surface not yet ready. renderOverlay() must not be called until
                // surfaceReady == true (mirrors the implicit guarantee at first startup).
                if (!surfaceReady || isPausedByUser || isPausedByPanel || isPausedByCluster) {
                    try { Thread.sleep(66L) } catch (e: InterruptedException) { break }
                    continue
                }

                try {
                    screenCaptureManager?.drainStaleFrames()
                    val buffer = screenCaptureManager?.acquireFrame()

                    if (buffer != null) {
                        QeightJNI.processFrame(buffer)
                        // Double-check surfaceReady after processFrame — a
                        // surfaceDestroyed() can race in between the two calls.
                        if (surfaceReady) {
                            QeightJNI.renderOverlay()
                        }
                    } else {
                        try { Thread.sleep(16L) } catch (e: InterruptedException) { break }
                    }

                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    logE(TAG, "captureLoop: exception in frame iteration", e)
                } finally {
                    screenCaptureManager?.releaseFrame()
                }
            }
            logD(TAG, "captureLoop: thread exiting")
        }, "QeightLoop").also { it.start() }
    }

    private fun stopCaptureLoop() {
        isLoopRunning.set(false)
        loopThread?.interrupt()
        try { loopThread?.join(2000) } catch (e: InterruptedException) {
            logW(TAG, "stopCaptureLoop: join interrupted")
        }
        loopThread = null
        logD(TAG, "stopCaptureLoop: done")
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Indirect-mode capture loop — separate consuming thread, but reads from
    // the SAME shared screenCaptureManager/VirtualDisplay as the pipeline
    // loop (see class-level comment on screenCaptureManager). Never runs
    // concurrently with the pipeline's loopThread — screenMode gates which
    // one is active — so there is no real frame-ownership race despite both
    // loops touching the same manager instance.
    // ──────────────────────────────────────────────────────────────────────────

    private fun startIndirectLoop() {
        val isRunning = AtomicBoolean(true)
        indirectLoopThread = Thread({
            logD(TAG, "indirectLoop: started")
            while (isRunning.get() && !Thread.currentThread().isInterrupted) {
                try {
                    screenCaptureManager?.drainStaleFrames()
                    val buffer = screenCaptureManager?.acquireFrame()

                    if (buffer != null) {
                        indirectModeController?.onCleanFrame(buffer)
                    } else {
                        try { Thread.sleep(16L) } catch (e: InterruptedException) { break }
                    }

                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    logE(TAG, "indirectLoop: exception in frame iteration", e)
                } finally {
                    screenCaptureManager?.releaseFrame()
                }
            }
            logD(TAG, "indirectLoop: thread exiting")
        }, "QeightIndirectLoop").also { it.start() }
    }

    private fun stopIndirectLoop() {
        indirectLoopThread?.interrupt()
        try { indirectLoopThread?.join(2000) } catch (e: InterruptedException) {
            logW(TAG, "stopIndirectLoop: join interrupted")
        }
        indirectLoopThread = null
        logD(TAG, "stopIndirectLoop: done")
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Screen helpers
    // ──────────────────────────────────────────────────────────────────────────

    private fun readScreenDimensions() {
        try {
            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(metrics)
            screenWidth  = maxOf(metrics.widthPixels, metrics.heightPixels)
            screenHeight = minOf(metrics.widthPixels, metrics.heightPixels)
        } catch (e: Exception) {
            logE(TAG, "readScreenDimensions: Using defaults 2400x1080", e)
        }
        captureScale  = (720f / screenHeight).coerceIn(0.05f, 1.0f)
        captureWidth  = Math.round(screenWidth  * captureScale)
        captureHeight = Math.round(screenHeight * captureScale)
        logD(TAG, "readScreenDimensions: native=${screenWidth}x${screenHeight} " +
                "captureScale=$captureScale capture=${captureWidth}x${captureHeight}")
    }

    private fun loadRoiFromPrefs() {
        try {
            val json = prefs.getString(KEY_ROI_CALIBRATION, null)
            if (json != null) {
                val obj = JSONObject(json)
                roiX1 = Math.round(obj.getInt("roi_x1") * captureScale)
                roiY1 = Math.round(obj.getInt("roi_y1") * captureScale)
                roiX2 = Math.round(obj.getInt("roi_x2") * captureScale)
                roiY2 = Math.round(obj.getInt("roi_y2") * captureScale)
            } else {
                roiX1 = Math.round(442f  * captureWidth  / 2400f)
                roiY1 = Math.round(227f  * captureHeight / 1080f)
                roiX2 = Math.round(1958f * captureWidth  / 2400f)
                roiY2 = Math.round(1012f * captureHeight / 1080f)
            }
        } catch (e: Exception) {
            logE(TAG, "loadRoiFromPrefs: Exception", e)
        }
    }

    private fun reloadPoolTableBounds() {
        try {
            val json = prefs.getString("pool_table_calibration_json", null)
            if (json != null) {
                val obj = JSONObject(json)
                QeightJNI.setPoolTableBounds(
                    Math.round(obj.getInt("pool_x1") * captureScale),
                    Math.round(obj.getInt("pool_y1") * captureScale),
                    Math.round(obj.getInt("pool_x2") * captureScale),
                    Math.round(obj.getInt("pool_y2") * captureScale)
                )
                logD(TAG, "reloadPoolTableBounds: applied from prefs (capture-space)")
            } else {
                val defPoolX1 = Math.round(469f  * captureWidth  / 2400f)
                val defPoolY1 = Math.round(258f  * captureHeight / 1080f)
                val defPoolX2 = Math.round(1929f * captureWidth  / 2400f)
                val defPoolY2 = Math.round(983f  * captureHeight / 1080f)
                QeightJNI.setPoolTableBounds(defPoolX1, defPoolY1, defPoolX2, defPoolY2)
                logD(TAG,
                    "reloadPoolTableBounds: applied scaled defaults (capture-space) " +
                            "($defPoolX1,$defPoolY1,$defPoolX2,$defPoolY2)")
            }
        } catch (e: Exception) {
            logE(TAG, "reloadPoolTableBounds: $e")
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Rotation handling
    // ──────────────────────────────────────────────────────────────────────────
    // Dispatches to whichever capture manager is currently live based on
    // screenMode. Also updates IndirectModeController's geometry when a
    // rotation occurs while indirect mode is active so it doesn't keep
    // using pre-rotation capture dimensions for the remainder of the session.

    private fun handleRotation() {
        readScreenDimensions()

        // Only one ScreenCaptureManager exists now (shared between pipeline
        // and indirect mode), so rotation is handled on it unconditionally.
        // We still branch on screenMode purely to decide whether
        // IndirectModeController also needs its geometry refreshed.
        screenCaptureManager?.handleRotation(captureWidth, captureHeight)

        when (screenMode) {
            ScreenMode.PIPELINE -> {
                logD(TAG, "handleRotation: dispatched to shared capture manager (pipeline active)")
            }
            ScreenMode.INDIRECT -> {
                // Update the controller's held geometry so it doesn't keep using
                // the pre-rotation captureWidth/captureHeight/captureScale values
                // for the rest of the session.
                indirectModeController?.updateCaptureGeometry(captureWidth, captureHeight, captureScale)
                logD(TAG, "handleRotation: dispatched to shared capture manager " +
                        "(indirect active) and updated controller geometry")
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Render surface
    // ──────────────────────────────────────────────────────────────────────────

    private fun addRenderSurface() {
        try {
            renderSurface = SurfaceView(this).apply {
                setZOrderMediaOverlay(true)
                holder.setFormat(PixelFormat.RGBA_8888)
                holder.addCallback(object : SurfaceHolder.Callback {
                    override fun surfaceCreated(holder: SurfaceHolder) {
                        try {
                            if (rendererDestroyed) {
                                QeightJNI.initVulkan(assets)
                                rendererDestroyed = false
                                logD(TAG, "surfaceCreated: Re-initialized Vulkan after rotation")
                            } else {
                                QeightJNI.initVulkan(assets)
                            }

                            QeightJNI.initRenderer(holder.surface, captureWidth, captureHeight)

                            logD(TAG,
                                "surfaceCreated: initPipeline captureWidth=$captureWidth " +
                                        "captureHeight=$captureHeight captureScale=$captureScale"
                            )

                            QeightJNI.initPipeline(
                                captureWidth, captureHeight,
                                captureScale,
                                roiX1, roiY1, roiX2, roiY2
                            )
                            reloadPoolTableBounds()

                            try {
                                QeightJNI.setCbcReflections(cbcReflections)
                                QeightJNI.setTgtReflections(tgtReflections)
                                QeightJNI.setCueForce(cueForceStat)
                                QeightJNI.setCueSpin(cueSpinStat)
                                QeightJNI.setTrajectoryPower(trajectoryPowerPct)
                                QeightJNI.setCushionShots(cushionShotsEnabled)
                                QeightJNI.setLineThickness(lineThickness)
                                QeightJNI.setOverlayColor(
                                    overlayColorR, overlayColorG,
                                    overlayColorB, overlayColorA
                                )
                                logD(TAG,
                                    "surfaceCreated: Pipeline state applied — " +
                                            "cbcReflections=$cbcReflections, " +
                                            "tgtReflections=$tgtReflections, " +
                                            "cushionShots=$cushionShotsEnabled, " +
                                            "lineThickness=$lineThickness, " +
                                            "overlayColor=($overlayColorR,$overlayColorG," +
                                            "$overlayColorB,$overlayColorA)")
                            } catch (e: Exception) {
                                logW(TAG, "surfaceCreated: Failed to apply pipeline state (non-fatal)", e)
                            }

                            try {
                                QeightJNI.setParallelLinesFill(parallelFillEnabled, parallelFillAlpha)
                                QeightJNI.setParallelLinesVisible(!parallelFillEnabled)
                            } catch (e: Exception) {
                                logW(TAG, "setParallelLinesFill/Visible failed (non-fatal)", e)
                            }

                            try {
                                val dummyBuffer = ByteBuffer.allocateDirect(
                                    captureWidth * captureHeight * 4
                                )
                                QeightJNI.processFrame(dummyBuffer)
                                logD(TAG, "surfaceCreated: pipeline pre-warm complete")
                            } catch (e: Exception) {
                                logW(TAG, "surfaceCreated: pipeline pre-warm failed (non-fatal)", e)
                            }

                            surfaceReady = true
                            logD(TAG, "surfaceCreated: pipeline ready")
                        } catch (e: Exception) {
                            logE(TAG, "surfaceCreated: init failed", e)
                            Toast.makeText(
                                this@OverlayService,
                                "GPU pipeline failed: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                            stopSelf()
                        }
                    }

                    override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}

                    override fun surfaceDestroyed(h: SurfaceHolder) {
                        logD(TAG, "surfaceDestroyed: Pausing loop and tearing down renderer")

                        isPausedBySurface = true
                        surfaceReady = false

                        try { Thread.sleep(300L) } catch (e: InterruptedException) {
                            logW(TAG, "surfaceDestroyed: sleep interrupted")
                        }

                        try {
                            QeightJNI.destroyRenderer()
                            QeightJNI.destroyVulkan()
                            rendererDestroyed = true
                            logD(TAG, "surfaceDestroyed: Renderer destroyed safely")
                        } catch (e: Exception) {
                            logE(TAG, "surfaceDestroyed: destroyRenderer/destroyVulkan failed", e)
                        }

                        isPausedBySurface = false
                    }
                })
            }

            val params = WindowManager.LayoutParams(
                screenWidth,
                screenHeight,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                        WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.RGBA_8888
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                x = 0; y = 0
            }

            renderSurfaceParams = params
            windowManager.addView(renderSurface, params)

        } catch (e: Exception) {
            logE(TAG, "addRenderSurface: Exception", e)
            throw e
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // clearRenderedOverlay helper
    // ──────────────────────────────────────────────────────────────────────────

    private fun clearRenderedOverlay() {
        // No-op if we are not in pipeline mode — indirect mode does not use
        // the Vulkan renderer and must never call renderOverlay().
        if (screenMode != ScreenMode.PIPELINE) return
        try {
            if (surfaceReady) {
                val blankBuffer = ByteBuffer.allocateDirect(captureWidth * captureHeight * 4)
                QeightJNI.processFrame(blankBuffer)
                QeightJNI.renderOverlay()
                logD(TAG, "clearRenderedOverlay: blank frame pushed")
            } else {
                logD(TAG, "clearRenderedOverlay: surface not ready, skipping")
            }
        } catch (e: Exception) {
            logW(TAG, "clearRenderedOverlay: failed (non-fatal)", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // togglePause helper
    // ──────────────────────────────────────────────────────────────────────────

    private fun togglePause() {
        // Pause/resume only makes sense for the pipeline.
        if (screenMode != ScreenMode.PIPELINE) return
        try {
            if (!isPausedByUser) {
                isPausedByUser = true
                screenCaptureManager?.pauseCapture()
                clearRenderedOverlay()
                logD(TAG, "togglePause: paused and overlay cleared")
            } else {
                screenCaptureManager?.resumeCapture()
                isPausedByUser = false
                logD(TAG, "togglePause: resumed")
            }
            val pauseText = if (isPausedByUser) "▶" else "⏸"
            clusterPauseButton?.text = pauseText
        } catch (e: Exception) {
            logE(TAG, "togglePause: Exception", e)
            Toast.makeText(this, "Pause/Resume failed: ${e.message}", Toast.LENGTH_SHORT).show()
            isPausedByUser = !isPausedByUser
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Cluster pause / resume helpers
    // ──────────────────────────────────────────────────────────────────────────

    private fun pauseForCluster() {
        if (screenMode != ScreenMode.PIPELINE) return
        if (isPausedByCluster) return
        isPausedByCluster = true
        if (!isPausedByUser) {
            screenCaptureManager?.pauseCapture()
        }
        clearRenderedOverlay()
        logD(TAG, "pauseForCluster: pipeline paused while cluster is open")
    }

    private fun resumeAfterCluster() {
        if (screenMode != ScreenMode.PIPELINE) return
        if (!isPausedByCluster) return
        isPausedByCluster = false
        if (!isPausedByUser) {
            screenCaptureManager?.resumeCapture()
            logD(TAG, "resumeAfterCluster: pipeline resumed after cluster collapsed")
        } else {
            logD(TAG, "resumeAfterCluster: user-pause still active, not resuming")
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Indirect mode launcher
    // ──────────────────────────────────────────────────────────────────────────

    private fun launchIndirectMode() {
        // ── 1. Collapse cluster and close settings panel ──────────────────────
        // Both must be forced closed before indirect mode starts. The panel can
        // remain open and interactive (wired to live pipeline setters) even when
        // the cluster collapses — this is a real interference path that we close
        // explicitly here rather than relying on showPanel()'s guard alone.
        if (clusterExpanded) collapseCluster()
        if (isPanelVisible) hidePanel()

        // Clear whatever the pipeline last rendered so it doesn't sit frozen
        // on screen while the pipeline is stopped. Must happen before stopping
        // the capture loop because clearRenderedOverlay() calls renderOverlay().
        clearRenderedOverlay()

        // ── 2. Stop the pipeline's consuming loop, but keep the shared capture
        //       manager (and its single VirtualDisplay) alive ──────────────────
        // We must NOT call screenCaptureManager.stop() here — that releases
        // the VirtualDisplay, and Android 14+ forbids calling
        // MediaProjection#createVirtualDisplay() a second time on the same
        // token (see step 6 below). Since pipeline and indirect mode are
        // mutually exclusive, we just stop the pipeline's frame-consuming
        // loop thread and pause the shared manager; indirect mode resumes it
        // and starts its own loop against the very same VirtualDisplay.
        stopCaptureLoop()
        screenCaptureManager?.pauseCapture()

        // ── 3. Set native gate to INDIRECT ────────────────────────────────────
        screenMode = ScreenMode.INDIRECT
        QeightJNI.setScreenMode(1)          // 1 == INDIRECT

        // ── 4. Hide the floating button cluster ───────────────────────────────
        floatingButton?.visibility = View.GONE

        // ── 5. Re-derive capture geometry fresh so indirect mode never starts
        //       from stale field values (e.g. post-rotation) ──────────────────
        readScreenDimensions()
        val freshCaptureWidth  = captureWidth
        val freshCaptureHeight = captureHeight
        val freshCaptureScale  = captureScale

        // ── 6. Build and start indirect capture manager + loop ────────────────
        val projection = mediaProjection ?: run {
            logE(TAG, "launchIndirectMode: mediaProjection is null, cannot start")
            // Roll back to PIPELINE mode cleanly.
            screenMode = ScreenMode.PIPELINE
            QeightJNI.setScreenMode(0)
            floatingButton?.visibility = View.VISIBLE
            startScreenCapture()
            return
        }

        logD(TAG, "launchIndirectMode: step6 — reusing shared screenCaptureManager " +
                "(w=$freshCaptureWidth h=$freshCaptureHeight scale=$freshCaptureScale)")

        val manager = screenCaptureManager
        if (manager == null) {
            // First-ever capture start (app just launched straight into
            // indirect mode with no prior pipeline run) — construct the one
            // and only ScreenCaptureManager for this MediaProjection's
            // lifetime. Every subsequent mode switch reuses this instance.
            screenCaptureManager = ScreenCaptureManager(
                context             = this,
                mediaProjection     = projection,
                screenWidth         = freshCaptureWidth,
                screenHeight        = freshCaptureHeight,
                onProjectionStopped = {
                    logW(TAG, "screenCaptureManager: MediaProjection stopped")
                    mainHandler.post { stopSelf() }
                }
            )
            screenCaptureManager?.start()
            logD(TAG, "launchIndirectMode: step6 — first-ever ScreenCaptureManager constructed + started")
        } else {
            // Reuse the existing VirtualDisplay — resize in place if the
            // geometry changed (e.g. rotation happened while paused) rather
            // than releasing/recreating it, then resume frame delivery.
            if (manager.captureWidth != freshCaptureWidth || manager.captureHeight != freshCaptureHeight) {
                manager.handleRotation(screenWidth, screenHeight)
            }
            manager.resumeCapture()
            logD(TAG, "launchIndirectMode: step6 — shared screenCaptureManager resumed")
        }
        logD(TAG, "launchIndirectMode: step6 — screenCaptureManager ready")

        // ── 7. Construct IndirectModeController ───────────────────────────────
        if (indirectModeController == null) {
            logD(TAG, "launchIndirectMode: step7 — indirectModeController is null, constructing")
            val density      = resources.displayMetrics.density
            val buttonSizePx = (55 * density).toInt()
            val marginRight  = (4  * density).toInt()

            val poolJson = prefs.getString("pool_table_calibration_json", null)
            val (pLeft, pTop, pRight, pBottom) = if (poolJson != null) {
                try {
                    val obj = JSONObject(poolJson)
                    listOf(
                        obj.getInt("pool_x1"), obj.getInt("pool_y1"),
                        obj.getInt("pool_x2"), obj.getInt("pool_y2")
                    )
                } catch (e: Exception) {
                    listOf(
                        (469f  * screenWidth  / 2400f).toInt(),
                        (258f  * screenHeight / 1080f).toInt(),
                        (1929f * screenWidth  / 2400f).toInt(),
                        (983f  * screenHeight / 1080f).toInt()
                    )
                }
            } else {
                listOf(
                    (469f  * screenWidth  / 2400f).toInt(),
                    (258f  * screenHeight / 1080f).toInt(),
                    (1929f * screenWidth  / 2400f).toInt(),
                    (983f  * screenHeight / 1080f).toInt()
                )
            }

            val roiJson = prefs.getString(KEY_ROI_CALIBRATION, null)
            val (pR, pNs) = if (roiJson != null) {
                try {
                    val obj = JSONObject(roiJson)
                    Pair(
                        obj.optInt("pocket_r", 40),
                        obj.optInt("pocket_ns_shift", 30)
                    )
                } catch (e: Exception) { Pair(40, 30) }
            } else Pair(40, 30)

            logD(TAG, "launchIndirectMode: step7 — geometry resolved " +
                    "pool=($pLeft,$pTop,$pRight,$pBottom) pocketR=$pR pocketNs=$pNs " +
                    "buttonSizePx=$buttonSizePx marginRight=$marginRight " +
                    "nativeScreen=${screenWidth}x${screenHeight}")

            try {
                indirectModeController = IndirectModeController(
                    context                     = this@OverlayService,
                    windowManager               = windowManager,
                    // IndirectModeController and the pipeline now share the
                    // SAME ScreenCaptureManager instance (and its single
                    // VirtualDisplay) — see the class-level comment on
                    // screenCaptureManager for why. This is safe because
                    // pipeline and indirect mode are mutually exclusive and
                    // never consume frames concurrently.
                    screenCaptureManager        = screenCaptureManager!!,
                    captureWidth                = freshCaptureWidth,
                    captureHeight               = freshCaptureHeight,
                    captureScale                = freshCaptureScale,
                    floatingButtonSizePx        = buttonSizePx,
                    floatingButtonRightMarginPx = marginRight,
                    nativeScreenWidth           = screenWidth,
                    nativeScreenHeight          = screenHeight,
                    poolTableLeft               = pLeft,
                    poolTableTop                = pTop,
                    poolTableRight              = pRight,
                    poolTableBottom             = pBottom,
                    poolPocketR                 = pR,
                    poolPocketNsShift           = pNs,
                    onExit                      = { onIndirectModeExit() }
                )
                logD(TAG, "launchIndirectMode: step7 — IndirectModeController constructed OK")
            } catch (t: Throwable) {
                logE(TAG, "launchIndirectMode: step7 — IndirectModeController constructor threw", t)
                throw t
            }
        } else {
            logD(TAG, "launchIndirectMode: step7 — indirectModeController already exists, reusing")
        }

        try {
            logD(TAG, "launchIndirectMode: step8 — calling start() + setMagnifierZoom()")
            // start() first: it builds magnifierView (via
            // buildAllPersistentViews() -> buildMagnifierPanel()) if it
            // doesn't exist yet. Calling setMagnifierZoom() afterward
            // guarantees magnifierView is non-null so the zoom is applied
            // directly, rather than relying solely on
            // IndirectModeController's pendingMagnifierZoom fallback.
            indirectModeController?.start()
            indirectModeController?.setMagnifierZoom(indirectMagnifierZoom.toFloat())
            logD(TAG, "launchIndirectMode: step8 — indirectModeController.start() returned OK")
        } catch (t: Throwable) {
            logE(TAG, "launchIndirectMode: step8 — indirectModeController.start() threw", t)
            throw t
        }

        // ── 8. Start the indirect capture loop ────────────────────────────────
        logD(TAG, "launchIndirectMode: step9 — calling startIndirectLoop()")
        startIndirectLoop()
        logD(TAG, "launchIndirectMode: step9 — startIndirectLoop() returned OK")

        logD(TAG, "launchIndirectMode: indirect mode active")
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Indirect mode exit callback
    // ──────────────────────────────────────────────────────────────────────────

    private fun onIndirectModeExit() {
        logD(TAG, "onIndirectModeExit(): called. Caller stack:\n" +
                Thread.currentThread().stackTrace
                    .drop(1)
                    .take(12)
                    .joinToString("\n") { "    at $it" })
        // ── 1. Stop only the indirect mode's consuming loop ────────────────────
        // IMPORTANT: pipeline and indirect mode share a single
        // ScreenCaptureManager instance now (see class-level comment on
        // screenCaptureManager). There is no separate "indirectCaptureManager"
        // — that field was a leftover from before the shared-manager refactor
        // and was always null, but calling stop()/rebuilding a fresh
        // ScreenCaptureManager below (as this function used to do) tore down
        // the *live* shared capture session (VirtualDisplay/Surface) out from
        // under the pipeline and silently started a second, orphaned one in
        // its place. That's what was causing capture to effectively "stop"
        // (and re-prompt) on exiting indirect mode: we must stop the INDIRECT
        // consumer only, never the underlying capture itself.
        stopIndirectLoop()

        // ── 2. Destroy native indirect solver and set mode back to PIPELINE ───
        QeightJNI.destroyIndirectSolver()
        screenMode = ScreenMode.PIPELINE
        QeightJNI.setScreenMode(0)          // 0 == PIPELINE

        // ── 3. Null controller and restore UI ────────────────────────────────
        indirectModeController = null
        floatingButton?.visibility = View.VISIBLE

        // ── 4. Resume the shared capture manager and restart the pipeline's
        // consuming loop. The capture session itself (VirtualDisplay/Surface)
        // was only paused in launchIndirectMode() via pauseCapture() — it was
        // never stopped — so we simply resume it here rather than
        // constructing a brand-new ScreenCaptureManager. This avoids
        // duplicate capture sessions and avoids re-triggering the
        // MediaProjection permission/capture prompt.
        if (screenCaptureManager == null) {
            logE(TAG, "onIndirectModeExit: screenCaptureManager is null, cannot resume pipeline")
            return
        }
        screenCaptureManager?.resumeCapture()
        startCaptureLoop()

        logD(TAG, "onIndirectModeExit: pipeline capture resumed (shared manager)")
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Cluster expand / collapse with window resizing
    // ──────────────────────────────────────────────────────────────────────────

    private fun expandCluster() {
        try {
            val density      = resources.displayMetrics.density
            val buttonSizePx = (55 * density).toInt()
            val marginRight  = (4  * density).toInt()
            val gapPx        = (20 * density).toInt()

            val canvasW       = buttonSizePx + gapPx + buttonSizePx
            val canvasH       = buttonSizePx + gapPx + buttonSizePx + gapPx + buttonSizePx
            val eightBallLeft = buttonSizePx + gapPx
            val eightBallTop  = buttonSizePx + gapPx

            fun frameParams(left: Int, top: Int): android.widget.FrameLayout.LayoutParams =
                android.widget.FrameLayout.LayoutParams(buttonSizePx, buttonSizePx).apply {
                    leftMargin = left
                    topMargin  = top
                }

            clusterPauseButton?.layoutParams    = frameParams(eightBallLeft, 0)
            clusterSettingsButton?.layoutParams = frameParams(0, eightBallTop)
            clusterIndirectButton?.layoutParams = frameParams(0, eightBallTop + buttonSizePx + gapPx)
            clusterStopButton?.layoutParams     = frameParams(eightBallLeft, eightBallTop + buttonSizePx + gapPx)
            eightBallButton?.layoutParams       = frameParams(eightBallLeft, eightBallTop)

            val oldEightBallScreenX = floatingButtonParams!!.x + buttonSizePx / 2
            val oldEightBallScreenY = floatingButtonParams!!.y + buttonSizePx / 2

            val newX = oldEightBallScreenX - eightBallLeft - buttonSizePx / 2
            val newY = oldEightBallScreenY - eightBallTop  - buttonSizePx / 2

            floatingButtonParams!!.width  = canvasW
            floatingButtonParams!!.height = canvasH
            floatingButtonParams!!.x = newX.coerceIn(0, screenWidth  - canvasW)
            floatingButtonParams!!.y = newY.coerceIn(0, screenHeight - canvasH)

            windowManager.updateViewLayout(floatingButton, floatingButtonParams)

            clusterSettingsButton?.visibility = View.VISIBLE
            clusterPauseButton?.visibility    = View.VISIBLE
            clusterStopButton?.visibility     = View.VISIBLE
            clusterIndirectButton?.visibility = View.VISIBLE
            clusterExpanded = true

            pauseForCluster()

            mainHandler.removeCallbacks(autoCollapseRunnable)
            mainHandler.postDelayed(autoCollapseRunnable, 3000L)

        } catch (e: Exception) {
            logE(TAG, "expandCluster: Exception", e)
        }
    }

    private fun collapseCluster() {
        try {
            val density      = resources.displayMetrics.density
            val buttonSizePx = (55 * density).toInt()
            @Suppress("UNUSED_VARIABLE")
            val marginRight  = (4  * density).toInt()
            val gapPx        = (20 * density).toInt()

            val eightBallLeft = buttonSizePx + gapPx
            val eightBallTop  = buttonSizePx + gapPx

            val oldEightBallScreenX = floatingButtonParams!!.x + eightBallLeft + buttonSizePx / 2
            val oldEightBallScreenY = floatingButtonParams!!.y + eightBallTop  + buttonSizePx / 2

            clusterSettingsButton?.visibility = View.GONE
            clusterPauseButton?.visibility    = View.GONE
            clusterStopButton?.visibility     = View.GONE
            clusterIndirectButton?.visibility = View.GONE
            clusterExpanded = false

            fun frameParams(left: Int, top: Int): android.widget.FrameLayout.LayoutParams =
                android.widget.FrameLayout.LayoutParams(buttonSizePx, buttonSizePx).apply {
                    leftMargin = left
                    topMargin  = top
                }

            eightBallButton?.layoutParams = frameParams(0, 0)

            val newX = oldEightBallScreenX - buttonSizePx / 2
            val newY = oldEightBallScreenY - buttonSizePx / 2

            floatingButtonParams!!.width  = buttonSizePx
            floatingButtonParams!!.height = buttonSizePx
            floatingButtonParams!!.x = newX.coerceIn(0, screenWidth  - buttonSizePx)
            floatingButtonParams!!.y = newY.coerceIn(0, screenHeight - buttonSizePx)

            windowManager.updateViewLayout(floatingButton, floatingButtonParams)

            mainHandler.removeCallbacks(autoCollapseRunnable)

            resumeAfterCluster()

        } catch (e: Exception) {
            logE(TAG, "collapseCluster: Exception", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Floating button cluster layout with dynamic window sizing
    // ──────────────────────────────────────────────────────────────────────────

    private fun addFloatingButton() {
        try {
            val density      = resources.displayMetrics.density
            val buttonSizePx = (55 * density).toInt()
            val marginRight  = (4  * density).toInt()
            val gapPx        = (20 * density).toInt()

            fun makeOvalButton(label: String): Button {
                val gd = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.OVAL
                    setColor(Color.BLACK)
                }
                return Button(this).apply {
                    text       = label
                    textSize   = 18f
                    typeface   = android.graphics.Typeface.create(
                        android.graphics.Typeface.DEFAULT,
                        android.graphics.Typeface.BOLD
                    )
                    setTextColor(Color.parseColor("#FF6D00"))
                    background = gd
                    gravity    = Gravity.CENTER
                    setPadding(0, 0, 0, 0)
                    isClickable = true
                    isFocusable = true
                }
            }

            val gd8 = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                setColor(Color.BLACK)
            }
            val eightBallBtn = Button(this).apply {
                text       = "8"
                textSize   = 22f
                typeface   = android.graphics.Typeface.create(
                    android.graphics.Typeface.DEFAULT,
                    android.graphics.Typeface.BOLD
                )
                setTextColor(Color.parseColor("#FF6D00"))
                background = gd8
                gravity    = Gravity.CENTER
                setPadding(0, 0, 0, 0)
                isClickable = true
                isFocusable = true
            }
            eightBallButton = eightBallBtn

            val settingsBtn = makeOvalButton("⚙").apply {
                visibility = View.GONE
                setOnClickListener {
                    collapseCluster()
                    if (isPanelVisible) hidePanel() else showPanel()
                }
            }
            clusterSettingsButton = settingsBtn

            val pauseBtn = makeOvalButton(if (isPausedByUser) "▶" else "⏸").apply {
                visibility = View.GONE
                setOnClickListener {
                    collapseCluster()
                    togglePause()
                }
            }
            clusterPauseButton = pauseBtn

            val stopBtn = makeOvalButton("⏹").apply {
                visibility = View.GONE
                setOnClickListener {
                    // Mode-gate: tile/cluster stop cannot fire during indirect mode.
                    if (screenMode != ScreenMode.PIPELINE) return@setOnClickListener
                    collapseCluster()
                    stopSelf()
                }
            }
            clusterStopButton = stopBtn

            val indirectBtn = makeOvalButton("I").apply {
                visibility = View.GONE
                setOnClickListener {
                    collapseCluster()
                    launchIndirectMode()
                }
            }
            clusterIndirectButton = indirectBtn

            val container = object : android.widget.FrameLayout(this) {
                override fun onTouchEvent(event: MotionEvent): Boolean = false
            }.apply {
                isClickable = false
                isFocusable = false
                setBackgroundColor(Color.TRANSPARENT)
            }

            fun frameParams(left: Int, top: Int): android.widget.FrameLayout.LayoutParams =
                android.widget.FrameLayout.LayoutParams(buttonSizePx, buttonSizePx).apply {
                    leftMargin = left
                    topMargin  = top
                }

            eightBallBtn.layoutParams = frameParams(0, 0)
            pauseBtn.layoutParams     = frameParams(0, 0)
            stopBtn.layoutParams      = frameParams(0, 0)
            settingsBtn.layoutParams  = frameParams(0, 0)
            indirectBtn.layoutParams  = frameParams(0, 0)

            container.addView(pauseBtn)
            container.addView(settingsBtn)
            container.addView(stopBtn)
            container.addView(indirectBtn)
            container.addView(eightBallBtn)

            val collapsedW = buttonSizePx
            val collapsedH = buttonSizePx
            val initX = screenWidth  - marginRight - collapsedW
            val initY = screenHeight / 2 - collapsedH / 2

            val params = WindowManager.LayoutParams(
                collapsedW,
                collapsedH,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                x = initX
                y = initY
            }

            floatingButtonParams = params

            var touchStartX = 0f
            var touchStartY = 0f
            var lastTouchY  = 0f
            var isDragging  = false
            val tapSlop     = 8 * density

            eightBallBtn.setOnTouchListener { view, event ->
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        touchStartX = event.rawX
                        touchStartY = event.rawY
                        lastTouchY  = event.rawY
                        isDragging  = false
                        view.parent.requestDisallowInterceptTouchEvent(true)
                        true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        if (!isDragging &&
                            (Math.abs(event.rawX - touchStartX) > tapSlop ||
                                    Math.abs(event.rawY - touchStartY) > tapSlop)
                        ) {
                            isDragging = true
                        }
                        if (isDragging) {
                            if (clusterExpanded) collapseCluster()

                            val currentH = floatingButtonParams!!.height
                            val newY = (floatingButtonParams!!.y +
                                    (event.rawY - lastTouchY)).toInt()
                                .coerceIn(0, screenHeight - currentH)
                            floatingButtonParams!!.y = newY
                            val currentW = floatingButtonParams!!.width
                            floatingButtonParams!!.x = screenWidth - marginRight - currentW
                            lastTouchY = event.rawY
                            try {
                                windowManager.updateViewLayout(floatingButton, floatingButtonParams)
                                if (isPanelVisible) updatePanelPosition()
                            } catch (e: Exception) {
                                logE(TAG, "FloatingButton drag: updateViewLayout error", e)
                            }
                        }
                        true
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        view.parent.requestDisallowInterceptTouchEvent(false)
                        if (!isDragging) {
                            if (isPanelVisible) {
                                hidePanel()
                            } else {
                                if (clusterExpanded) collapseCluster() else expandCluster()
                            }
                        }
                        isDragging = false
                        true
                    }
                    else -> false
                }
            }

            floatingButton = container
            windowManager.addView(container, params)

        } catch (e: Exception) {
            logE(TAG, "addFloatingButton: Exception", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Panel show / hide / position helpers
    // ──────────────────────────────────────────────────────────────────────────

    private fun togglePanel() {
        if (isPanelVisible) hidePanel() else showPanel()
    }

    private fun showPanel() {
        // Direct fix for the panel-open gap: the panel must not open while
        // indirect mode is active. The forced-close in launchIndirectMode()
        // covers the entry path; this guard covers any code path that calls
        // showPanel() independently (e.g. a future feature, tile action, etc.).
        if (screenMode == ScreenMode.INDIRECT) {
            logD(TAG, "showPanel: suppressed — screenMode is INDIRECT")
            return
        }
        try {
            isPausedByPanel = true

            if (floatingPanel == null) buildPanel()
            val params = buildPanelLayoutParams()
            floatingPanelParams = params
            if (floatingPanel != null && floatingPanel?.parent == null) {
                windowManager.addView(floatingPanel, params)
            }
            isPanelVisible = true
        } catch (e: Exception) {
            logE(TAG, "showPanel: Exception", e)
        }
    }

    private fun hidePanel() {
        try {
            if (floatingPanel?.parent != null) windowManager.removeView(floatingPanel)
            isPanelVisible  = false
            isPausedByPanel = false
        } catch (e: Exception) {
            logE(TAG, "hidePanel: Exception", e)
        }
    }

    private fun updatePanelPosition() {
        try {
            if (isPanelVisible && floatingPanel?.parent != null) {
                windowManager.updateViewLayout(floatingPanel, buildPanelLayoutParams())
            }
        } catch (e: Exception) {
            logE(TAG, "updatePanelPosition: Exception", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // buildPanelLayoutParams
    // ──────────────────────────────────────────────────────────────────────────

    private fun buildPanelLayoutParams(): WindowManager.LayoutParams {
        val density      = resources.displayMetrics.density
        val panelWidth   = (screenWidth * 0.26f).toInt()
            .coerceAtMost((240 * density).toInt())
        val panelHeight  = (screenHeight * 0.60f).toInt()
        val buttonSizePx = (55 * density).toInt()
        val marginRight  = (4  * density).toInt()
        val buttonX = floatingButtonParams?.x ?: (screenWidth  - buttonSizePx - marginRight)
        val buttonY = floatingButtonParams?.y ?: (screenHeight / 2 - buttonSizePx / 2)

        return WindowManager.LayoutParams(
            panelWidth,
            panelHeight,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (buttonX - panelWidth - (4 * density).toInt())
                .coerceIn(0, screenWidth - panelWidth)
            y = buttonY.coerceIn(0, screenHeight - panelHeight)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // buildPanel
    // ──────────────────────────────────────────────────────────────────────────

    private fun buildPanel() {
        val density = resources.displayMetrics.density
        val padding = (12 * density).toInt()

        val themeContext = ContextThemeWrapper(this, com.ashraf.qeight.R.style.Theme_Qeight)

        val contentLayout = LinearLayout(themeContext).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
        }

        contentLayout.addView(TextView(themeContext).apply {
            text = "QEIGHT"
            setTextColor(Color.parseColor("#FF6D00"))
            textSize = 13f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, (12 * density).toInt())
        })

        contentLayout.addView(Button(themeContext).apply {
            text = "⊞  CALIBRATE ROI"
            setTextColor(Color.BLACK)
            setBackgroundResource(R.drawable.btn_orange_filled)
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (44 * density).toInt()
            ).apply { bottomMargin = (12 * density).toInt() }
            setOnClickListener {
                hidePanel()
                launchCalibration()
            }
        })

        contentLayout.addView(makeDivider(themeContext, density))

        contentLayout.addView(TextView(themeContext).apply {
            text = "SETTINGS"
            setTextColor(Color.WHITE)
            textSize = 10f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, (8 * density).toInt())
        })

        addSpinnerRow(contentLayout, themeContext, "CBC Reflections", cbcReflections, 0, 8) { v ->
            cbcReflections = v
            try { QeightJNI.setCbcReflections(v) } catch (e: Exception) {
                logE(TAG, "setCbcReflections JNI failed", e)
            }
        }

        addSpinnerRow(contentLayout, themeContext, "TGT Reflections", tgtReflections, 0, 8) { v ->
            tgtReflections = v
            try { QeightJNI.setTgtReflections(v) } catch (e: Exception) {
                logE(TAG, "setTgtReflections JNI failed", e)
            }
        }

        addSpinnerRow(contentLayout, themeContext, "Line Thickness", lineThickness, 1, 8) { v ->
            lineThickness = v
            try { QeightJNI.setLineThickness(v) } catch (e: Exception) {
                logE(TAG, "setLineThickness JNI failed", e)
            }
        }

        addSpinnerRow(contentLayout, themeContext, "Cue Force", cueForceStat, 0, 16) { v ->
            cueForceStat = v
            try { QeightJNI.setCueForce(v) } catch (e: Exception) {
                logE(TAG, "setCueForce JNI failed", e)
            }
        }

        addSpinnerRow(contentLayout, themeContext, "Cue Spin", cueSpinStat, 0, 16) { v ->
            cueSpinStat = v
            try { QeightJNI.setCueSpin(v) } catch (e: Exception) {
                logE(TAG, "setCueSpin JNI failed", e)
            }
        }

        addSpinnerRow(contentLayout, themeContext, "Magnifier Zoom", indirectMagnifierZoom, 1, 15) { v ->
            indirectMagnifierZoom = v
            indirectModeController?.setMagnifierZoom(v.toFloat())
        }

        contentLayout.addView(TextView(themeContext).apply {
            text = "Trajectory Power"
            setTextColor(Color.WHITE)
            textSize = 11f
            setPadding(0, (6 * density).toInt(), 0, (2 * density).toInt())
        })
        val powerRow = LinearLayout(themeContext).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 0, 0, (4 * density).toInt())
        }
        val powerOptions = intArrayOf(50, 75, 95, 100)
        powerOptions.forEachIndexed { index, pct ->
            val btn = Button(themeContext).apply {
                text = "$pct%"
                textSize = 9f
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
                ).apply { if (index < powerOptions.size - 1) marginEnd = (4 * density).toInt() }
                setOnClickListener {
                    trajectoryPowerPct = pct
                    updateTrajectoryPowerButtonColors()
                    try { QeightJNI.setTrajectoryPower(pct) } catch (e: Exception) {
                        logE(TAG, "setTrajectoryPower JNI failed", e)
                    }
                }
            }
            trajectoryPowerButtons[index] = btn
            powerRow.addView(btn)
        }
        contentLayout.addView(powerRow)
        updateTrajectoryPowerButtonColors()

        val fillRow = LinearLayout(themeContext).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
        }
        fillRow.addView(TextView(themeContext).apply {
            text = "Guide Band Fill"
            setTextColor(Color.WHITE)
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        fillRow.addView(SwitchCompat(themeContext).apply {
            isChecked = parallelFillEnabled
            setOnCheckedChangeListener { _, checked ->
                parallelFillEnabled = checked
                try {
                    QeightJNI.setParallelLinesFill(parallelFillEnabled, parallelFillAlpha)
                    QeightJNI.setParallelLinesVisible(!parallelFillEnabled)
                } catch (e: Exception) {
                    logE(TAG, "setParallelLinesFill/Visible JNI failed", e)
                }
            }
        })
        contentLayout.addView(fillRow)

        addSpinnerRow(
            contentLayout, themeContext, "Fill Opacity %",
            (parallelFillAlpha * 100).roundToInt(), 0, 100
        ) { v ->
            parallelFillAlpha = v / 100f
            try {
                QeightJNI.setParallelLinesFill(parallelFillEnabled, parallelFillAlpha)
            } catch (e: Exception) {
                logE(TAG, "setParallelLinesFill JNI failed", e)
            }
        }

        val cushionRow = LinearLayout(themeContext).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
        }
        cushionRow.addView(TextView(themeContext).apply {
            text = "Cushion Shots"
            setTextColor(Color.WHITE)
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        cushionRow.addView(SwitchCompat(themeContext).apply {
            isChecked = cushionShotsEnabled
            setOnCheckedChangeListener { _, checked ->
                cushionShotsEnabled = checked
                try { QeightJNI.setCushionShots(checked) } catch (e: Exception) {
                    logE(TAG, "setCushionShots JNI failed", e)
                }
            }
        })
        contentLayout.addView(cushionRow)

        val colorRow = LinearLayout(themeContext).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (6 * density).toInt(), 0, (6 * density).toInt())
        }
        colorRow.addView(TextView(themeContext).apply {
            text = "Overlay Color"
            setTextColor(Color.WHITE)
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        val colorBox = View(themeContext).apply {
            layoutParams = LinearLayout.LayoutParams(
                (30 * density).toInt(), (30 * density).toInt())
            setBackgroundColor(Color.rgb(overlayColorR, overlayColorG, overlayColorB))
        }
        colorBox.setOnClickListener { showColorPicker(colorBox) }
        colorRow.addView(colorBox)
        contentLayout.addView(colorRow)

        contentLayout.addView(makeDivider(themeContext, density, topMargin = 8, bottomMargin = 8))

        val dragBar = LinearLayout(themeContext).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#FF6D00"))
            setPadding(0, (6 * density).toInt(), 0, (6 * density).toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (28 * density).toInt()
            )
            addView(TextView(themeContext).apply {
                text = "⠿  QEIGHT  ⠿"
                setTextColor(Color.WHITE)
                textSize = 11f
                gravity = Gravity.CENTER
            })
        }

        val scrollView = ScrollView(themeContext).apply {
            addView(contentLayout)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f)
        }

        val outerLayout = LinearLayout(themeContext).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.panel_background)
        }
        outerLayout.addView(dragBar)
        outerLayout.addView(scrollView)

        var dragStartRawX   = 0f
        var dragStartRawY   = 0f
        var dragStartParamX = 0
        var dragStartParamY = 0
        val densityLocal = resources.displayMetrics.density

        dragBar.setOnTouchListener { _, event ->
            val panelW = (screenWidth * 0.26f).toInt()
                .coerceAtMost((240 * densityLocal).toInt())
            val panelH = (screenHeight * 0.60f).toInt()

            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    dragStartRawX   = event.rawX
                    dragStartRawY   = event.rawY
                    dragStartParamX = floatingPanelParams?.x ?: 0
                    dragStartParamY = floatingPanelParams?.y ?: 0
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - dragStartRawX).toInt()
                    val dy = (event.rawY - dragStartRawY).toInt()
                    floatingPanelParams?.x =
                        (dragStartParamX + dx).coerceIn(0, screenWidth  - panelW)
                    floatingPanelParams?.y =
                        (dragStartParamY + dy).coerceIn(0, screenHeight - panelH)
                    try {
                        if (outerLayout.parent != null)
                            windowManager.updateViewLayout(outerLayout, floatingPanelParams)
                    } catch (e: Exception) { /* ignore during layout transitions */ }
                    true
                }
                else -> false
            }
        }

        floatingPanel = outerLayout
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Divider factory helper
    // ──────────────────────────────────────────────────────────────────────────

    private fun makeDivider(
        ctx: Context,
        density: Float,
        topMargin: Int = 4,
        bottomMargin: Int = 8
    ): View = View(ctx).apply {
        setBackgroundResource(R.drawable.divider_dark)
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 1
        ).apply {
            this.topMargin    = (topMargin    * density).toInt()
            this.bottomMargin = (bottomMargin * density).toInt()
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Calibration
    // ──────────────────────────────────────────────────────────────────────────

    private fun launchCalibration() {
        // Explicit guard — calibration is only normally reachable via the panel
        // (which showPanel() already blocks in INDIRECT mode), but guard it here
        // too since it is in principle callable from anywhere in this file.
        if (screenMode == ScreenMode.INDIRECT) {
            logD(TAG, "launchCalibration: suppressed — screenMode is INDIRECT")
            return
        }
        try {
            isPausedByPanel = true

            if (surfaceReady) {
                try { QeightJNI.setCalibrationMode(true) } catch (e: Exception) {
                    logW(TAG, "launchCalibration: setCalibrationMode unavailable", e)
                }
            }

            val calibDoneReceiver = object : BroadcastReceiver() {
                override fun onReceive(ctx: Context, intent: Intent) {
                    LocalBroadcastManager.getInstance(ctx).unregisterReceiver(this)

                    isPausedByPanel = false

                    if (surfaceReady) {
                        try { QeightJNI.setCalibrationMode(false) } catch (_: Exception) {}
                    }

                    if (intent.action == "com.ashraf.qeight.CALIBRATION_SAVED") {
                        reloadPoolTableBounds()
                    }
                }
            }

            LocalBroadcastManager.getInstance(this).registerReceiver(
                calibDoneReceiver,
                IntentFilter().apply {
                    addAction("com.ashraf.qeight.CALIBRATION_SAVED")
                    addAction("com.ashraf.qeight.CALIBRATION_DISMISSED")
                }
            )

            calibrationManager = CalibrationManager(this, prefs).also { it.startCalibration() }
            logD(TAG, "launchCalibration: CalibrationManager started")
        } catch (e: Exception) {
            logE(TAG, "launchCalibration: Exception", e)
            Toast.makeText(this, "Calibration failed: ${e.message}", Toast.LENGTH_SHORT).show()
            isPausedByPanel = false
            if (surfaceReady) {
                try { QeightJNI.setCalibrationMode(false) } catch (_: Exception) {}
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Spinner row helper
    // ──────────────────────────────────────────────────────────────────────────

    private fun addSpinnerRow(
        parent: LinearLayout,
        ctx: Context,
        label: String,
        initial: Int,
        min: Int,
        max: Int,
        onChange: (Int) -> Unit
    ) {
        val density = resources.displayMetrics.density
        var current = initial

        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (3 * density).toInt(), 0, (3 * density).toInt())
        }

        row.addView(TextView(ctx).apply {
            text = label
            setTextColor(Color.WHITE)
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })

        val valueView = TextView(ctx).apply {
            text = initial.toString()
            setTextColor(Color.WHITE)
            textSize = 12f
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                (32 * density).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        val btnMinus = Button(ctx).apply {
            text = "−"; textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(
                (30 * density).toInt(), (30 * density).toInt())
            setPadding(0, 0, 0, 0)
            setOnClickListener {
                if (current > min) { current--; valueView.text = current.toString(); onChange(current) }
            }
        }

        val btnPlus = Button(ctx).apply {
            text = "+"; textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(
                (30 * density).toInt(), (30 * density).toInt())
            setPadding(0, 0, 0, 0)
            setOnClickListener {
                if (current < max) { current++; valueView.text = current.toString(); onChange(current) }
            }
        }

        row.addView(btnMinus)
        row.addView(valueView)
        row.addView(btnPlus)
        parent.addView(row)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Trajectory power button highlight helper
    // ──────────────────────────────────────────────────────────────────────────

    private fun updateTrajectoryPowerButtonColors() {
        val active   = Color.parseColor("#FF6D00")
        val inactive = Color.parseColor("#444444")
        val powerOptions = intArrayOf(50, 75, 95, 100)
        trajectoryPowerButtons.forEachIndexed { index, btn ->
            btn?.setBackgroundColor(
                if (powerOptions[index] == trajectoryPowerPct) active else inactive)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Color picker grid
    // ──────────────────────────────────────────────────────────────────────────

    private fun showColorPicker(colorBoxView: View) {
        try {
            val density = resources.displayMetrics.density

            val colors = listOf(
                "#FFFFFF" to "White",       "#000000" to "Black",
                "#FF0000" to "Red",         "#00FF00" to "Lime",
                "#0000FF" to "Blue",        "#FFFF00" to "Yellow",
                "#FF00FF" to "Magenta",     "#00FFFF" to "Cyan",
                "#FF8000" to "Orange",      "#8000FF" to "Purple",
                "#FF1493" to "Deep Pink",   "#00FF80" to "Spring Green",
                "#1E90FF" to "Dodger Blue", "#FFD700" to "Gold",
                "#40E0D0" to "Turquoise",   "#9400D3" to "Violet",
                "#7CFC00" to "Lawn Green",  "#FF6347" to "Tomato",
                "#00BFFF" to "Deep Sky Blue","#FF69B4" to "Hot Pink",
                "#ADFF2F" to "Chartreuse",  "#FF8C00" to "Dark Orange",
                "#8B0000" to "Dark Red",    "#006400" to "Dark Green",
                "#00008B" to "Dark Blue",   "#FFC0CB" to "Light Pink",
                "#98FB98" to "Pale Green",  "#87CEFA" to "Light Sky Blue",
                "#C0C0C0" to "Silver",      "#808080" to "Gray"
            )

            val dialogWidth = min(
                (320 * density).toInt(),
                (screenWidth * 0.45f).toInt()
            )

            val outerLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(Color.parseColor("#1A1A1A"))
            }

            fun dismiss() {
                try {
                    if (outerLayout.parent != null) windowManager.removeView(outerLayout)
                } catch (e: Exception) {
                    logE(TAG, "showColorPicker: dismiss failed", e)
                }
            }

            val titleRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(
                    (8 * density).toInt(), (8 * density).toInt(),
                    (8 * density).toInt(), (4 * density).toInt()
                )
            }
            titleRow.addView(TextView(this).apply {
                text = "Overlay Color"
                setTextColor(Color.WHITE)
                textSize = 16f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                layoutParams = LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
            titleRow.addView(Button(this).apply {
                text = "✕"
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.TRANSPARENT)
                textSize = 14f
                setPadding(0, 0, 0, 0)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
                setOnClickListener { dismiss() }
            })
            outerLayout.addView(titleRow)

            val gridLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
            }

            val columnsPerRow = 4
            var currentRow: LinearLayout? = null

            colors.forEachIndexed { index, (hex, name) ->
                if (index % columnsPerRow == 0) {
                    currentRow = LinearLayout(this).apply {
                        orientation = LinearLayout.HORIZONTAL
                    }
                    gridLayout.addView(currentRow)
                }

                val cell = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    minimumHeight = (48 * density).toInt()
                    layoutParams = LinearLayout.LayoutParams(
                        0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    isClickable = true
                    isFocusable = true
                    setOnClickListener {
                        val parsed = Color.parseColor(hex)
                        overlayColorR = Color.red(parsed)
                        overlayColorG = Color.green(parsed)
                        overlayColorB = Color.blue(parsed)
                        try {
                            QeightJNI.setOverlayColor(
                                overlayColorR, overlayColorG,
                                overlayColorB, overlayColorA
                            )
                        } catch (e: Exception) {
                            logE(TAG, "setOverlayColor JNI failed", e)
                        }
                        colorBoxView.setBackgroundColor(
                            Color.rgb(overlayColorR, overlayColorG, overlayColorB))
                        dismiss()
                    }
                }

                val nameView = TextView(this).apply {
                    text = name
                    textSize = 9f
                    setTextColor(Color.WHITE)
                    layoutParams = LinearLayout.LayoutParams(
                        0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        leftMargin = (4 * density).toInt()
                    }
                    setPadding((4 * density).toInt(), 0, 0, 0)
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }

                val swatchSize   = (36 * density).toInt()
                val swatchMargin = (2  * density).toInt()
                val swatch = View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(swatchSize, swatchSize).apply {
                        setMargins(swatchMargin, swatchMargin, swatchMargin, swatchMargin)
                    }
                    setBackgroundColor(Color.parseColor(hex))
                }

                cell.addView(nameView)
                cell.addView(swatch)
                currentRow!!.addView(cell)
            }

            val remainder = colors.size % columnsPerRow
            if (remainder != 0) {
                repeat(columnsPerRow - remainder) {
                    currentRow!!.addView(View(this).apply {
                        layoutParams = LinearLayout.LayoutParams(
                            0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    })
                }
            }

            val scrollView = ScrollView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    (screenHeight * 0.55f).toInt()
                )
                addView(gridLayout)
            }
            outerLayout.addView(scrollView)

            val wlp = WindowManager.LayoutParams(
                dialogWidth,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            ).apply { gravity = Gravity.CENTER }

            windowManager.addView(outerLayout, wlp)

        } catch (e: Exception) {
            logE(TAG, "showColorPicker: Exception", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Screen capture — pipeline only
    // ──────────────────────────────────────────────────────────────────────────

    private fun startScreenCapture() {
        try {
            val projection = mediaProjection ?: return
            screenCaptureManager = ScreenCaptureManager(
                context             = this,
                mediaProjection     = projection,
                screenWidth         = captureWidth,
                screenHeight        = captureHeight,
                onProjectionStopped = {
                    logW(TAG, "ScreenCaptureManager: MediaProjection stopped, stopping service")
                    mainHandler.post { stopSelf() }
                }
            )
            screenCaptureManager?.start()
            startCaptureLoop()
            logD(TAG, "startScreenCapture: ScreenCaptureManager started, loop launched")
        } catch (e: Exception) {
            logE(TAG, "startScreenCapture: Exception", e)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Notification helpers
    // ──────────────────────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Qeight Overlay",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            setSound(null, null)
            enableVibration(false)
            enableLights(false)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildForegroundNotification(): Notification {
        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, OverlayService::class.java).apply { action = "STOP_SERVICE" },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Qeight")
            .setContentText("Overlay active")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setSilent(true)
            .addAction(android.R.drawable.ic_delete, "Stop", stopIntent)
            .build()
    }
}