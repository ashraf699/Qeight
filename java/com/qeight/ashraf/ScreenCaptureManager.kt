package com.ashraf.qeight

import android.content.Context
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import java.nio.ByteBuffer
import java.util.concurrent.Semaphore

/**
 * ScreenCaptureManager — pull-based, single-frame-on-demand screen capture.
 *
 * Architecture:
 * - VirtualDisplay is created once and kept alive for the service lifetime.
 *   It is NEVER destroyed/recreated for pause/resume — doing so violates
 *   MediaProjection's single-VirtualDisplay-per-instance constraint on Android 14+.
 * - Pause/resume is implemented purely via the [isPaused] flag: when paused,
 *   acquireFrame() returns null immediately without touching the ImageReader.
 *   The VirtualDisplay continues to run but frames are ignored.
 * - drainStaleFrames() discards accumulated frames and semaphore permits.
 * - acquireFrame() blocks until a frame arrives, then returns its buffer.
 * - releaseFrame() closes the Image and returns it to the pool.
 *
 * Capture resolution:
 * - Downscaled to ~720px height via captureScale.
 * - On rotation, handleRotation() recreates the pipeline at new dimensions.
 *
 * Recovery:
 * - onProjectionStopped fires when MediaProjection is revoked by the system.
 *
 * Thread safety:
 * - drainStaleFrames(), acquireFrame(), releaseFrame() are called from the
 *   single "QeightLoop" thread.
 * - pauseCapture() / resumeCapture() may be called from any thread; they are
 *   thread-safe via the [isPaused] volatile flag.
 * - stop() and handleRotation() may be called from any thread; they unblock
 *   acquireFrame() safely via isReconfiguring and semaphore release.
 */
class ScreenCaptureManager(
    private val context: Context,
    private val mediaProjection: MediaProjection,
    private val screenWidth: Int,
    private val screenHeight: Int,
    private val onProjectionStopped: () -> Unit = {},
    private val onRotationReconfigured: (width: Int, height: Int) -> Unit = { _, _ -> }
) {

    companion object {
        private const val TAG = "ScreenCaptureManager"
        private const val VIRTUAL_DISPLAY_NAME = "QeightCapture"
        private const val MAX_IMAGES = 2
        private const val NULL_IMAGE_RETRY_DELAY_MS = 3L
        private const val TARGET_CAPTURE_HEIGHT = 720f
        private const val CAPTURE_SCALE_MIN = 0.05f
        private const val CAPTURE_SCALE_MAX = 1.0f

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

    // ── Capture-resolution properties ─────────────────────────────────────────

    @Volatile
    var captureScale: Float = computeScale(screenHeight)
        private set

    @Volatile
    var captureWidth: Int = computeWidth(screenWidth, captureScale)
        private set

    @Volatile
    var captureHeight: Int = computeHeight(screenHeight, captureScale)
        private set

    // Dedicated HandlerThread for MediaProjection and ImageReader callbacks.
    private var captureThread: HandlerThread? = null
    private var captureHandler: Handler? = null

    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null

    @Volatile private var isRunning = false
    @Volatile private var isReconfiguring = false
    @Volatile private var isPaused = false

    private val frameSemaphore = Semaphore(0)

    private var currentImage: android.media.Image? = null

    // Stored so it can be unregistered in stop(), preventing stale instances
    // from receiving onStop() events after this ScreenCaptureManager is done.
    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            logD(TAG, "MediaProjection stopped by system")
            isRunning = false
            frameSemaphore.release()
            captureHandler?.post { onProjectionStopped() }
        }
    }

    // ── Static helpers ────────────────────────────────────────────────────────

    private fun computeScale(nativeHeight: Int): Float =
        (TARGET_CAPTURE_HEIGHT / nativeHeight.toFloat())
            .coerceIn(CAPTURE_SCALE_MIN, CAPTURE_SCALE_MAX)

    private fun computeWidth(nativeWidth: Int, scale: Float): Int =
        Math.round(nativeWidth * scale).coerceAtLeast(1)

    private fun computeHeight(nativeHeight: Int, scale: Float): Int =
        Math.round(nativeHeight * scale).coerceAtLeast(1)

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    fun start() {
        logD(
            TAG,
            "start: native=${screenWidth}x${screenHeight}, " +
                    "capture=${captureWidth}x${captureHeight}, " +
                    "captureScale=$captureScale"
        )

        captureThread = HandlerThread("QeightCapture").also { thread ->
            thread.start()
            captureHandler = Handler(thread.looper)
        }

        imageReader = buildImageReader(captureWidth, captureHeight)

        val density = context.resources.displayMetrics.densityDpi

        mediaProjection.registerCallback(projectionCallback, captureHandler)

        virtualDisplay = buildVirtualDisplay(captureWidth, captureHeight, density)

        isRunning = true
        isPaused  = false
        logD(TAG, "start: VirtualDisplay created (surface connected)")
    }

    // ── Rotation handling ─────────────────────────────────────────────────────

    fun handleRotation(newNativeWidth: Int, newNativeHeight: Int) {
        if (!isRunning) return

        logD(TAG, "handleRotation: new native=${newNativeWidth}x${newNativeHeight}")

        isReconfiguring = true
        frameSemaphore.release()
        releaseFrame()

        try { virtualDisplay?.release() } catch (e: Exception) {
            logE(TAG, "handleRotation: vd release: $e")
        }
        try { imageReader?.close() } catch (e: Exception) {
            logE(TAG, "handleRotation: ir close: $e")
        }
        virtualDisplay = null
        imageReader    = null

        val newScale  = computeScale(newNativeHeight)
        val newWidth  = computeWidth(newNativeWidth, newScale)
        val newHeight = computeHeight(newNativeHeight, newScale)

        captureScale  = newScale
        captureWidth  = newWidth
        captureHeight = newHeight

        logD(
            TAG,
            "handleRotation: new capture=${captureWidth}x${captureHeight}, " +
                    "captureScale=$captureScale"
        )

        frameSemaphore.drainPermits()

        val density    = context.resources.displayMetrics.densityDpi
        imageReader    = buildImageReader(captureWidth, captureHeight)
        virtualDisplay = buildVirtualDisplay(captureWidth, captureHeight, density)

        if (isPaused) {
            virtualDisplay?.setSurface(null)
        }

        isReconfiguring = false

        logD(TAG, "handleRotation: reconfiguration complete")
        captureHandler?.post { onRotationReconfigured(captureWidth, captureHeight) }
    }

    // ── Per-frame API ─────────────────────────────────────────────────────────

    fun drainStaleFrames() {
        if (!isRunning) return
        frameSemaphore.drainPermits()
        try {
            imageReader?.acquireLatestImage()?.close()
        } catch (e: Exception) {
            logW(TAG, "drainStaleFrames: exception flushing ImageReader pool: $e")
        }
    }

    fun acquireFrame(): ByteBuffer? {
        if (!isRunning || isReconfiguring || isPaused) return null

        try {
            frameSemaphore.acquire()
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            return null
        }

        if (!isRunning || isReconfiguring || isPaused) return null

        var image = imageReader?.acquireLatestImage()
        if (image == null) {
            logW(
                TAG,
                "acquireFrame: acquireLatestImage returned null, " +
                        "retrying after ${NULL_IMAGE_RETRY_DELAY_MS}ms"
            )
            try {
                Thread.sleep(NULL_IMAGE_RETRY_DELAY_MS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
            image = imageReader?.acquireLatestImage()
        }

        if (image == null) {
            logW(TAG, "acquireFrame: acquireLatestImage returned null after retry — skipping frame")
            return null
        }

        if (image.width != captureWidth || image.height != captureHeight) {
            logW(
                TAG,
                "acquireFrame: unexpected size ${image.width}x${image.height}, " +
                        "expected ${captureWidth}x${captureHeight}"
            )
            image.close()
            return null
        }

        val planes = image.planes
        if (planes.isEmpty()) {
            logW(TAG, "acquireFrame: image has no planes")
            image.close()
            return null
        }

        val plane = planes[0]

        if (plane.pixelStride != 4) {
            logW(TAG, "acquireFrame: unexpected pixelStride=${plane.pixelStride}")
            image.close()
            return null
        }

        return if (plane.rowStride == captureWidth * 4) {
            currentImage = image
            plane.buffer
        } else {
            logD(TAG, "acquireFrame: non-contiguous rows (rowStride=${plane.rowStride}), compacting")
            val compacted = compactRowStride(
                plane.buffer,
                plane.rowStride,
                captureWidth,
                captureHeight
            )
            image.close()
            currentImage = null
            compacted
        }
    }

    fun releaseFrame() {
        try {
            currentImage?.close()
        } catch (e: Exception) {
            logE(TAG, "releaseFrame: exception closing image", e)
        } finally {
            currentImage = null
        }
    }

    /**
     * Pauses frame delivery by setting a flag; VirtualDisplay remains alive.
     * The capture loop will see isPaused == true and stop calling acquireFrame(),
     * or acquireFrame() will return null immediately.
     * Safe to call from any thread.
     */
    fun pauseCapture() {
        if (!isRunning) {
            logD(TAG, "pauseCapture: not running, ignoring")
            return
        }

        if (isPaused) {
            logD(TAG, "pauseCapture: already paused")
            return
        }

        isPaused = true
        virtualDisplay?.setSurface(null)

        frameSemaphore.drainPermits()
        try {
            imageReader?.acquireLatestImage()?.close()
        } catch (e: Exception) {
            logW(TAG, "pauseCapture: pool drain exception (non-fatal)", e)
        }

        logD(TAG, "pauseCapture: paused (VirtualDisplay still running)")
    }

    /**
     * Resumes frame delivery by clearing the pause flag; VirtualDisplay was never destroyed.
     * Safe to call from any thread.
     */
    fun resumeCapture() {
        if (!isRunning) {
            logW(TAG, "resumeCapture: not running, ignoring")
            return
        }

        if (!isPaused) {
            logD(TAG, "resumeCapture: not paused, ignoring")
            return
        }

        frameSemaphore.drainPermits()
        try {
            imageReader?.acquireLatestImage()?.close()
        } catch (e: Exception) {
            logW(TAG, "resumeCapture: pool drain exception (non-fatal)", e)
        }

        virtualDisplay?.setSurface(imageReader?.surface)
        isPaused = false
        logD(TAG, "resumeCapture: resumed")
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    fun stop() {
        logD(TAG, "stop: Stopping screen capture")
        isRunning = false
        isPaused  = false
        frameSemaphore.release()

        // Unregister the stored callback before releasing resources so this
        // instance no longer receives onStop() events from the shared
        // mediaProjection after it has been torn down.
        try { mediaProjection.unregisterCallback(projectionCallback) } catch (e: Exception) {
            logE(TAG, "stop: unregisterCallback: $e")
        }

        // Quit and confirm-dead the capture HandlerThread BEFORE releasing
        // virtualDisplay/imageReader. imageReader's OnImageAvailableListener
        // and mediaProjection's Callback are both dispatched on this
        // thread's Looper — closing imageReader or releasing virtualDisplay
        // while that thread could still be mid-dispatch of a queued
        // callback is a genuine race (ImageReader.close() racing its own
        // onImageAvailable() callback is a documented source of
        // IllegalStateException / native misbehaviour). This previously ran
        // AFTER virtualDisplay.release()/imageReader.close(), which could
        // intermittently misbehave rather than crash outright — consistent
        // with an occasional-flicker symptom rather than a deterministic
        // failure.
        //
        // Verify the HandlerThread actually died rather than assuming
        // join() succeeded. A silently-swallowed join timeout here leaves a
        // zombie HandlerThread alive with its Handler still registered as
        // the ImageReader's OnImageAvailableListener and (transitively, via
        // the captured onProjectionStopped lambda) tied to this
        // ScreenCaptureManager instance and the OverlayService that
        // constructed it. Each session that leaks one of these compounds —
        // this is the "works for the first few sessions, then
        // flickers/degrades" pattern: it isn't a single missed cleanup,
        // it's zombie threads/listeners accumulating one per
        // incompletely-torn-down session.
        val thread = captureThread
        try {
            thread?.quitSafely()
            val joined = thread?.let {
                it.join(2000)
                !it.isAlive
            } ?: true
            if (!joined) {
                logE(TAG, "stop: HandlerThread did not terminate within timeout — " +
                        "forcing interrupt() to avoid a zombie capture thread surviving " +
                        "into the next session")
                @Suppress("DEPRECATION")
                thread?.interrupt()
                // Give the forced interrupt a brief additional window; if it
                // still hasn't died, there is nothing more we can safely do
                // from here — log loudly so this is visible in diagnostics
                // rather than silently passing.
                thread?.join(500)
                if (thread?.isAlive == true) {
                    logE(TAG, "stop: HandlerThread STILL alive after forced interrupt — " +
                            "this session will leak a zombie capture thread, and the " +
                            "virtualDisplay/imageReader release below may still race it")
                }
            }
        } catch (e: Exception) {
            logE(TAG, "stop: thread join: $e")
        }

        // Only now — after the capture thread is confirmed stopped (or we've
        // done everything possible to stop it) — release the display and
        // reader that thread was delivering callbacks for.
        try { virtualDisplay?.release()  } catch (e: Exception) { logE(TAG, "stop: vd release: $e")  }
        try { imageReader?.close()       } catch (e: Exception) { logE(TAG, "stop: ir close: $e")    }

        releaseFrame()

        virtualDisplay = null
        imageReader    = null
        captureThread  = null
        captureHandler = null

        logD(TAG, "stop: done")
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private fun buildImageReader(width: Int, height: Int): ImageReader =
        ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, MAX_IMAGES)
            .also { reader ->
                reader.setOnImageAvailableListener(
                    { frameSemaphore.release() },
                    captureHandler
                )
            }

    private fun buildVirtualDisplay(width: Int, height: Int, density: Int): VirtualDisplay? =
        try {
            mediaProjection.createVirtualDisplay(
                VIRTUAL_DISPLAY_NAME,
                width,
                height,
                density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null,
                captureHandler
            )
        } catch (e: Exception) {
            logE(TAG, "buildVirtualDisplay: exception", e)
            null
        }

    private fun compactRowStride(
        src: ByteBuffer,
        rowStride: Int,
        width: Int,
        height: Int
    ): ByteBuffer {
        val rowBytes = width * 4
        val dst      = ByteBuffer.allocateDirect(rowBytes * height)
        val savedPos = src.position()

        val requiredCapacity = savedPos + (height - 1) * rowStride + rowBytes
        if (src.capacity() < requiredCapacity) {
            logW(
                TAG,
                "compactRowStride: buffer too small — capacity=${src.capacity()}, " +
                        "savedPos=$savedPos, height=$height, rowStride=$rowStride, " +
                        "rowBytes=$rowBytes, required=$requiredCapacity. " +
                        "Returning zeroed buffer."
            )
            dst.rewind()
            return dst
        }

        val tmp = ByteArray(rowBytes)
        for (row in 0 until height) {
            src.position(savedPos + row * rowStride)
            src.get(tmp, 0, rowBytes)
            dst.put(tmp)
        }
        dst.rewind()
        src.position(savedPos)
        return dst
    }
}