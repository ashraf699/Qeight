package com.ashraf.qeight

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import android.widget.Toast

/**
 * TrampolineActivity — completely transparent, no-animation Activity.
 *
 * Its sole job is to host permission dialogs (screen capture, usage stats)
 * on top of whatever is currently on screen — including 8 Ball Pool in
 * landscape — without disturbing the game's orientation or bringing the
 * QEight home screen to the front.
 *
 * Critical window flags are applied in onCreate BEFORE setContentView
 * (there is no setContentView) so the window never shows a background,
 * never dims the game, and never causes a visible task switch.
 *
 * Modes (passed via EXTRA_MODE):
 *
 *   MODE_FULL
 *     • Request screen-capture permission if not already held.
 *     • Start OverlayService.
 *     • Launch 8 Ball Pool.
 *     • finish().
 *
 *   MODE_SILENT
 *     • Request screen-capture permission if not already held.
 *     • Start OverlayService.
 *     • Do NOT launch 8 Ball Pool.
 *     • finish().
 *
 *   MODE_REQUEST_USAGE_STATS
 *     • Open the Usage Access settings screen.
 *     • finish() immediately — user returns to whatever they were doing.
 *     • Next tile tap will re-evaluate.
 *
 * Theme: Theme.QEight.Transparent (defined in themes.xml) — fully
 * transparent window so nothing flashes on screen.
 */
class TrampolineActivity : Activity() {

    companion object {
        private const val TAG = "TrampolineActivity"

        const val EXTRA_MODE        = "trampoline_mode"
        const val EXTRA_FROM_TILE   = "from_tile"

        const val MODE_FULL                = "full"
        const val MODE_SILENT              = "silent"
        const val MODE_REQUEST_USAGE_STATS = "usage_stats"

        private const val RC_SCREEN_CAPTURE = 1001
        private const val RC_USAGE_STATS    = 1002

        private const val EIGHT_BALL_PKG = "com.miniclip.eightballpool"

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

    private lateinit var projectionManager: MediaProjectionManager
    private var mode = MODE_FULL

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ----------------------------------------------------------------
        // Apply window flags BEFORE anything is drawn.
        //
        // FLAG_NOT_TOUCHABLE           — we only host a system dialog;
        //                               our own window surface should
        //                               never intercept touches.
        // FLAG_LAYOUT_IN_SCREEN        — keep layout stable when the
        //                               system dialog appears.
        // FLAG_SHOW_WHEN_LOCKED        — let the permission dialog appear
        //                               even on the lock screen edge-case.
        //
        // We do NOT set FLAG_KEEP_SCREEN_ON or anything that would pull
        // focus away from the game before the system dialog appears.
        // ----------------------------------------------------------------
        window.apply {
            // Prevent our transparent window from creating a dark scrim
            // / dim behind it — the game must stay fully visible.
            clearFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
            addFlags(
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN  or
                        WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED
            )
            // Zero out the dim amount so even if the system tries to dim,
            // nothing changes behind us.
            attributes = attributes.also { it.dimAmount = 0f }
        }

        // No setContentView — window is fully transparent, nothing to inflate.
        projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE)
                as MediaProjectionManager

        mode = intent?.getStringExtra(EXTRA_MODE) ?: MODE_FULL
        logD(TAG, "onCreate: mode=$mode")

        when (mode) {
            MODE_REQUEST_USAGE_STATS -> {
                openUsageStatsSettings()
                // finish immediately — user handles settings and comes back
                finish()
            }
            MODE_FULL, MODE_SILENT -> {
                // Ask screen-capture permission — the system dialog floats
                // over the game. Our transparent window is never seen.
                requestScreenCapture()
            }
            else -> {
                logW(TAG, "onCreate: unknown mode=$mode, finishing")
                finish()
            }
        }
    }

    // -------------------------------------------------------------------------
    // Permission requests
    // -------------------------------------------------------------------------

    private fun requestScreenCapture() {
        try {
            startActivityForResult(
                projectionManager.createScreenCaptureIntent(),
                RC_SCREEN_CAPTURE
            )
        } catch (e: Exception) {
            logE(TAG, "requestScreenCapture failed", e)
            Toast.makeText(this, "Failed to request screen capture", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun openUsageStatsSettings() {
        try {
            startActivity(
                Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
            )
        } catch (e: Exception) {
            logE(TAG, "openUsageStatsSettings failed", e)
            // Non-fatal — next tile tap will retry.
        }
    }

    // -------------------------------------------------------------------------
    // Result handling
    // -------------------------------------------------------------------------

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        when (requestCode) {
            RC_SCREEN_CAPTURE -> handleScreenCaptureResult(resultCode, data)
            RC_USAGE_STATS    -> finish() // tile will re-evaluate on next tap
            else              -> finish()
        }
    }

    private fun handleScreenCaptureResult(resultCode: Int, data: Intent?) {
        if (resultCode == RESULT_OK && data != null) {
            logD(TAG, "handleScreenCaptureResult: granted, starting service (mode=$mode)")
            startOverlayService(data, resultCode)

            if (mode == MODE_FULL) {
                launchEightBallPool()
            }
        } else {
            logW(TAG, "handleScreenCaptureResult: denied or cancelled")
            Toast.makeText(this, "Screen capture permission required", Toast.LENGTH_SHORT).show()
        }
        finish()
    }

    // -------------------------------------------------------------------------
    // Service start
    // -------------------------------------------------------------------------

    private fun startOverlayService(projectionData: Intent, resultCode: Int) {
        try {
            startForegroundService(Intent(this, OverlayService::class.java).apply {
                putExtra(OverlayService.EXTRA_PROJECTION_DATA, projectionData)
                putExtra(OverlayService.EXTRA_PROJECTION_RESULT_CODE, resultCode)
            })
            logD(TAG, "startOverlayService: launched")
        } catch (e: Exception) {
            logE(TAG, "startOverlayService failed", e)
            Toast.makeText(this, "Failed to start QEight: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // -------------------------------------------------------------------------
    // 8 Ball Pool launch (MODE_FULL only)
    // -------------------------------------------------------------------------

    private fun launchEightBallPool() {
        try {
            var launched = false

            packageManager.getLaunchIntentForPackage(EIGHT_BALL_PKG)?.let {
                it.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
                try {
                    startActivity(it)
                    launched = true
                } catch (_: Exception) {}
            }

            if (!launched) try {
                startActivity(Intent(Intent.ACTION_MAIN).apply {
                    addCategory(Intent.CATEGORY_LAUNCHER)
                    setPackage(EIGHT_BALL_PKG)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
                })
                launched = true
            } catch (_: Exception) {}

            if (!launched) try {
                startActivity(
                    Intent(
                        Intent.ACTION_VIEW,
                        Uri.parse("market://details?id=$EIGHT_BALL_PKG")
                    ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
                )
            } catch (_: Exception) {
                Toast.makeText(this, "Please install 8 Ball Pool", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            logE(TAG, "launchEightBallPool failed", e)
        }
    }
}