package com.ashraf.qeight

import android.annotation.SuppressLint
import android.app.PendingIntent
import android.app.StatusBarManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import android.util.Log
import androidx.annotation.RequiresApi
import java.util.concurrent.Executor

class QEightTileService : TileService() {

    companion object {
        private const val TAG = "QEightTileService"

        // Request codes for PendingIntents — kept unique to avoid clobbering
        private const val RC_MAIN_ACTIVITY       = 100
        private const val RC_PERMISSION_SETTINGS = 200
        private const val RC_TRAMPOLINE_BASE     = 300

        /**
         * SharedPreferences key (in "qeight_prefs", same file MainActivity
         * uses) tracking whether the QS tile has been added by the user.
         * Kept in sync from onTileAdded()/onTileRemoved(). There is no
         * direct system API to *query* whether a tile is currently added —
         * these lifecycle callbacks are the only reliable signal — so we
         * persist it here for MainActivity to read on every resume.
         */
        const val KEY_TILE_ADDED = "qs_tile_added"

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

        /**
         * Request the system to refresh this tile's state.
         * Called from OverlayService when it starts/stops.
         */
        fun requestTileUpdate(context: Context) {
            val component = ComponentName(context, QEightTileService::class.java)
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    val manager = context.getSystemService(StatusBarManager::class.java)
                    if (manager != null) {
                        try {
                            val method = StatusBarManager::class.java.getMethod(
                                "requestTileServiceListeningState",
                                ComponentName::class.java
                            )
                            method.invoke(manager, component)
                            logD(TAG, "requestTileUpdate: via StatusBarManager (API 34+)")
                            return
                        } catch (e: NoSuchMethodException) {
                            logW(TAG, "requestTileServiceListeningState not found, falling back", e)
                        } catch (e: Exception) {
                            logW(TAG, "requestTileServiceListeningState failed, falling back", e)
                        }
                    }
                }
                requestListeningState(context, component)
                logD(TAG, "requestTileUpdate: via requestListeningState")
            } catch (e: Exception) {
                logW(TAG, "requestTileUpdate failed (non-fatal)", e)
            }
        }

        /**
         * Drawn at runtime so system tinting cannot flatten it.
         *
         * IMPORTANT: The system QS tile ALWAYS tints [Tile.icon] to a single flat
         * color (white when STATE_ACTIVE, translucent gray/white when
         * STATE_INACTIVE) before drawing it — it discards the bitmap's RGB
         * channels entirely and recolors using only the alpha (shape) channel.
         *
         * That means any detail created purely through *color contrast* (e.g. a
         * black "8" drawn on top of a white circle, both fully opaque) is
         * invisible once tinted — the system sees one solid opaque blob and
         * paints the whole thing a single color. That's exactly the plain
         * filled circle bug seen in the Quick Settings panel.
         *
         * The fix: encode the "8" glyph as real *transparency* (a hole punched
         * into the alpha channel) rather than as a differently-colored fill. A
         * transparent hole stays transparent no matter what flat tint color the
         * system applies to the rest of the shape, so the figure-eight
         * silhouette remains visible under any tint.
         */
        fun createEightBallIcon(context: Context): Icon {
            return try {
                val size   = 192
                val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                val cx     = size / 2f
                val cy     = size / 2f
                val radius = size / 2f - 2f // bigger ball — minimal margin
                val paint  = Paint(Paint.ANTI_ALIAS_FLAG)

                // 1. Solid ball silhouette. This is the only "filled" shape —
                // it's what the system will flat-tint. The "8" below is *cut
                // out* as a transparent hole, so it survives tinting instead of
                // disappearing into a single-color blob.
                paint.reset()
                paint.isAntiAlias = true
                paint.style = Paint.Style.FILL
                paint.color = Color.BLACK
                canvas.drawCircle(cx, cy, radius, paint)

                // 2. Cut the "8" glyph out of the ball as a transparent hole.
                // PorterDuff.Mode.DST_OUT erases destination pixels wherever the
                // source (the glyph) is drawn, regardless of the source's own
                // color — so this reads correctly whether the final system tint
                // ends up white, gray, or anything else.
                paint.reset()
                paint.isAntiAlias = true
                paint.style     = Paint.Style.FILL
                paint.color     = Color.BLACK // color is irrelevant under DST_OUT
                paint.typeface  = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                paint.textAlign = Paint.Align.CENTER
                paint.textSize  = radius * 1.3f // bigger "8"
                paint.xfermode  = android.graphics.PorterDuffXfermode(
                    android.graphics.PorterDuff.Mode.DST_OUT
                )

                val bounds = Rect()
                paint.getTextBounds("8", 0, 1, bounds)
                canvas.drawText("8", cx, cy + bounds.height() / 2f, paint)
                paint.xfermode = null

                Icon.createWithBitmap(bitmap)
            } catch (e: Exception) {
                logE(TAG, "createEightBallIcon: bitmap failed, using safe fallback", e)
                createFallbackEightBallIcon()
            }
        }

        /**
         * Last-resort fallback icon, used only if [createEightBallIcon] itself
         * throws (e.g. Typeface/Canvas failure). Deliberately does NOT use
         * Icon.createWithResource(R.drawable.ic_eight_ball) or any other
         * static drawable resource.
         *
         * Root cause of the "sometimes draws a solid ball" bug: both former
         * fallback call sites pointed at R.drawable.ic_eight_ball, a static
         * XML/PNG drawable. The system tints Tile.icon to a single flat color
         * using ONLY the alpha channel (see the big comment on
         * createEightBallIcon above) — so unless that drawable resource
         * happens to encode its "8" as genuine alpha transparency rather than
         * color contrast, it collapses into a solid filled circle once
         * tinted. This was invisible during normal operation (the runtime
         * bitmap path is correct and used almost always) and only showed up
         * on the fallback path — e.g. right after install/before setup, or
         * on ColorOS/OPlus where Canvas/Typeface initialization can behave
         * differently early in the tile's lifecycle — which matches the
         * reported symptom exactly.
         *
         * Fix: this fallback is built the same way as the primary icon (flat
         * black circle + DST_OUT "8" cutout), just with a plainer typeface
         * and no try/catch-able extras, so it is guaranteed correct under
         * flat tinting with no dependency on any external drawable resource.
         */
        private fun createFallbackEightBallIcon(): Icon {
            val size   = 192
            val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val cx     = size / 2f
            val cy     = size / 2f
            val radius = size / 2f - 2f
            val paint  = Paint(Paint.ANTI_ALIAS_FLAG)

            // Solid ball silhouette — the only filled shape, safe to flat-tint.
            paint.style = Paint.Style.FILL
            paint.color = Color.BLACK
            canvas.drawCircle(cx, cy, radius, paint)

            // Cut the "8" out as a transparent hole via DST_OUT, exactly like
            // the primary icon, so it survives tinting under any color.
            paint.reset()
            paint.isAntiAlias = true
            paint.style     = Paint.Style.FILL
            paint.color     = Color.BLACK // irrelevant under DST_OUT
            paint.typeface  = Typeface.DEFAULT_BOLD
            paint.textAlign = Paint.Align.CENTER
            paint.textSize  = radius * 1.3f
            paint.xfermode  = android.graphics.PorterDuffXfermode(
                android.graphics.PorterDuff.Mode.DST_OUT
            )

            val bounds = Rect()
            paint.getTextBounds("8", 0, 1, bounds)
            canvas.drawText("8", cx, cy + bounds.height() / 2f, paint)
            paint.xfermode = null

            return Icon.createWithBitmap(bitmap)
        }

        /**
         * Returns true if we know the QS tile has already been added
         * (persisted from a prior onTileAdded() callback). This is a
         * best-effort signal — a user could also have added the tile
         * manually via long-press-on-QS-panel > Edit, which fires
         * onTileAdded() the same way, so this stays accurate either way
         * once the tile has been added at least once.
         */
        fun isTileAdded(context: Context): Boolean =
            context.getSharedPreferences("qeight_prefs", Context.MODE_PRIVATE)
                .getBoolean(KEY_TILE_ADDED, false)

        /**
         * Prompts the system "Add tile to Quick Settings?" dialog directly
         * (API 33+ only — TileService.requestAddTileService has no
         * equivalent on older versions, where users must add tiles manually
         * via the QS panel's Edit screen).
         *
         * onResult receives true if the user accepted (tile added — mirrors
         * what onTileAdded() will also report), false otherwise.
         */
        @RequiresApi(Build.VERSION_CODES.TIRAMISU)
        fun requestAddTile(context: Context, onResult: (Boolean) -> Unit) {
            try {
                val statusBarManager = context.getSystemService(StatusBarManager::class.java)
                val label = try { context.getString(R.string.app_name) } catch (e: Exception) { "Qeight" }
                statusBarManager?.requestAddTileService(
                    ComponentName(context, QEightTileService::class.java),
                    label,
                    createEightBallIcon(context),
                    Executor { it.run() }, // run callback synchronously on caller's thread
                    { result ->
                        val added = result == StatusBarManager.TILE_ADD_REQUEST_RESULT_TILE_ADDED
                        logD(TAG, "requestAddTile: result=$result added=$added")
                        if (added) {
                            context.getSharedPreferences("qeight_prefs", Context.MODE_PRIVATE)
                                .edit().putBoolean(KEY_TILE_ADDED, true).apply()
                        }
                        onResult(added)
                    }
                )
            } catch (e: Exception) {
                logE(TAG, "requestAddTile failed", e)
                onResult(false)
            }
        }

        /** Returns true if the SYSTEM_ALERT_WINDOW permission has been granted. */
        fun hasOverlayPermission(context: Context): Boolean =
            Settings.canDrawOverlays(context)

        /** Returns true if OverlayService is currently running. */
        fun isServiceRunning(context: Context): Boolean {
            return try {
                val am = context.getSystemService(ACTIVITY_SERVICE)
                        as android.app.ActivityManager
                @Suppress("DEPRECATION")
                am.getRunningServices(Int.MAX_VALUE)
                    .any { it.service.className == OverlayService::class.java.name }
            } catch (e: Exception) {
                logW(TAG, "isServiceRunning: exception, assuming false", e)
                false
            }
        }
    }

    // -------------------------------------------------------------------------
    // TileService lifecycle
    // -------------------------------------------------------------------------

    override fun onStartListening() {
        super.onStartListening()
        logD(TAG, "onStartListening")
        refreshTile()
    }

    override fun onStopListening() {
        super.onStopListening()
        logD(TAG, "onStopListening")
    }

    override fun onTileAdded() {
        super.onTileAdded()
        logD(TAG, "onTileAdded")
        // Persist so MainActivity's status dot / add-tile prompt can know the
        // tile is already present without needing a live TileService binding.
        getSharedPreferences("qeight_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_TILE_ADDED, true).apply()
        refreshTile()
    }

    override fun onTileRemoved() {
        super.onTileRemoved()
        logD(TAG, "onTileRemoved")
        getSharedPreferences("qeight_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_TILE_ADDED, false).apply()
    }

    override fun onClick() {
        val permissionGranted = hasOverlayPermission(this)
        val serviceRunning    = isOverlayServiceRunning()
        logD(TAG, "onClick — permissionGranted=$permissionGranted, serviceRunning=$serviceRunning")

        try {
            when {
                serviceRunning     -> stopOverlayService()
                !permissionGranted -> openPermissionSettings()
                else               -> startOverlayService()
            }
        } catch (e: SecurityException) {
            logE(TAG, "onClick: SecurityException", e)
            refreshTile()
        } catch (e: IllegalStateException) {
            logE(TAG, "onClick: IllegalStateException", e)
            refreshTile()
        } catch (e: Exception) {
            logE(TAG, "onClick: unexpected exception", e)
            refreshTile()
        }
    }

    // -------------------------------------------------------------------------
    // Service control
    // -------------------------------------------------------------------------

    private fun stopOverlayService() {
        try {
            stopService(Intent(this, OverlayService::class.java))
            logD(TAG, "stopOverlayService: requested")
            // Tile refreshed via requestTileUpdate() called from OverlayService.onDestroy()
        } catch (e: Exception) {
            logE(TAG, "stopOverlayService: failed", e)
            refreshTile()
        }
    }

    private fun startOverlayService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            launchTrampoline(TrampolineActivity.MODE_FULL)
            logD(TAG, "startOverlayService: trampoline (API 34+)")
        } else {
            launchActivityViaPendingIntent(
                intent = Intent(this, MainActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                    addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    putExtra("from_tile", true)
                },
                requestCode = RC_MAIN_ACTIVITY
            )
            logD(TAG, "startOverlayService: MainActivity (API < 34)")
        }
    }

    /**
     * Send the user to the system overlay-permission screen.
     * Works on all supported API levels without using the deprecated
     * startActivityAndCollapse(Intent) overload.
     */
    private fun openPermissionSettings() {
        logD(TAG, "openPermissionSettings: opening MANAGE_OVERLAY_PERMISSION")
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:$packageName")
        ).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        launchActivityViaPendingIntent(intent, RC_PERMISSION_SETTINGS)
    }

    // -------------------------------------------------------------------------
    // Tile rendering
    // -------------------------------------------------------------------------

    private fun refreshTile() {
        val tile = qsTile ?: run {
            logW(TAG, "refreshTile: qsTile is null — not listening")
            return
        }

        // Icon
        try {
            tile.icon = createEightBallIcon(this)
        } catch (e: Exception) {
            logE(TAG, "refreshTile: icon failed", e)
            try {
                tile.icon = createFallbackEightBallIcon()
            } catch (e2: Exception) {
                logE(TAG, "refreshTile: fallback icon also failed", e2)
            }
        }

        // State
        val permissionGranted = hasOverlayPermission(this)
        val serviceRunning    = isOverlayServiceRunning()
        logD(TAG, "refreshTile: permissionGranted=$permissionGranted, serviceRunning=$serviceRunning")

        when {
            serviceRunning -> {
                tile.state = Tile.STATE_ACTIVE
                tile.label = "Qeight Active"
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    tile.subtitle = "Tap to stop"
                }
            }
            !permissionGranted -> {
                tile.state = Tile.STATE_INACTIVE
                tile.label = "Qeight"
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    tile.subtitle = "Grant permission"
                }
            }
            else -> {
                tile.state = Tile.STATE_INACTIVE
                tile.label = "Qeight"
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    tile.subtitle = "Tap to start"
                }
            }
        }

        try {
            tile.updateTile()
            logD(TAG, "refreshTile: done — state=${tile.state}, label=${tile.label}")
        } catch (e: Exception) {
            logE(TAG, "refreshTile: updateTile() threw", e)
        }
    }

    private fun refreshTileUnavailable() {
        val tile = qsTile ?: return
        try {
            tile.state = Tile.STATE_UNAVAILABLE
            tile.label = "Qeight"
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                tile.subtitle = "Setup required"
            }
            tile.updateTile()
        } catch (e: Exception) {
            logE(TAG, "refreshTileUnavailable: failed", e)
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun isOverlayServiceRunning() = isServiceRunning(this)

    /**
     * The single, canonical way to launch any activity from this TileService.
     *
     * • On **all API levels** we use [startActivityAndCollapse] with a
     *   [PendingIntent] — the only non-deprecated overload available.
     * • The deprecated [startActivityAndCollapse(Intent)] overload is
     *   intentionally never called.
     */
    @SuppressLint("StartActivityAndCollapseDeprecated")
    private fun launchActivityViaPendingIntent(intent: Intent, requestCode: Int) {
        try {
            val pi = PendingIntent.getActivity(
                this,
                requestCode,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                // API 34+: only non-deprecated overload
                startActivityAndCollapse(pi)
            } else {
                // API 24–33: PendingIntent overload was added in API 34, so we
                // must use the Intent overload here; suppress the warning because
                // this branch genuinely cannot use the PendingIntent overload.
                @Suppress("DEPRECATION")
                startActivityAndCollapse(intent)
            }
            logD(TAG, "launchActivityViaPendingIntent: rc=$requestCode launched")
        } catch (e: Exception) {
            logE(TAG, "launchActivityViaPendingIntent: rc=$requestCode failed", e)
            refreshTileUnavailable()
        }
    }

    /**
     * API 34+: launch TrampolineActivity so it can start OverlayService
     * from a proper foreground context.
     */
    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private fun launchTrampoline(mode: String) {
        val intent = Intent(this, TrampolineActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
            addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY)
            addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS)
            addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION)
            putExtra(TrampolineActivity.EXTRA_MODE, mode)
            putExtra(TrampolineActivity.EXTRA_FROM_TILE, true)
        }
        try {
            val pi = PendingIntent.getActivity(
                this,
                RC_TRAMPOLINE_BASE + (mode.hashCode() and 0xFF),
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            startActivityAndCollapse(pi)
            logD(TAG, "launchTrampoline: mode=$mode launched")
        } catch (e: Exception) {
            logE(TAG, "launchTrampoline: mode=$mode failed", e)
            refreshTileUnavailable()
        }
    }
}