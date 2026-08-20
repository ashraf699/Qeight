package com.ashraf.qeight

import android.accessibilityservice.AccessibilityServiceInfo
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.app.Dialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.text.Html
import android.text.method.LinkMovementMethod
import android.util.DisplayMetrics
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityManager
import android.view.animation.AccelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val PREFS_NAME = "qeight_prefs"
        private const val KEY_ROI_CALIBRATION = "roi_calibration_json"
        private const val KEY_HOME_SHOWN = "home_screen_shown"
        private const val KEY_TILE_PROMPT_SHOWN = "tile_prompt_shown"
        private const val KEY_ACCESSIBILITY_PROMPT_SHOWN = "accessibility_prompt_shown"
        // Persisted permanently once the user has ever granted the accessibility
        // service. Unlike KEY_ACCESSIBILITY_PROMPT_SHOWN (which only prevents
        // the auto-prompt from repeating), this flag is used to suppress the
        // hard-block in startOverlayService() on cold-start before the
        // accessibility service instance has had time to bind — a timing window
        // that is particularly wide on ColorOS/OPlus where the system is slower
        // to rebind background services after a cold app launch.
        private const val KEY_ACCESSIBILITY_GRANTED = "accessibility_service_granted"
        // Stamped with PackageManager's firstInstallTime the first time this
        // flag is written. Compared against the live firstInstallTime on every
        // cold start to detect a reinstall — see
        // reconcileAccessibilityGrantedFlagWithInstall(). This is necessary
        // because KEY_ACCESSIBILITY_GRANTED (like all SharedPreferences data)
        // can survive a reinstall via Android's Auto Backup for Apps or
        // OEM-level "restore app data" flows, even though accessibility
        // service grants themselves are NEVER restored by the OS — the user
        // must always manually re-enable the service in system Settings after
        // a reinstall. Without this check, a restored-but-stale
        // KEY_ACCESSIBILITY_GRANTED=true would cause isAccessibilityServiceEnabled()
        // to report granted forever, even though the service was never
        // actually re-enabled on the new install.
        private const val KEY_INSTALL_TIME = "recorded_first_install_time"
        // Counts consecutive isAccessibilityServiceEnabled() calls where ALL
        // THREE runtime checks (instance / AccessibilityManager / Settings.Secure)
        // agreed the service is absent, while KEY_ACCESSIBILITY_GRANTED was still
        // true. Used to distinguish a real revocation (toggle turned off in
        // Settings) from a transient cold-start bind-lag window. Reset to 0 the
        // moment any check succeeds.
        private const val KEY_CONSECUTIVE_NEGATIVE_CHECKS = "accessibility_consecutive_negative_checks"
        // How many consecutive all-negative checks are required before we trust
        // it as a real revocation and clear KEY_ACCESSIBILITY_GRANTED. >1 so a
        // single cold-start timing gap (the case the tiebreaker exists for)
        // never trips it, but a genuinely-off toggle gets caught within a
        // couple of onResume cycles instead of never.
        private const val NEGATIVE_CHECKS_THRESHOLD = 2
        private const val KEY_BATTERY_PROMPT_SHOWN = "battery_optimization_prompt_shown"
        private const val COLOR_GREEN = "#00C853"
        private const val COLOR_RED = "#D50000"
        const val ACTION_SERVICE_STOPPED = "com.ashraf.qeight.SERVICE_STOPPED"

        /**
         * Master switch for verbose logging.
         * Set to [true] to enable all Log.d / Log.w / Log.e output.
         * Default is [false] (logging OFF) for production builds.
         */
        private const val LOGGING_ENABLED = true

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

    private lateinit var prefs: SharedPreferences
    private lateinit var projectionManager: MediaProjectionManager
    private lateinit var accessibilityManager: AccessibilityManager
    private var projectionData: Intent? = null
    private var splashDismissed = false
    private var splashDialog: Dialog? = null
    private var homeScreenShown = false
    private var pendingServiceStart = false

    private lateinit var overlayPermissionLauncher: ActivityResultLauncher<Intent>
    private lateinit var projectionLauncher: ActivityResultLauncher<Intent>
    private lateinit var accessibilitySettingsLauncher: ActivityResultLauncher<Intent>

    private var dotOverlay: View? = null
    private var dotCapture: View? = null
    private var dotRoi: View? = null
    private var dotAccessibility: View? = null
    private var accessibilityDotContainer: View? = null
    private var dotTile: View? = null
    private var tileDotContainer: View? = null
    private var btnStart: Button? = null

    // Receives CALIBRATION_SAVED broadcast → refresh dots
    private val calibrationReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            updateStatusDots()
        }
    }

    // Receives SERVICE_STOPPED broadcast → null out stale token, refresh dots
    private val serviceStoppedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            projectionData = null
            updateStatusDots()
        }
    }

    // Guards to prevent double-register / double-unregister crashes
    private var calibrationReceiverRegistered = false
    private var serviceStoppedReceiverRegistered = false

    // =========================================================================
    // Lifecycle
    // =========================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        // Must run before any accessibility check (isAccessibilityServiceEnabled,
        // maybePromptForAccessibility, startOverlayService, etc.) touches
        // KEY_ACCESSIBILITY_GRANTED, so a stale flag restored from a backup
        // onto a reinstall is cleared before it can be trusted.
        reconcileAccessibilityGrantedFlagWithInstall()
        projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                as MediaProjectionManager
        accessibilityManager = getSystemService(Context.ACCESSIBILITY_SERVICE)
                as AccessibilityManager
        registerLaunchers()

        // If we are being recreated after the landscape orientation change that
        // inflateHomeScreen() triggers, KEY_HOME_SHOWN will be true in the bundle.
        // Skip the splash entirely and go straight to the home screen so the
        // splash is never shown a second time.
        if (savedInstanceState?.getBoolean(KEY_HOME_SHOWN, false) == true) {
            splashDismissed = true
            inflateHomeScreen()
        } else {
            // Genuine first launch — show the splash in the system's natural
            // orientation (portrait). inflateHomeScreen() will force landscape
            // later, after the splash is dismissed.
            showCustomSplash()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // Persist the flag across the Activity recreation that is caused by the
        // orientation change triggered inside inflateHomeScreen().
        outState.putBoolean(KEY_HOME_SHOWN, homeScreenShown)
    }

    override fun onResume() {
        super.onResume()
        if (splashDismissed) {
            // Immediate dot refresh with whatever state is available now.
            updateStatusDots()

            // On ColorOS/OPlus (Realme, Oppo, OnePlus) the accessibility
            // service instance binds lazily — it may not be connected yet
            // when onResume fires right after a cold start. Deferring the
            // prompt check by ~600 ms gives the system time to bind, so
            // isAccessibilityServiceEnabled() sees the live instance and
            // doesn't spuriously trigger the prompt (or the dot stays red
            // when it should be green).
            window.decorView.postDelayed({
                if (!isFinishing && !isDestroyed) {
                    // Refresh dots a second time after the bind window.
                    updateStatusDots()
                    // Check accessibility service state and potentially prompt.
                    maybePromptForAccessibility()
                    // Check battery optimization state and potentially prompt.
                    // Important on OEM-customized Android builds (ColorOS/Oplus,
                    // MIUI, etc.) where a correctly-declared foreground service
                    // with MediaProjection can still be silently killed by the
                    // OEM's own power manager unless the app is explicitly
                    // whitelisted.
                    maybePromptForBatteryOptimization()
                }
            }, 600L)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Unregister both receivers to avoid leaks
        if (calibrationReceiverRegistered) {
            try {
                LocalBroadcastManager.getInstance(this)
                    .unregisterReceiver(calibrationReceiver)
            } catch (e: Exception) {
                logE(TAG, "onDestroy unregisterReceiver calibration", e)
            }
            calibrationReceiverRegistered = false
        }
        if (serviceStoppedReceiverRegistered) {
            try {
                LocalBroadcastManager.getInstance(this)
                    .unregisterReceiver(serviceStoppedReceiver)
            } catch (e: Exception) {
                logE(TAG, "onDestroy unregisterReceiver serviceStopped", e)
            }
            serviceStoppedReceiverRegistered = false
        }
    }

    // =========================================================================
    // Launcher registration
    // =========================================================================

    private fun registerLaunchers() {
        overlayPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { _ ->
            // Refresh dots AND nudge tile — overlay state just changed
            updateStatusDots()
            if (pendingServiceStart) {
                if (Settings.canDrawOverlays(this)) {
                    if (projectionData == null) {
                        try {
                            projectionLauncher.launch(
                                projectionManager.createScreenCaptureIntent()
                            )
                        } catch (e: Exception) {
                            logE(TAG, "projection launch failed", e)
                            pendingServiceStart = false
                        }
                    } else {
                        doStartService()
                        pendingServiceStart = false
                    }
                } else {
                    pendingServiceStart = false
                    Toast.makeText(
                        this, "Overlay permission required", Toast.LENGTH_LONG
                    ).show()
                }
            }
        }

        projectionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK && result.data != null) {
                projectionData = result.data
                updateStatusDots()
                if (pendingServiceStart) {
                    doStartService()
                    pendingServiceStart = false
                }
            } else {
                pendingServiceStart = false
                Toast.makeText(
                    this, "Screen capture permission required", Toast.LENGTH_LONG
                ).show()
                updateStatusDots()
            }
        }

        accessibilitySettingsLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { _ ->
            // User returned from Accessibility settings.
            // Give the system a moment to settle — on ColorOS/OPlus the
            // AccessibilityManager list and Settings.Secure string can lag
            // behind the actual toggle by ~300 ms after returning to the app.
            window.decorView.postDelayed({
                // If the service is now absent according to all runtime
                // checks, clear the persistent granted flag so the dot turns
                // red correctly. If it's present (or the binding is just
                // catching up), the flag is refreshed/kept by the check itself.
                if (!isAccessibilityServiceEnabled()) {
                    maybeClearAccessibilityGrantedFlag()
                }
                updateStatusDots()

                // If the user enabled the service and there's a pending start,
                // continue the permission flow.
                if (pendingServiceStart && isAccessibilityServiceEnabled()) {
                    startOverlayService()
                }
            }, 400L)
        }
    }

    // =========================================================================
    // Accessibility Service detection
    // =========================================================================

    /**
     * Detects whether this launch is happening on a "fresh" install relative
     * to whatever wrote KEY_ACCESSIBILITY_GRANTED, and wipes the flag if so.
     *
     * KEY_ACCESSIBILITY_GRANTED is meant to persist ONLY for the lifetime of
     * a single install — per the product requirement, it must NOT survive a
     * reinstall. But plain SharedPreferences can survive a reinstall via:
     *   • Android's Auto Backup for Apps (on by default for apps targeting
     *     API 23+ unless explicitly opted out in the manifest), which backs
     *     up shared_prefs to the user's Google Drive and restores it
     *     automatically on the next install.
     *   • OEM-level "restore app data" / clone features (Samsung Smart
     *     Switch, MIUI backup, etc.) that do the same thing outside of
     *     Android's own backup agent.
     *
     * Accessibility service grants themselves are never restored by the OS —
     * the user always has to manually re-enable the service in system
     * Settings after a reinstall. So a restored KEY_ACCESSIBILITY_GRANTED=true
     * with no real grant behind it is pure false-positive risk: every runtime
     * check (instance/AccessibilityManager/Settings.Secure) will correctly
     * report "not enabled", but isAccessibilityServiceEnabled()'s cold-start
     * tiebreaker will still trust the stale flag and report granted anyway.
     *
     * Fix: stamp PackageManager's firstInstallTime into prefs the first time
     * we ever see this install. On every subsequent cold start, compare the
     * live firstInstallTime against the stamped one:
     *   • Match → same install as when the flag was set (or no flag yet);
     *     leave everything as-is.
     *   • Mismatch (or no stamp recorded at all, e.g. prefs came from a
     *     restored backup that predates this check) → this app data does not
     *     belong to the current install. Wipe KEY_ACCESSIBILITY_GRANTED and
     *     re-stamp the current firstInstallTime, so stale grants never leak
     *     across a reinstall.
     *
     * Called once, early in onCreate(), before any accessibility check runs.
     */
    private fun reconcileAccessibilityGrantedFlagWithInstall() {
        try {
            val liveInstallTime = packageManager.getPackageInfo(packageName, 0).firstInstallTime
            val recordedInstallTime = prefs.getLong(KEY_INSTALL_TIME, -1L)

            if (recordedInstallTime == -1L) {
                // No stamp recorded yet. Could be a genuine first-ever launch
                // (nothing to reconcile, nothing granted yet), OR it could be
                // prefs restored from a backup taken before this check
                // existed — in which case KEY_ACCESSIBILITY_GRANTED may
                // already be true despite no real grant on this install.
                // Either way, the safe move is to clear any stale grant flag
                // and stamp the current install time as the new baseline.
                if (prefs.getBoolean(KEY_ACCESSIBILITY_GRANTED, false)) {
                    logD(TAG, "reconcileAccessibilityGrantedFlagWithInstall: no install " +
                            "stamp recorded but KEY_ACCESSIBILITY_GRANTED was true — " +
                            "likely a restored backup predating this check; clearing it")
                }
                prefs.edit()
                    .putLong(KEY_INSTALL_TIME, liveInstallTime)
                    .putBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                    .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                    .apply()
                return
            }

            if (recordedInstallTime != liveInstallTime) {
                // This app data does not belong to the current install —
                // either restored from backup onto a reinstall, or restored
                // onto a different device/profile. Wipe the stale grant flag
                // and re-stamp so this install starts clean.
                logD(TAG, "reconcileAccessibilityGrantedFlagWithInstall: install time " +
                        "mismatch (recorded=$recordedInstallTime, live=$liveInstallTime) — " +
                        "treating as reinstall, clearing KEY_ACCESSIBILITY_GRANTED")
                prefs.edit()
                    .putLong(KEY_INSTALL_TIME, liveInstallTime)
                    .putBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                    .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                    .apply()
            }
            // else: same install as last time this flag was touched — leave
            // KEY_ACCESSIBILITY_GRANTED exactly as it is.
        } catch (e: Exception) {
            // If we can't determine install time for some reason, fail safe:
            // do NOT trust a persisted grant flag we can't verify belongs to
            // this install.
            logW(TAG, "reconcileAccessibilityGrantedFlagWithInstall: failed to read " +
                    "firstInstallTime — clearing KEY_ACCESSIBILITY_GRANTED to be safe", e)
            prefs.edit()
                .putBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                .apply()
        }
    }

    /**
     * Checks if IndirectGestureService is currently enabled in system
     * Accessibility settings.
     *
     * Uses three independent checks in priority order so that no single
     * method's failure causes a false-negative on ColorOS / OPlus (Realme,
     * Oppo, OnePlus) where:
     *   • The service instance is null at cold-start because the OS binds
     *     background accessibility services lazily after the app process starts.
     *   • Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES may be empty or
     *     absent briefly right after the user toggles the service ON and
     *     returns to the app, before the OS commits the change.
     *   • AccessibilityManager.getEnabledAccessibilityServiceList() is the
     *     most up-to-date runtime view and is what the system settings UI
     *     itself relies on — making it the most reliable single source here.
     *
     * A persistent SharedPreferences flag (KEY_ACCESSIBILITY_GRANTED) is
     * also maintained: once any check returns true it is written, and it is
     * only cleared if ALL three checks agree the service is gone. This means
     * a cold-start timing window (instance not yet bound, Secure string not
     * yet readable) does NOT produce a spurious "not enabled" result after
     * the user has previously granted access.
     *
     * IMPORTANT: KEY_ACCESSIBILITY_GRANTED must never be trusted across a
     * reinstall — see reconcileAccessibilityGrantedFlagWithInstall(), which
     * is called once in onCreate() before this function is ever invoked, and
     * clears the flag if the persisted prefs don't belong to the current
     * install (e.g. restored via Android Auto Backup or an OEM data-restore
     * flow).
     */
    private fun isAccessibilityServiceEnabled(): Boolean {
        try {
            val ourClass = IndirectGestureService::class.java.name

            // ── Method 1: live instance (fastest; null on cold-start) ─────────
            if (IndirectGestureService.instance != null) {
                logD(TAG, "isAccessibilityServiceEnabled: instance bound ✓")
                prefs.edit()
                    .putBoolean(KEY_ACCESSIBILITY_GRANTED, true)
                    .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                    .apply()
                return true
            }

            // ── Method 2: AccessibilityManager runtime list ───────────────────
            // Most reliable on Android 13-15 and on ColorOS/OPlus; reflects
            // the live system state the same way the Settings UI does.
            try {
                val enabled = accessibilityManager.getEnabledAccessibilityServiceList(
                    AccessibilityServiceInfo.FEEDBACK_ALL_MASK
                )
                val foundViaAm = enabled.any { info ->
                    val id = info.id  // "com.example/.ServiceName"
                    id.contains(ourClass) || id.startsWith(packageName) && id.contains("IndirectGestureService")
                }
                if (foundViaAm) {
                    logD(TAG, "isAccessibilityServiceEnabled: found via AccessibilityManager ✓")
                    prefs.edit()
                        .putBoolean(KEY_ACCESSIBILITY_GRANTED, true)
                        .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                        .apply()
                    return true
                }
                logD(TAG, "isAccessibilityServiceEnabled: not in AccessibilityManager list")
            } catch (e: Exception) {
                logW(TAG, "isAccessibilityServiceEnabled: AccessibilityManager check failed", e)
            }

            // ── Method 3: Settings.Secure string ─────────────────────────────
            // Secondary; covers edge cases where AM list is briefly stale.
            try {
                val enabledStr = Settings.Secure.getString(
                    contentResolver,
                    Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
                )
                logD(TAG, "isAccessibilityServiceEnabled: enabledServices = $enabledStr")
                if (!enabledStr.isNullOrEmpty()) {
                    val fullName  = "$packageName/$ourClass"
                    val shortName = "$packageName/.IndirectGestureService"
                    if (enabledStr.contains(fullName) || enabledStr.contains(shortName)) {
                        logD(TAG, "isAccessibilityServiceEnabled: found via Settings.Secure ✓")
                        prefs.edit()
                            .putBoolean(KEY_ACCESSIBILITY_GRANTED, true)
                            .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                            .apply()
                        return true
                    }
                }
            } catch (e: Exception) {
                logW(TAG, "isAccessibilityServiceEnabled: Settings.Secure check failed", e)
            }

            // ── All runtime checks say not enabled ────────────────────────────
            // Before returning false, consult the persistent grant flag.
            // If the user granted the service in a previous session, don't
            // treat a SINGLE negative reading as a revocation — it may just be
            // a cold-start bind-lag / OPlus timing window. But don't trust the
            // flag forever either: count consecutive all-negative readings and
            // only clear the flag (real revocation) once we've seen enough of
            // them in a row that a timing window can no longer explain it.
            val previouslyGranted = prefs.getBoolean(KEY_ACCESSIBILITY_GRANTED, false)
            if (previouslyGranted) {
                val negativeStreak = prefs.getInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0) + 1
                if (negativeStreak < NEGATIVE_CHECKS_THRESHOLD) {
                    logD(TAG, "isAccessibilityServiceEnabled: runtime checks negative " +
                            "($negativeStreak/$NEGATIVE_CHECKS_THRESHOLD) but " +
                            "KEY_ACCESSIBILITY_GRANTED is set — treating as enabled (cold-start " +
                            "bind lag or OPlus timing window); will re-verify on next check")
                    prefs.edit().putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, negativeStreak).apply()
                    return true
                }
                logD(TAG, "isAccessibilityServiceEnabled: $negativeStreak consecutive " +
                        "all-negative checks — treating as a genuine revocation (user turned " +
                        "the service off in Settings), clearing KEY_ACCESSIBILITY_GRANTED")
                prefs.edit()
                    .putBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                    .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                    .apply()
                return false
            }

            logD(TAG, "isAccessibilityServiceEnabled: not enabled (no prior grant recorded)")
            return false

        } catch (e: Exception) {
            logE(TAG, "isAccessibilityServiceEnabled: unexpected exception", e)
            // Last resort: trust the persistent flag if available.
            return prefs.getBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                    || IndirectGestureService.instance != null
        }
    }

    /**
     * Clears the persistent accessibility-granted flag. Called when the user
     * returns from Accessibility settings and all runtime checks confirm the
     * service is truly disabled — i.e. they actively turned it off.
     * This is the only place KEY_ACCESSIBILITY_GRANTED is ever set to false.
     */
    private fun maybeClearAccessibilityGrantedFlag() {
        // Only revoke the flag if the instance is gone AND both runtime
        // sources agree the service is absent. This avoids clearing it
        // during a transient cold-start bind gap.
        if (IndirectGestureService.instance != null) return
        try {
            val enabled = accessibilityManager.getEnabledAccessibilityServiceList(
                AccessibilityServiceInfo.FEEDBACK_ALL_MASK
            )
            val ourClass = IndirectGestureService::class.java.name
            val stillPresent = enabled.any { info ->
                info.id.contains(ourClass) ||
                        info.id.startsWith(packageName) && info.id.contains("IndirectGestureService")
            }
            if (!stillPresent) {
                val enabledStr = Settings.Secure.getString(
                    contentResolver,
                    Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
                ) ?: ""
                val inSecure = enabledStr.contains("$packageName/") &&
                        enabledStr.contains("IndirectGestureService")
                if (!inSecure) {
                    logD(TAG, "maybeClearAccessibilityGrantedFlag: service confirmed absent — clearing flag")
                    prefs.edit()
                        .putBoolean(KEY_ACCESSIBILITY_GRANTED, false)
                        .putInt(KEY_CONSECUTIVE_NEGATIVE_CHECKS, 0)
                        .commit()
                }
            }
        } catch (e: Exception) {
            logW(TAG, "maybeClearAccessibilityGrantedFlag: check failed, leaving flag as-is", e)
        }
    }

    /**
     * Opens the system Accessibility settings screen.
     * Android 15 compatible - uses multiple fallback methods.
     */
    private fun openAccessibilitySettings() {
        try {
            // Method 1: Try to open our specific service settings (works on some devices)
            try {
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY)
                    addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS)
                }
                accessibilitySettingsLauncher.launch(intent)
                logD(TAG, "openAccessibilitySettings: launched via launcher")
                return
            } catch (e: Exception) {
                logW(TAG, "openAccessibilitySettings: launcher failed, trying startActivity", e)
            }

            // Method 2: Direct startActivity (fallback)
            val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            startActivity(intent)
            logD(TAG, "openAccessibilitySettings: launched via startActivity")

        } catch (e: Exception) {
            logE(TAG, "openAccessibilitySettings: all methods failed", e)

            // Show helpful error message with manual instructions
            AlertDialog.Builder(this)
                .setTitle("Cannot Open Settings")
                .setMessage("Please enable the accessibility service manually:\n\n" +
                        "1. Open Settings\n" +
                        "2. Go to Accessibility\n" +
                        "3. Find 'Q8 Force Bar Control'\n" +
                        "4. Toggle it ON\n" +
                        "5. Return to this app")
                .setPositiveButton("OK", null)
                .show()
        }
    }

    // =========================================================================
    // Splash screen
    // =========================================================================

    private fun showCustomSplash() {
        try {
            val dialog = Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
            splashDialog = dialog
            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(metrics)

            val splashView = SplashView(this, metrics.widthPixels, metrics.heightPixels)

            val gd = GestureDetector(this,
                object : GestureDetector.SimpleOnGestureListener() {
                    override fun onSingleTapUp(e: MotionEvent): Boolean {
                        splashView.triggerOutro { dismissSplash() }
                        return true
                    }
                    override fun onFling(
                        e1: MotionEvent?, e2: MotionEvent, vX: Float, vY: Float
                    ): Boolean {
                        splashView.triggerOutro { dismissSplash() }
                        return true
                    }
                })

            splashView.setOnTouchListener { _, event -> gd.onTouchEvent(event); true }
            dialog.setContentView(splashView)
            dialog.window?.apply {
                setLayout(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.MATCH_PARENT
                )
                addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS)
                addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
            }
            dialog.setCancelable(false)
            dialog.show()
            splashView.startAnimation()
        } catch (e: Exception) {
            logE(TAG, "showCustomSplash failed", e)
            inflateHomeScreen()
        }
    }

    private fun dismissSplash() {
        if (splashDismissed) return
        splashDismissed = true
        val dialog = splashDialog ?: run { inflateHomeScreen(); return }
        val root = dialog.window?.decorView ?: run {
            try { dialog.dismiss() } catch (_: Exception) {}
            inflateHomeScreen()
            return
        }
        root.animate().alpha(0f).setDuration(200).withEndAction {
            try { dialog.dismiss() } catch (e: Exception) { logE(TAG, "dismiss threw", e) }
            inflateHomeScreen()
        }.start()
    }

    // =========================================================================
    // Home screen
    // =========================================================================

    private fun inflateHomeScreen() {
        try {
            // Mark home screen shown FIRST, before requestedOrientation,
            // so onSaveInstanceState captures it during the recreation cycle.
            homeScreenShown = true

            // Force landscape before setContentView so the display context,
            // the layout, overlay-permission intent, and projectionLauncher.launch()
            // all happen while the Activity is already landscape — guaranteeing
            // the MediaProjection token carries the correct landscape geometry.
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE

            setContentView(R.layout.activity_main)

            dotOverlay = findViewById(R.id.dot_overlay_permission)
            dotCapture = findViewById(R.id.dot_screen_capture)
            dotRoi     = findViewById(R.id.dot_roi_calibrated)
            dotAccessibility = findViewById(R.id.dot_accessibility_service)
            accessibilityDotContainer = findViewById(R.id.accessibility_dot_container)
            dotTile = findViewById(R.id.dot_qs_tile)
            tileDotContainer = findViewById(R.id.tile_dot_container)
            btnStart   = findViewById(R.id.btn_start)

            // Tap the QS Tile row to (re-)trigger the add-tile request at
            // any time, e.g. if the user removed the tile later.
            tileDotContainer?.setOnClickListener {
                if (QEightTileService.isTileAdded(this)) {
                    Toast.makeText(
                        this, "Quick Settings tile already added", Toast.LENGTH_SHORT
                    ).show()
                } else {
                    requestAddQsTile()
                }
            }

            // Make accessibility dot clickable to open settings
            accessibilityDotContainer?.setOnClickListener {
                if (!isAccessibilityServiceEnabled()) {
                    showAccessibilityPromptDialog()
                } else {
                    Toast.makeText(
                        this,
                        "Accessibility service already enabled",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            // START button is always enabled — permissions are requested on tap
            btnStart?.isEnabled = true
            btnStart?.alpha = 1f
            btnStart?.setOnClickListener {
                try {
                    startOverlayService()
                } catch (e: Exception) {
                    Toast.makeText(
                        this, "Failed to start: ${e.message}", Toast.LENGTH_SHORT
                    ).show()
                }
            }

            // Register calibration receiver (unregistered in onDestroy)
            if (!calibrationReceiverRegistered) {
                LocalBroadcastManager.getInstance(this).registerReceiver(
                    calibrationReceiver,
                    IntentFilter("com.ashraf.qeight.CALIBRATION_SAVED")
                )
                calibrationReceiverRegistered = true
            }

            // Register service-stopped receiver (unregistered in onDestroy).
            // When OverlayService calls stopSelf() it broadcasts ACTION_SERVICE_STOPPED.
            // We null out projectionData so the next START tap re-requests a fresh
            // MediaProjection token instead of replaying the stale dead one.
            if (!serviceStoppedReceiverRegistered) {
                LocalBroadcastManager.getInstance(this).registerReceiver(
                    serviceStoppedReceiver,
                    IntentFilter(ACTION_SERVICE_STOPPED)
                )
                serviceStoppedReceiverRegistered = true
            }

            updateStatusDots()

            // ── Quick Settings Tile onboarding prompt ────────────────────────
            // Requests adding the QS tile once on first launch (API 33+ uses
            // the real system add-tile dialog); skipped entirely if the tile
            // is already added, and never re-triggered automatically after
            // the first attempt (the status row remains tappable to retry).
            maybePromptForTile()

        } catch (e: Exception) {
            logE(TAG, "inflateHomeScreen fatal", e)
            Toast.makeText(
                this, "App init failed: ${e.message}", Toast.LENGTH_LONG
            ).show()
        }
    }

    // =========================================================================
    // Quick Settings Tile onboarding
    // =========================================================================

    /**
     * On first app launch (after the start/home page is shown), asks the
     * system to add the Quick Settings tile — unless it's already been added,
     * in which case we do nothing (never re-prompt).
     *
     * API 33+ (Tiramisu): uses the real StatusBarManager.requestAddTileService
     * system dialog — the user gets a native "Add tile to Quick Settings?"
     * prompt with Add/Cancel, and onTileAdded() fires (persisting
     * KEY_TILE_ADDED) if they accept.
     *
     * API < 33: there is no programmatic add-tile API at all, so we fall back
     * to a one-time explanatory dialog pointing the user to the manual
     * QS-panel "Edit" flow instead.
     */
    private fun maybePromptForTile() {
        if (QEightTileService.isTileAdded(this)) {
            logD(TAG, "maybePromptForTile: tile already added, skipping")
            return
        }

        val alreadyShown = prefs.getBoolean(KEY_TILE_PROMPT_SHOWN, false)
        if (alreadyShown) {
            logD(TAG, "maybePromptForTile: prompt already shown once, skipping")
            return
        }

        // Delay slightly so the prompt appears *after* the home screen is
        // fully visible, rather than racing with the splash outro animation.
        window.decorView.postDelayed({
            try {
                requestAddQsTile()
            } catch (e: Exception) {
                logE(TAG, "maybePromptForTile: requestAddQsTile threw", e)
            }
        }, 400L)
    }

    /**
     * Fires the actual add-tile request (API 33+) or the manual-instructions
     * fallback dialog (API < 33). Marks KEY_TILE_PROMPT_SHOWN so the
     * auto-trigger in maybePromptForTile() never fires a second time,
     * regardless of whether the user accepted or dismissed it — the tile
     * status dot (driven by QEightTileService.isTileAdded()) remains the
     * live source of truth afterward, and the row stays tappable to retry.
     */
    private fun requestAddQsTile() {
        prefs.edit().putBoolean(KEY_TILE_PROMPT_SHOWN, true).apply()

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            QEightTileService.requestAddTile(this) { added ->
                runOnUiThread {
                    logD(TAG, "requestAddQsTile: result added=$added")
                    updateStatusDots()
                    if (!added) {
                        Toast.makeText(
                            this, "You can add the tile anytime from here", Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        } else {
            showManualTilePromptDialog()
        }
    }

    /** API < 33 fallback: explains the manual QS-panel Edit flow. */
    private fun showManualTilePromptDialog() {
        val dialogBuilder = AlertDialog.Builder(this, androidx.appcompat.R.style.Theme_AppCompat_Dialog_Alert)
            .setTitle(R.string.tile_prompt_title)
            .setMessage(Html.fromHtml(getString(R.string.tile_prompt_message), Html.FROM_HTML_MODE_LEGACY))
            .setPositiveButton(R.string.tile_prompt_positive) { dialog, _ -> dialog.dismiss() }
            .setNegativeButton(R.string.tile_prompt_negative) { dialog, _ -> dialog.dismiss() }
            .setCancelable(true)

        val dialog = dialogBuilder.create()
        dialog.setOnShowListener {
            val messageView = dialog.findViewById<TextView>(android.R.id.message)
            messageView?.movementMethod = LinkMovementMethod.getInstance()
        }
        dialog.show()
        logD(TAG, "Manual tile prompt dialog shown (API < 33)")
    }

    // =========================================================================
    // Accessibility permission prompting
    // =========================================================================

    /**
     * Shows a prompt to enable the accessibility service if it's not enabled.
     *
     * The auto-prompt fires at most once ever (KEY_ACCESSIBILITY_PROMPT_SHOWN
     * is written permanently on first show, never reset). After that, the user
     * can still tap the accessibility status row to reach settings manually.
     *
     * Critically, we do NOT prompt if KEY_ACCESSIBILITY_GRANTED is set —
     * that flag means the user has previously enabled the service and the
     * current "not enabled" reading is likely a cold-start bind lag on
     * ColorOS/OPlus, not a genuine revocation.
     */
    private fun maybePromptForAccessibility() {
        // If runtime checks (or the persistent grant flag) say it's enabled,
        // no prompt needed.
        if (isAccessibilityServiceEnabled()) {
            logD(TAG, "maybePromptForAccessibility: enabled — skipping prompt")
            return
        }

        // If the user previously granted access, the "not enabled" result is
        // most likely a cold-start bind lag. Don't prompt — let it resolve.
        if (prefs.getBoolean(KEY_ACCESSIBILITY_GRANTED, false)) {
            logD(TAG, "maybePromptForAccessibility: KEY_ACCESSIBILITY_GRANTED set — " +
                    "skipping prompt (assuming cold-start lag)")
            return
        }

        // Auto-prompt fires only once, ever.
        if (prefs.getBoolean(KEY_ACCESSIBILITY_PROMPT_SHOWN, false)) {
            logD(TAG, "maybePromptForAccessibility: prompt already shown once, skipping auto-prompt")
            return
        }

        // Write permanently with commit() so the flag survives a crash or
        // process kill between here and the dialog being shown.
        prefs.edit().putBoolean(KEY_ACCESSIBILITY_PROMPT_SHOWN, true).commit()

        // Delay slightly so the prompt appears after the home screen settles.
        window.decorView.postDelayed({
            try {
                showAccessibilityPromptDialog()
            } catch (e: Exception) {
                logE(TAG, "maybePromptForAccessibility: showAccessibilityPromptDialog threw", e)
            }
        }, 800L)
    }

    // =========================================================================
    // Battery optimization — required so ColorOS/Oplus (and other OEM-modified
    // Android builds) don't silently kill OverlayService shortly after
    // indirect mode starts capturing. A correctly-declared foreground
    // service with FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION can still be
    // killed by the OEM's own power manager unless the app is explicitly
    // exempted from battery optimization — standard Android APIs (foreground
    // service + notification) are necessary but not sufficient on these
    // OEM builds.
    // =========================================================================

    /**
     * True if this app is already exempted from Android's battery
     * optimizations (i.e. Doze/App Standby won't restrict it).
     *
     * Note: this only reflects *stock* Android's battery optimization state.
     * OEM-specific power managers (ColorOS "Startup Manager", "Background
     * Freeze", "Abnormal App Optimization", etc.) are separate systems with
     * no public API to query or request programmatically — this check and
     * prompt can only get the user to the right *stock* settings screen;
     * OEM-specific toggles still need to be set manually by the user via
     * their device's own battery/security app.
     */
    private fun isIgnoringBatteryOptimizations(): Boolean {
        return try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            val ignoring = pm.isIgnoringBatteryOptimizations(packageName)
            logD(TAG, "isIgnoringBatteryOptimizations: $ignoring")
            ignoring
        } catch (e: Exception) {
            logE(TAG, "isIgnoringBatteryOptimizations: exception, assuming false", e)
            false
        }
    }

    /**
     * Shows a prompt to exempt the app from battery optimization if it
     * isn't already exempted. Only shown once per app launch, and only
     * after the accessibility prompt has had its chance to show first
     * (avoids stacking two dialogs back-to-back on first run).
     */
    private fun maybePromptForBatteryOptimization() {
        if (isIgnoringBatteryOptimizations()) {
            logD(TAG, "maybePromptForBatteryOptimization: already ignoring optimizations, skipping")
            return
        }

        val alreadyShownThisSession = prefs.getBoolean(KEY_BATTERY_PROMPT_SHOWN, false)
        if (alreadyShownThisSession) {
            logD(TAG, "maybePromptForBatteryOptimization: already shown this session, skipping")
            return
        }

        prefs.edit().putBoolean(KEY_BATTERY_PROMPT_SHOWN, true).apply()

        // Slightly longer delay than the accessibility prompt so the two
        // never overlap if both are due to show on the same cold start.
        window.decorView.postDelayed({
            try {
                showBatteryOptimizationPromptDialog()
            } catch (e: Exception) {
                logE(TAG, "maybePromptForBatteryOptimization: showBatteryOptimizationPromptDialog threw", e)
            }
        }, 1400L)
    }

    /**
     * Shows the battery-optimization exemption dialog with instructions.
     * Mirrors [showAccessibilityPromptDialog]'s structure and style.
     */
    private fun showBatteryOptimizationPromptDialog() {
        val dialogBuilder = AlertDialog.Builder(this, androidx.appcompat.R.style.Theme_AppCompat_Dialog_Alert)
            .setTitle("Keep Qeight running reliably")
            .setMessage(
                "Some phone brands (especially Oppo/OnePlus/Realme ColorOS, " +
                        "Xiaomi, and Vivo devices) can silently stop Qeight's overlay " +
                        "shortly after Indirect Mode starts, even with all permissions " +
                        "granted.\n\n" +
                        "To prevent this, please:\n" +
                        "1. Tap \"Open Settings\" below and choose \"Don't optimize\" / " +
                        "\"Allow background activity\" for Qeight.\n\n" +
                        "2. If your phone has a separate \"Startup Manager\", " +
                        "\"Autostart\", or \"Protected Apps\" screen (common on Oppo/" +
                        "OnePlus/Realme/Xiaomi), please also enable Qeight there — " +
                        "this step can't be opened automatically and must be found " +
                        "manually in your phone's Settings or built-in Security app."
            )
            .setPositiveButton("Open Settings") { dialog, _ ->
                logD(TAG, "Battery prompt: user tapped 'Open Settings'")
                dialog.dismiss()
                openBatteryOptimizationSettings()
            }
            .setNegativeButton("Not Now") { dialog, _ ->
                logD(TAG, "Battery prompt dismissed via 'Not Now'")
                dialog.dismiss()
            }
            .setCancelable(true)

        dialogBuilder.create().show()
        logD(TAG, "Battery optimization prompt dialog shown")
    }

    /**
     * Opens the system's battery-optimization exemption request screen for
     * this app. Uses the direct-request intent first (shows the system's
     * native "Allow"/"Deny" confirmation for this specific app); falls back
     * to the general battery-optimization list if that's unavailable.
     */
    private fun openBatteryOptimizationSettings() {
        try {
            val intent = Intent(
                Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                Uri.parse("package:$packageName")
            )
            startActivity(intent)
            logD(TAG, "openBatteryOptimizationSettings: launched direct-request intent")
        } catch (e: Exception) {
            logW(TAG, "openBatteryOptimizationSettings: direct-request intent failed, " +
                    "falling back to general battery optimization list", e)
            try {
                startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            } catch (e2: Exception) {
                logE(TAG, "openBatteryOptimizationSettings: fallback also failed", e2)
                Toast.makeText(
                    this, "Couldn't open battery settings — please find " +
                            "Qeight in your phone's battery settings manually",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    /**
     * Shows the accessibility permission dialog with instructions.
     */
    private fun showAccessibilityPromptDialog() {
        val dialogBuilder = AlertDialog.Builder(this, androidx.appcompat.R.style.Theme_AppCompat_Dialog_Alert)
            .setTitle(R.string.accessibility_prompt_title)
            .setMessage(Html.fromHtml(getString(R.string.accessibility_prompt_message), Html.FROM_HTML_MODE_LEGACY))
            .setPositiveButton(R.string.accessibility_prompt_positive) { dialog, _ ->
                logD(TAG, "Accessibility prompt: user tapped 'Open Settings'")
                dialog.dismiss()
                openAccessibilitySettings()
            }
            .setNegativeButton(R.string.accessibility_prompt_negative) { dialog, _ ->
                logD(TAG, "Accessibility prompt dismissed via 'Not Now'")
                dialog.dismiss()
            }
            .setCancelable(true)

        val dialog = dialogBuilder.create()
        dialog.setOnShowListener {
            val messageView = dialog.findViewById<TextView>(android.R.id.message)
            messageView?.movementMethod = LinkMovementMethod.getInstance()
        }
        dialog.show()
        logD(TAG, "Accessibility prompt dialog shown")
    }

    // =========================================================================
    // Status dots
    // =========================================================================

    /**
     * Updates the four status dots and nudges the QS tile.
     *
     * Dot 1 — overlay permission (SYSTEM_ALERT_WINDOW)
     * Dot 2 — screen-capture token held in memory
     * Dot 3 — ROI calibration saved in SharedPreferences
     * Dot 4 — Accessibility service enabled
     *
     * The START button is always enabled — never gated on dot state.
     */
    fun updateStatusDots() {
        try {
            val overlayOk = Settings.canDrawOverlays(this)
            val captureOk = (projectionData != null)
            val roiOk     = prefs.getString(KEY_ROI_CALIBRATION, null) != null
            val accessibilityOk = isAccessibilityServiceEnabled()
            val tileOk    = QEightTileService.isTileAdded(this)

            setDotColor(dotOverlay, overlayOk)
            setDotColor(dotCapture, captureOk)
            setDotColor(dotRoi,     roiOk)
            setDotColor(dotAccessibility, accessibilityOk)
            setDotColor(dotTile, tileOk)

            // Keep button always enabled — never disable it based on dot state
            btnStart?.isEnabled = true
            btnStart?.alpha     = 1f

            // ── Tile refresh ──────────────────────────────────────────────────
            QEightTileService.requestTileUpdate(this)

        } catch (e: Exception) {
            logE(TAG, "updateStatusDots", e)
        }
    }

    private fun setDotColor(dot: View?, green: Boolean) {
        dot ?: return
        dot.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.parseColor(if (green) COLOR_GREEN else COLOR_RED))
        }
    }

    // =========================================================================
    // Permission gating + service start
    // =========================================================================

    /**
     * Sequential permission gatekeeper triggered by START button tap.
     *
     * Flow:
     *   0. Accessibility missing → prompt → wait for user
     *   1. Overlay missing    → request overlay → (callback) → request projection → start
     *   2. Projection missing → request projection → (callback) → start
     *   3. All granted        → start immediately
     *
     * Accessibility check uses [isAccessibilityServiceEnabled] which consults
     * the persistent KEY_ACCESSIBILITY_GRANTED flag as a tiebreaker — so a
     * cold-start bind lag on ColorOS/OPlus (where the service instance hasn't
     * connected yet but was granted in a prior session) does NOT block the
     * start flow or re-prompt the user.
     */
    private fun startOverlayService() {
        // Check accessibility service first (can't auto-request, must prompt).
        // isAccessibilityServiceEnabled() returns true if any runtime check
        // confirms the service is on, OR if the persistent grant flag is set
        // (covers cold-start bind lag on ColorOS/OPlus).
        if (!isAccessibilityServiceEnabled()) {
            logD(TAG, "startOverlayService: accessibility service not enabled, prompting")
            showAccessibilityPromptDialog()
            return
        }

        if (!Settings.canDrawOverlays(this)) {
            pendingServiceStart = true
            try {
                overlayPermissionLauncher.launch(
                    Intent(
                        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                )
            } catch (e: Exception) {
                Toast.makeText(
                    this, "Failed to open overlay settings", Toast.LENGTH_SHORT
                ).show()
                pendingServiceStart = false
            }
            return
        }

        if (projectionData == null) {
            pendingServiceStart = true
            try {
                projectionLauncher.launch(projectionManager.createScreenCaptureIntent())
            } catch (e: Exception) {
                Toast.makeText(
                    this, "Failed to request screen capture", Toast.LENGTH_SHORT
                ).show()
                pendingServiceStart = false
            }
            return
        }

        doStartService()
    }

    /**
     * Performs the actual service start + 8 Ball Pool launch.
     * Only called after all permissions are confirmed.
     */
    private fun doStartService() {
        startForegroundService(Intent(this, OverlayService::class.java).apply {
            projectionData?.let { putExtra(OverlayService.EXTRA_PROJECTION_DATA, it) }
            putExtra(OverlayService.EXTRA_PROJECTION_RESULT_CODE, RESULT_OK)
        })

        try {
            var launched = false

            packageManager.getLaunchIntentForPackage("com.miniclip.eightballpool")?.let {
                it.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                try { startActivity(it); launched = true } catch (_: Exception) {}
            }

            if (!launched) try {
                startActivity(Intent(Intent.ACTION_MAIN).apply {
                    addCategory(Intent.CATEGORY_LAUNCHER)
                    setPackage("com.miniclip.eightballpool")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                })
                launched = true
            } catch (_: Exception) {}

            if (!launched) try {
                startActivity(
                    Intent(
                        Intent.ACTION_VIEW,
                        Uri.parse("market://details?id=com.miniclip.eightballpool")
                    ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
                )
            } catch (_: Exception) {
                Toast.makeText(
                    this, "Please install 8 Ball Pool", Toast.LENGTH_LONG
                ).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "8 Ball Pool not installed", Toast.LENGTH_SHORT).show()
        }

        moveTaskToBack(true)
    }

    // =========================================================================
    // SplashView
    // =========================================================================

    inner class SplashView(
        context: Context,
        private val screenW: Int,
        private val screenH: Int
    ) : View(context) {

        private val density = resources.displayMetrics.density

        private val charBoxSize     = 100f
        private val textSize        = 94f
        private val textStrokeWidth = 9f

        private val eightRestSize = screenH * 0.90f
        private val eightScaleMax = 6.0f
        private val eightCenterX  = screenW * 0.5f
        private val eightCenterY  = screenH * 0.46f

        private var eightScale = eightScaleMax
        private var eightAlpha = 0

        private val layerPaint = Paint()

        private val eightFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            typeface  = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
            shader    = LinearGradient(
                0f, 0f, 0f, screenH.toFloat(),
                intArrayOf(Color.parseColor("#FF4500"), Color.parseColor("#FF8C00")),
                floatArrayOf(0f, 1f), Shader.TileMode.CLAMP
            )
        }

        private val eightGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            typeface   = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign  = Paint.Align.CENTER
            maskFilter = android.graphics.BlurMaskFilter(
                50f * density, android.graphics.BlurMaskFilter.Blur.OUTER
            )
            color = Color.parseColor("#FFD700")
        }

        private val sweepGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style       = Paint.Style.STROKE
            strokeWidth = 18f * density
            color       = Color.parseColor("#FFD700")
            strokeCap   = Paint.Cap.ROUND
            maskFilter  = android.graphics.BlurMaskFilter(
                26f * density, android.graphics.BlurMaskFilter.Blur.NORMAL
            )
        }

        private val eightYOff   = eightCenterY + eightRestSize * 0.35f
        private val eightHalfW  = eightRestSize * 0.22f
        private val eightTopY   = eightCenterY - eightRestSize * 0.37f
        private val eightMidY   = eightCenterY
        private val eightBtmY   = eightCenterY + eightRestSize * 0.35f
        private val topOvalRect = RectF(
            eightCenterX - eightHalfW, eightTopY,
            eightCenterX + eightHalfW, eightMidY
        )
        private val botOvalRect = RectF(
            eightCenterX - eightHalfW, eightMidY,
            eightCenterX + eightHalfW, eightBtmY
        )

        private val logoBoxSize = screenW * 0.52f
        private val logoBoxLeft = eightCenterX - logoBoxSize * 0.5f
        private val logoBoxTop  = eightCenterY - logoBoxSize * 0.5f
        private val boxRect     = RectF(
            logoBoxLeft, logoBoxTop,
            logoBoxLeft + logoBoxSize, logoBoxTop + logoBoxSize
        )

        private val placeholderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#1A1A1A")
        }
        private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style       = Paint.Style.STROKE
            color       = Color.parseColor("#FF6D00")
            strokeWidth = 4f
        }

        private val word1 = "C R Ξ Ʌ T Ξ D"
        private val word2 = "B Y"
        private val word3 = "Ʌ S H R Ʌ F"

        private val textTypeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)

        private val textY: Float

        private var activeWord     = -1
        private var wordAlpha      = 0f
        private var expandProgress = 0f
        private var goldPos        = 2f

        private var glowSweepProgress = 0f
        private var logoAlpha         = 0

        private var outroStarted  = false
        private var outroCallback: (() -> Unit)? = null

        private var logoBitmap: Bitmap? = null

        init {
            setLayerType(LAYER_TYPE_SOFTWARE, null)
            try {
                context.assets.open("splash_logo.png").use { s ->
                    logoBitmap = BitmapFactory.decodeStream(s)
                }
            } catch (e: Exception) {
                logW(TAG, "splash_logo.png not found")
            }

            val logoBottom = logoBoxTop + logoBoxSize
            val remaining  = screenH - logoBottom
            textY = logoBottom + remaining * 0.45f
        }

        fun startAnimation() { playNetflixIntro() }

        fun triggerOutro(onDone: () -> Unit) {
            if (outroStarted) return
            outroStarted  = true
            outroCallback = onDone
            playOutro()
        }

        // ----- animation stages -----

        private fun playNetflixIntro() {
            eightAlpha = 0
            eightScale = eightScaleMax
            val scaleInterp = DecelerateInterpolator(2.5f)
            val alphaInterp = DecelerateInterpolator(1.8f)

            ValueAnimator.ofFloat(0f, 1f).apply {
                duration     = 1600
                interpolator = LinearInterpolator()
                addUpdateListener { v ->
                    val p  = v.animatedValue as Float
                    val sp = scaleInterp.getInterpolation(p)
                    eightScale = eightScaleMax - sp * (eightScaleMax - 1f)
                    val ap = alphaInterp.getInterpolation(p)
                    eightAlpha = (ap * 255f).toInt().coerceIn(0, 255)
                    invalidate()
                }
                doOnEnd { startGlowSweep() }
            }.start()
        }

        private fun startGlowSweep() {
            ValueAnimator.ofFloat(0f, 1f).apply {
                duration     = 700
                interpolator = LinearInterpolator()
                addUpdateListener {
                    glowSweepProgress = animatedValue as Float
                    invalidate()
                }
                doOnEnd { startLogoFade() }
            }.start()
        }

        private fun startLogoFade() {
            ValueAnimator.ofInt(0, 255).apply {
                duration     = 500
                interpolator = DecelerateInterpolator(1.5f)
                addUpdateListener { logoAlpha = animatedValue as Int; invalidate() }
                doOnEnd { startWord1Sequence() }
            }.start()
        }

        private fun runWordSequence(
            wordIndex: Int,
            expandMs:  Long,
            goldMs:    Long,
            onDone:    () -> Unit
        ) {
            activeWord     = wordIndex
            expandProgress = 0f
            goldPos        = 2f
            wordAlpha      = 0f

            ValueAnimator.ofFloat(0f, 1f).apply {
                duration     = 150
                interpolator = LinearInterpolator()
                addUpdateListener { wordAlpha = it.animatedValue as Float; invalidate() }
                doOnEnd {
                    ValueAnimator.ofFloat(0f, 1f).apply {
                        duration     = expandMs
                        interpolator = DecelerateInterpolator(2f)
                        addUpdateListener {
                            expandProgress = it.animatedValue as Float
                            invalidate()
                        }
                        doOnEnd {
                            goldPos = 0f
                            ValueAnimator.ofFloat(0f, 1f).apply {
                                duration     = goldMs
                                interpolator = LinearInterpolator()
                                addUpdateListener {
                                    goldPos = it.animatedValue as Float
                                    invalidate()
                                }
                                doOnEnd { onDone() }
                            }.start()
                        }
                    }.start()
                }
            }.start()
        }

        private fun startWord1Sequence() {
            runWordSequence(wordIndex = 0, expandMs = 700L, goldMs = 500L) {
                ValueAnimator.ofFloat(1f, 0f).apply {
                    duration     = 150
                    interpolator = LinearInterpolator()
                    addUpdateListener { wordAlpha = it.animatedValue as Float; invalidate() }
                    doOnEnd { startWord2Sequence() }
                }.start()
            }
        }

        private fun startWord2Sequence() {
            runWordSequence(wordIndex = 1, expandMs = 500L, goldMs = 350L) {
                ValueAnimator.ofFloat(1f, 0f).apply {
                    duration     = 150
                    interpolator = LinearInterpolator()
                    addUpdateListener { wordAlpha = it.animatedValue as Float; invalidate() }
                    doOnEnd { startWord3Sequence() }
                }.start()
            }
        }

        private fun startWord3Sequence() {
            runWordSequence(wordIndex = 2, expandMs = 700L, goldMs = 450L) {
                postDelayed({ triggerOutro { dismissSplash() } }, 500L)
            }
        }

        private fun playOutro() {
            ValueAnimator.ofFloat(wordAlpha, 0f).apply {
                duration     = 250
                interpolator = LinearInterpolator()
                addUpdateListener { wordAlpha = it.animatedValue as Float; invalidate() }
                doOnEnd {
                    activeWord = -1
                    ValueAnimator.ofInt(logoAlpha, 0).apply {
                        duration     = 200
                        interpolator = LinearInterpolator()
                        addUpdateListener {
                            logoAlpha = it.animatedValue as Int
                            invalidate()
                        }
                        doOnEnd { playNetflixOutro() }
                    }.start()
                }
            }.start()
        }

        private fun playNetflixOutro() {
            val scaleInterp = AccelerateInterpolator(2.0f)
            val alphaInterp = AccelerateInterpolator(1.6f)

            ValueAnimator.ofFloat(0f, 1f).apply {
                duration     = 700
                interpolator = LinearInterpolator()
                addUpdateListener {
                    val p  = animatedValue as Float
                    val sp = scaleInterp.getInterpolation(p)
                    eightScale = 1f + sp * (eightScaleMax - 1f)
                    val ap = alphaInterp.getInterpolation(p)
                    eightAlpha = ((1f - ap) * 255f).toInt().coerceIn(0, 255)
                    invalidate()
                }
                doOnEnd { outroCallback?.invoke() }
            }.start()
        }

        // ----- drawing -----

        override fun onDraw(canvas: Canvas) {
            canvas.drawColor(Color.BLACK)
            draw8(canvas)

            if (logoAlpha > 0) {
                canvas.saveLayerAlpha(boxRect, logoAlpha)
                drawLogoBox(canvas)
                canvas.restore()
            }

            if (activeWord in 0..2 && wordAlpha > 0f) {
                drawCurrentWord(canvas)
            }
        }

        private fun draw8(canvas: Canvas) {
            layerPaint.alpha = eightAlpha
            val bounds = RectF(0f, 0f, screenW.toFloat(), screenH.toFloat())
            canvas.saveLayer(bounds, layerPaint)

            canvas.save()
            canvas.scale(eightScale, eightScale, eightCenterX, eightCenterY)

            eightGlowPaint.textSize = eightRestSize
            canvas.drawText("8", eightCenterX, eightYOff, eightGlowPaint)

            eightFillPaint.textSize = eightRestSize
            canvas.drawText("8", eightCenterX, eightYOff, eightFillPaint)

            canvas.restore()

            if (glowSweepProgress > 0f && eightScale < 1.05f && eightAlpha > 200) {
                drawGlowSweep(canvas)
            }

            canvas.restore()
        }

        private fun drawGlowSweep(canvas: Canvas) {
            val angle = glowSweepProgress * 360f
            canvas.drawPath(
                Path().apply { addArc(topOvalRect, 90f, angle) }, sweepGlowPaint
            )
            canvas.drawPath(
                Path().apply { addArc(botOvalRect, 270f, angle) }, sweepGlowPaint
            )
        }

        private fun drawLogoBox(canvas: Canvas) {
            canvas.drawRoundRect(boxRect, 16f, 16f, placeholderPaint)
            if (logoBitmap != null) {
                canvas.drawBitmap(
                    logoBitmap!!, null, boxRect, Paint(Paint.FILTER_BITMAP_FLAG)
                )
            } else {
                canvas.drawText(
                    "Q",
                    logoBoxLeft + logoBoxSize * 0.5f,
                    logoBoxTop  + logoBoxSize * 0.65f,
                    Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color     = Color.parseColor("#FF6D00")
                        textSize  = logoBoxSize * 0.5f
                        textAlign = Paint.Align.CENTER
                        typeface  = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                    }
                )
            }
            canvas.drawRoundRect(boxRect, 16f, 16f, borderPaint)
        }

        private fun drawCurrentWord(canvas: Canvas) {
            val currentText = when (activeWord) {
                0 -> word1
                1 -> word2
                2 -> word3
                else -> return
            }

            val glyphs = currentText.filter { it != ' ' }.map { it.toString() }
            val n = glyphs.size
            if (n == 0) return

            val totalBoxWidth = n * charBoxSize
            val maxGap = if (n > 1) {
                ((screenW - totalBoxWidth) / (n - 1)).coerceAtLeast(0f)
            } else 0f
            val currentGap = expandProgress * maxGap

            val totalWidth = (totalBoxWidth + currentGap * (if (n > 1) n - 1 else 0))
                .coerceAtMost(screenW.toFloat())

            val startX = ((screenW - totalWidth) * 0.5f).coerceAtLeast(0f)

            val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                typeface    = textTypeface
                style       = Paint.Style.FILL_AND_STROKE
                strokeWidth = textStrokeWidth
                textAlign   = Paint.Align.CENTER
                textSize    = this@SplashView.textSize
            }

            val goldBandWidth =
                if (n > 1) (1.5f / (n - 1)).coerceIn(0.15f, 0.45f) else 0.5f

            for (i in glyphs.indices) {
                val charPos = if (n > 1) i.toFloat() / (n - 1) else 0.5f
                val dist    = abs(charPos - goldPos)
                val a       = (wordAlpha * 255f).toInt().coerceIn(0, 255)

                val charColor:    Int
                val shadowColor:  Int
                val shadowRadius: Float

                when {
                    dist < goldBandWidth -> {
                        val intensity = 1f - dist / goldBandWidth
                        charColor    = Color.argb(a, 255, 215, 0)
                        val glowA    = (intensity * a * 0.8f).toInt().coerceIn(0, 255)
                        shadowColor  = Color.argb(glowA, 255, 240, 80)
                        shadowRadius = 32f * intensity
                    }
                    charPos > goldPos -> {
                        val dimA     = (a * 0.3f).toInt().coerceIn(0, 255)
                        charColor    = Color.argb(dimA, 120, 120, 120)
                        shadowColor  = Color.TRANSPARENT
                        shadowRadius = 0f
                    }
                    else -> {
                        charColor    = Color.argb(a, 240, 240, 240)
                        shadowColor  = Color.TRANSPARENT
                        shadowRadius = 0f
                    }
                }

                textPaint.color = charColor
                if (shadowRadius > 0f) {
                    textPaint.setShadowLayer(shadowRadius, 0f, 0f, shadowColor)
                } else {
                    textPaint.setShadowLayer(0f, 0f, 0f, Color.TRANSPARENT)
                }

                val boxCentreX =
                    startX + i * (charBoxSize + currentGap) + charBoxSize * 0.5f
                val clampedX = boxCentreX.coerceIn(
                    charBoxSize * 0.5f, screenW - charBoxSize * 0.5f
                )

                canvas.drawText(glyphs[i], clampedX, textY, textPaint)
            }
        }

        // ----- helpers -----

        private fun ValueAnimator.doOnEnd(block: () -> Unit) {
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) = block()
            })
        }
    }
}