/**
 * IndirectModeController.kt
 *
 * Manages the "Indirect Shot" (multi-cushion) overlay mode for the Q8 app.
 *
 * State machine (simplified, linear):
 *   IDLE → LOCKING_CUE_BALL → LOCKING_TARGET_BALL → AIMING → LOCKED
 *
 * D-pad + table-tap driven marker placement for ball locking
 * (LOCKING_CUE_BALL / LOCKING_TARGET_BALL): the dpad cross nudges the
 * active ring marker in pixels-per-step, sized by the four step-size
 * buttons flanking Up/Down.
 *
 * AIMING uses a completely different interaction model: the dpad cross is
 * fully hidden. Aiming is driven purely by drag-to-aim directly on the
 * pool table region (see onTableDragTouch) — a drag continuously re-aims
 * using a sensitivity-scaled delta of the raw angle-to-finger direction.
 * Tapping the table has no effect on the aim angle; only movement while
 * the finger is down rotates the shot. The sensitivity scale is chosen via
 * a vertical strip of the same four
 * values (10, 1, .1, .01), centered where the dpad's Left arrow sits in
 * the locking states. Shot power/commit is a separate gesture entirely:
 * dragging the force-bar overlay sets power live and releasing it (while
 * the evaluated shot pots) fires the shot via the accessibility gesture
 * service and transitions to LOCKED.
 *
 * All internal coordinates tracked as Double; rounded to Int only at:
 *   (a) WindowManager LayoutParams x/y for ring marker views
 *   (b) Final args passed to QeightJNI.updateIndirectAim
 */
package com.ashraf.qeight

import android.content.Context
import android.content.pm.ActivityInfo
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.PI
import kotlin.math.roundToInt

// ---------------------------------------------------------------------------
// DashedRingView — hollow dashed/dotted circle marker (visual only, not touchable)
// ---------------------------------------------------------------------------

private class DashedRingView(context: Context, private val strokeColor: Int) : View(context) {

    companion object {
        // Rendering-only radius for the cue-ball / target-ball ring markers.
        // Purely visual — does NOT feed the native solver. Physics still uses
        // the separate BALL_RADIUS_PX constant unchanged.
        const val RING_RADIUS_PX = 24f
    }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 4f
        color       = strokeColor
        pathEffect  = DashPathEffect(floatArrayOf(8f, 5f), 0f)
    }

    init { setWillNotDraw(false) }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width  / 2f
        val cy = height / 2f
        canvas.drawCircle(cx, cy, RING_RADIUS_PX, paint)
    }
}

// ---------------------------------------------------------------------------
// ForceBarView — vertical, touchable force-bar track (background track +
// fill indicator + drag-position marker). Visual only; touch handling is
// wired externally via setOnTouchListener by the controller.
// ---------------------------------------------------------------------------

private class ForceBarView(context: Context) : View(context) {

    private var power: Double = 0.0   // [0.0, 1.0], 0 = empty, 1 = full

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(140, 40, 40, 40)
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FF6D00")
    }
    private val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    init { setWillNotDraw(false) }

    /** Updates the current power fraction [0,1] and triggers a redraw. */
    fun setPower(power: Double) {
        this.power = power.coerceIn(0.0, 1.0)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        // Background track — full height.
        canvas.drawRect(0f, 0f, w, h, trackPaint)

        // Fill indicator — grows downward from the top as power increases
        // (top of track = 0.0 = no power, bottom = 1.0 = full power).
        val fillBottom = h * power.toFloat()
        canvas.drawRect(0f, 0f, w, fillBottom, fillPaint)

        // Marker at the current fill edge (bottom of the filled region).
        val markerY = fillBottom
        canvas.drawRect(0f, (markerY - 4f).coerceAtLeast(0f), w, (markerY + 4f).coerceAtMost(h), markerPaint)
    }
}


// ---------------------------------------------------------------------------
// IndirectModeController
// ---------------------------------------------------------------------------

class IndirectModeController(
    private val context: Context,
    private val windowManager: WindowManager,
    private val screenCaptureManager: ScreenCaptureManager,
    // Change 3: captureWidth/captureHeight/captureScale are now var fields
    // so updateCaptureGeometry() can update them on rotation without
    // reconstructing this controller. The constructor values remain the
    // authoritative starting geometry for the lifetime of this instance;
    // OverlayService calls updateCaptureGeometry() if a rotation occurs
    // mid-session to keep pixel↔cm conversions consistent.
    captureWidth:  Int,
    captureHeight: Int,
    captureScale:  Float,
    private val floatingButtonSizePx:        Int,
    private val floatingButtonRightMarginPx: Int,
    private val nativeScreenWidth:  Int,
    private val nativeScreenHeight: Int,
    private val poolTableLeft:   Int,
    private val poolTableTop:    Int,
    private val poolTableRight:  Int,
    private val poolTableBottom: Int,
    private val poolPocketR:       Int,
    private val poolPocketNsShift: Int,
    private val onExit: () -> Unit
) {

    // Change 3: mutable capture geometry fields, initialized from constructor
    // params and updatable via updateCaptureGeometry(). All existing code
    // that previously read captureWidth/captureHeight/captureScale as val
    // constructor properties continues to work unchanged — the field names
    // are identical; only the storage becomes var instead of val.
    private var captureWidth:  Int   = captureWidth
    private var captureHeight: Int   = captureHeight
    private var captureScale:  Float = captureScale

    /**
     * Called by OverlayService when a rotation mid-session produces new
     * capture dimensions, so this controller's pixel↔cm conversions
     * (onCleanFrame crop math, onLockClicked captureScale multiplications,
     * toPointList() division) stay consistent with the live capture geometry
     * rather than using stale constructor values.
     *
     * Safe to call from any thread — all three fields are primitives written
     * atomically on the JVM; all their read-sites (onCleanFrame, onLockClicked,
     * toPointList) already run on threads that do not race with each other in
     * a way that would corrupt a mid-update read (onCleanFrame runs on the
     * capture thread, onLockClicked/toPointList run on the main thread, and
     * OverlayService calls this from the main thread as well).
     */
    fun updateCaptureGeometry(newWidth: Int, newHeight: Int, newScale: Float) {
        logD("updateCaptureGeometry ${captureWidth}x${captureHeight}@${captureScale}" +
                " → ${newWidth}x${newHeight}@${newScale}")
        captureWidth  = newWidth
        captureHeight = newHeight
        captureScale  = newScale
    }

    // -----------------------------------------------------------------------
    // Companion
    // -----------------------------------------------------------------------

    companion object {
        private const val TAG             = "IndirectModeController"
        private const val LOGGING_ENABLED = true

        private fun logD(msg: String) { if (LOGGING_ENABLED) Log.d(TAG, msg) }
        private fun logW(msg: String) { Log.w(TAG, msg) }
        private fun logE(msg: String, t: Throwable? = null) {
            if (t != null) Log.e(TAG, msg, t) else Log.e(TAG, msg)
        }

        const val BALL_RADIUS_PX  = 24f
        const val MAGNIFIER_ZOOM  = 7f

        private const val TAP_SLOP_PX = 12f

        // D-pad cluster — fixed rect (derived from the previous Copy-tool output).
        // The cross (arrows + lock) keeps this exact footprint; the four step-size
        // buttons are laid out beside/above/below it, so the overall container is
        // wider/taller than DPAD_SIZE alone.
        private const val DPAD_X    = 1955
        private const val DPAD_Y    = 415
        private const val DPAD_SIZE = 342

        // Center-to-center distance (px) from the Lock button (cross center)
        // to each of Up/Down/Left/Right. Requested spacing increase — arms
        // no longer sit directly adjacent to the center cell.
        private const val DPAD_ARM_OFFSET_PX = 50

        // Step-size choices selectable via the four small buttons flanking the
        // Up/Down arrows. Default is 1.
        // NOTE: While state == AIMING, these values are interpreted as
        // DEGREES PER STEP rather than pixels per step. Same buttons, same
        // labels, dual meaning depending on state.
        private const val STEP_TINY   = 0.01  // above Up
        private const val STEP_SMALL  = 0.1   // beside STEP_TINY
        private const val STEP_NORMAL = 1.0   // below Down
        private const val STEP_LARGE  = 10.0  // beside STEP_NORMAL

        // AIMING-only override for the sensitivity strip's "10"-labelled
        // button (btnSensLarge / wireSensButton). Locking-state step buttons
        // (btnStepLarge) keep using STEP_LARGE unchanged; only the AIMING
        // rotation-sensitivity effect of the "10" button uses this value.
        private const val SENS_LARGE_AIMING_VALUE = 5.0

        // Lateral gap between each step-size button pair.
        private const val STEP_BTN_GAP_PX = 30

        // Hold-repeat cadence for press-and-hold continuous movement.
        private const val DPAD_REPEAT_MS = 16L

        // Press-and-hold moves at this multiple of a single tap/click's step
        // size, per request ("hold moves twice as fast as clicking once").
        // Applied only to repeat ticks — the initial tap-step stays at 1x.
        private const val HOLD_SPEED_MULTIPLIER = 2.0

        // Force-bar gesture constants — captured from a fixed-resolution
        // reference screenshot, same as the other calibrated screen
        // coordinates in this file. Scaled against nativeScreenWidth /
        // nativeScreenHeight before use (see scaledForceBarX/TopY/BottomY/Width).
        private const val FORCE_BAR_REF_WIDTH  = 2400
        private const val FORCE_BAR_REF_HEIGHT = 1080

        private const val FORCE_BAR_X        = 203
        private const val FORCE_BAR_TOP_Y    = 345   // min power (0.0)
        private const val FORCE_BAR_BOTTOM_Y = 895   // max power (1.0)
        private const val FORCE_BAR_WIDTH    = 64

        // Change 2: maximum milliseconds to wait for an in-flight gesture to
        // complete before proceeding with teardown. Bounded so stop() never
        // hangs indefinitely even if the gesture service exposes no completion
        // callback — 500 ms is well above any realistic gesture dispatch
        // latency and small enough to be imperceptible to the user.
        private const val GESTURE_DRAIN_TIMEOUT_MS = 500L

        // Polling interval while waiting for an in-flight gesture (change 2).
        private const val GESTURE_DRAIN_POLL_MS = 16L
    }

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------

    private enum class State { IDLE, LOCKING_CUE_BALL, LOCKING_TARGET_BALL, AIMING, LOCKED }

    @Volatile private var state = State.IDLE
    private val mainHandler = Handler(Looper.getMainLooper())

    // -----------------------------------------------------------------------
    // Shot data (Double throughout)
    // -----------------------------------------------------------------------

    private var virtualX: Double = 0.0
    private var virtualY: Double = 0.0

    private var cueBallX:    Double = 0.0
    private var cueBallY:    Double = 0.0
    private var targetBallX: Double = 0.0
    private var targetBallY: Double = 0.0

    // Current aim angle in radians. Initialized when entering AIMING state
    // from the pixel-space direction cueBall → targetBall (Y-down convention).
    private var currentAngleRad: Double = 0.0

    // Live force-bar power fraction [0.0, 1.0] while the user's finger is on
    // the force bar. Meaningless outside an active drag.
    private var currentDragPower: Double = 0.0

    // Most recent power value at which the evaluated shot was confirmed to
    // pot. Reset to null at the start of every new force-bar drag gesture
    // (ACTION_DOWN), not just once per AIMING session.
    private var latchedPotPower: Double? = null

    // True while a updateAimLive() call's native-call sequence
    // (updateIndirectAim + the getIndirectPathPx/renderOverlay follow-ups)
    // is in flight. Prevents a queued touch-move sample from re-entering
    // updateAimLive() with a stale or partially-updated native cache
    // mid-sequence.
    private var aimUpdateInFlight = false

    // Table-drag aiming: the raw angle-to-finger direction computed on the
    // previous ACTION_MOVE/ACTION_DOWN sample, used to compute a delta for
    // sensitivity-scaled incremental rotation. Null when no drag is active.
    private var lastRawDragAngleRad: Double? = null

    // Change 2: set to true by onForceBarTouch immediately before calling
    // IndirectGestureService.instance?.fireForceDrag(...), and back to false
    // once the gesture is known to have completed (or the bounded wait in
    // stop() expires). Checked in stop() to decide whether to wait before
    // tearing down native/overlay state.
    @Volatile private var gestureInFlight = false

    // -----------------------------------------------------------------------
    // D-pad step size — selected via the four step-size buttons (no slider).
    // NOTE: While state == AIMING this numeric value is no longer degrees-
    // per-step (the dpad cross is gone). Instead it is the SENSITIVITY
    // SCALE FACTOR applied to the raw angle-to-finger delta during table-
    // drag aiming (see onTableDragTouch): each move sample's raw delta
    // (current raw angle-to-finger minus the previous sample's) is
    // multiplied by this value before being applied to currentAngleRad.
    // 1.0 (default) tracks the finger 1:1; .1/.01 give progressively
    // finer/damped control; 10 gives exaggerated rotation per
    // unit of finger movement. In LOCKING_CUE_BALL/LOCKING_TARGET_BALL this
    // value keeps its original meaning: pixels-per-step for the dpad nudge.
    // -----------------------------------------------------------------------

    private var dpadStepPx: Double = STEP_SMALL   // default .1

    // -----------------------------------------------------------------------
    // Table geometry (Double)
    // -----------------------------------------------------------------------

    private var tableLeft:     Double = 0.0
    private var tableTop:      Double = 0.0
    private var tableRight:    Double = 0.0
    private var tableBottom:   Double = 0.0
    private var pocketNsShift: Double = 0.0

    // screenWidth/screenHeight are computed the same way
    // CalibrationManager computes its own screenWidth/screenHeight — via
    // getRealMetrics (true full physical display, not the windowed/inset
    // metrics getMetrics() would report) with landscape max/min
    // normalization — rather than trusting the passed-in
    // nativeScreenWidth/nativeScreenHeight constructor params, whatever
    // their actual source. indirectPathViewParams sizes the full-screen
    // path overlay off of these, so if they under-report the true display
    // bounds the path canvas ends up smaller than the real screen and its
    // edges look inset relative to CalibrationManager's rectangles (which
    // are always drawn against the true getRealMetrics bounds).
    private val screenWidth: Int
    private val screenHeight: Int
    init {
        val metrics = android.util.DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        screenWidth  = maxOf(metrics.widthPixels, metrics.heightPixels)
        screenHeight = minOf(metrics.widthPixels, metrics.heightPixels)
    }

    // Force-bar rect, scaled from the fixed reference resolution it was
    // captured at (FORCE_BAR_REF_WIDTH x FORCE_BAR_REF_HEIGHT) up to the
    // device's native screen resolution.
    private val scaledForceBarX: Int =
        (FORCE_BAR_X * (screenWidth.toDouble() / FORCE_BAR_REF_WIDTH)).roundToInt()
    private val scaledForceBarTopY: Int =
        (FORCE_BAR_TOP_Y * (screenHeight.toDouble() / FORCE_BAR_REF_HEIGHT)).roundToInt()
    private val scaledForceBarBottomY: Int =
        (FORCE_BAR_BOTTOM_Y * (screenHeight.toDouble() / FORCE_BAR_REF_HEIGHT)).roundToInt()
    private val scaledForceBarWidth: Int =
        (FORCE_BAR_WIDTH * (screenWidth.toDouble() / FORCE_BAR_REF_WIDTH)).roundToInt()

    init { loadPoolTableGeometry() }

    // -----------------------------------------------------------------------
    // Vibration
    // -----------------------------------------------------------------------

    private val vibrator: Vibrator? by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
                ?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    private fun vibrate(durationMs: Long = 40L) {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            v.vibrate(VibrationEffect.createOneShot(durationMs, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION") v.vibrate(durationMs)
        }
    }

    // -----------------------------------------------------------------------
    // WindowManager params factory
    // -----------------------------------------------------------------------

    private fun baseParams(
        w: Int,
        h: Int,
        gravity: Int       = Gravity.TOP or Gravity.START,
        x: Int             = 0,
        y: Int             = 0,
        touchable: Boolean = true,
        noLimits:  Boolean = false
    ): WindowManager.LayoutParams {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else
            @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE

        var flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
        if (!touchable) flags = flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        if (noLimits)   flags = flags or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS

        return WindowManager.LayoutParams(w, h, type, flags, PixelFormat.TRANSLUCENT).also {
            it.gravity           = gravity
            it.x                 = x
            it.y                 = y
            it.screenOrientation = ActivityInfo.SCREEN_ORIENTATION_USER_LANDSCAPE
        }
    }

    // -----------------------------------------------------------------------
    // dp → px
    // -----------------------------------------------------------------------

    private fun dp(v: Int): Int =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, v.toFloat(),
            context.resources.displayMetrics
        ).roundToInt()

    // -----------------------------------------------------------------------
    // Overlay registry
    // -----------------------------------------------------------------------

    private val addedViews = mutableListOf<View>()

    private fun addOverlayView(view: View, params: WindowManager.LayoutParams) {
        try {
            windowManager.addView(view, params)
            addedViews.add(view)
            logD("addOverlayView: ${view.javaClass.simpleName} (total=${addedViews.size})")
        } catch (e: Exception) { logE("addOverlayView failed", e) }
    }

    private fun removeOverlayView(view: View) {
        if (!addedViews.remove(view)) return
        try { windowManager.removeView(view) }
        catch (e: Exception) { logE("removeOverlayView failed", e) }
    }

    private fun removeOverlayViewIfAttached(view: View) {
        if (view.isAttachedToWindow) removeOverlayView(view)
        else addedViews.remove(view)
    }

    /**
     * Toggles [WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE] (and keeps
     * [WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE] set, as all overlay
     * windows already have it via [baseParams]) on an already-attached
     * window in place, via [WindowManager.updateViewLayout] — no
     * remove/re-add, so the view's state and any pending touch sequence on
     * OTHER windows is undisturbed. Used to make persistent buttons (like
     * startBtnContainer/resetBtnContainer) momentarily pass touches straight
     * through to whatever's beneath them — in our case, the game — instead
     * of intercepting the synthetic drag gesture during dispatch.
     *
     * Safe to call on a view that isn't currently attached; it's a no-op in
     * that case since there's no live window to update.
     */
    private fun setWindowTouchable(view: View, params: WindowManager.LayoutParams, touchable: Boolean) {
        if (!view.isAttachedToWindow) return
        params.flags = if (touchable) {
            params.flags and WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE.inv()
        } else {
            params.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        }
        try {
            windowManager.updateViewLayout(view, params)
        } catch (e: Exception) {
            logE("setWindowTouchable failed", e)
        }
    }

    // -----------------------------------------------------------------------
    // Started flag
    // -----------------------------------------------------------------------

    private var started = false

    // -----------------------------------------------------------------------
    // Persistent views
    // -----------------------------------------------------------------------

    // Add / Exit cluster (two buttons only — btnReset removed)
    private lateinit var startBtnContainer: FrameLayout
    private lateinit var startBtnParams:    WindowManager.LayoutParams
    private lateinit var resetBtnContainer: FrameLayout
    private lateinit var resetBtnParams:    WindowManager.LayoutParams
    private lateinit var btnAdd:  TextView   // dual-purpose: ▶ (start) or ↺ (restart)
    private lateinit var btnExit: TextView

    // Ring markers (FLAG_NOT_TOUCHABLE — visual only)
    private lateinit var cueBallMarker:        DashedRingView
    private lateinit var cueBallMarkerParams:  WindowManager.LayoutParams
    private var cueBallMarkerAdded = false

    private lateinit var targetBallMarker:       DashedRingView
    private lateinit var targetBallMarkerParams: WindowManager.LayoutParams
    private var targetBallMarkerAdded = false

    // Indirect path overlay (full-screen, visual only — draws live aim path)
    private var indirectPathView:       IndirectPathView? = null
    private var indirectPathViewParams: WindowManager.LayoutParams? = null
    private var indirectPathViewAdded = false

    // Force-bar overlay (vertical drag track — live power during AIMING)
    private var forceBarView:       ForceBarView? = null
    private var forceBarViewParams: WindowManager.LayoutParams? = null
    private var forceBarViewAdded = false

    // Magnifier panel
    private var magnifierView:        MagnifierView? = null
    private var magnifierPanelView:   LinearLayout?  = null
    private var magnifierPanelParams: WindowManager.LayoutParams? = null
    private var magnifierPanelW = 0
    private var magnifierPanelH = 0

    // The most recently requested zoom value, set via setMagnifierZoom().
    // Survives across (re)builds of magnifierView: setMagnifierZoom() can be
    // (and currently is, from OverlayService.launchIndirectMode()) called
    // BEFORE start()/buildAllPersistentViews() has ever run, at which point
    // magnifierView is still null and a plain "magnifierView?.zoomFactor = x"
    // assignment would silently no-op. buildMagnifierPanel() reads this field
    // instead of hardcoding MAGNIFIER_ZOOM, so whatever zoom the floating
    // panel last set is always applied to the freshly built view.
    private var pendingMagnifierZoom: Float = MAGNIFIER_ZOOM

    // AIMING-only magnifier auto-focus: while the force-bar gesture is being
    // dragged (ACTION_DOWN..ACTION_MOVE, before release), the magnifier
    // shows a zoomed view auto-focused on the first-ever bounce ghost circle
    // — i.e. IndirectPathData.segments[1], the first point after the cue
    // ball on the stitched path (see IndirectPathOverlay.kt: segments[0] is
    // the cue ball, segments[1] is the first cushion/ghost contact). Null
    // when not actively focusing (locking states keep using virtualX/Y as
    // before; onCleanFrame falls back to that when this is null).
    // Stored in CAPTURE-scale pixel space (matching onCleanFrame's crop
    // math), i.e. screen-space segment * captureScale.
    @Volatile private var aimingMagnifierFocus: Pair<Int, Int>? = null

    // D-pad cluster (4 directional buttons + lock)
    private var dpadContainer: FrameLayout? = null
    private var dpadParams:    WindowManager.LayoutParams? = null
    private lateinit var btnDpadUp:    TextView
    private lateinit var btnDpadDown:  TextView
    private lateinit var btnDpadLeft:  TextView
    private lateinit var btnDpadRight: TextView
    private lateinit var btnDpadLock:  TextView

    // Absolute screen x-center of the dpad's Left-arrow button, computed in
    // buildDpadCluster() and consumed by buildSensitivityStrip() to center
    // the AIMING sensitivity strip on that same x-position.
    private var dpadLeftArrowCenterX: Int = 0

    // Step-size buttons — two above Up (tiny, small), two below Down (normal, large).
    // Used only in LOCKING_CUE_BALL/LOCKING_TARGET_BALL for pixels-per-step.
    private lateinit var btnStepTiny:   TextView   // 0.01 — directly above Up
    private lateinit var btnStepSmall:  TextView   // 0.1  — beside btnStepTiny
    private lateinit var btnStepNormal: TextView   // 1    — directly below Down (default)
    private lateinit var btnStepLarge:  TextView   // 10   — beside btnStepNormal

    // AIMING-only vertical sensitivity strip — same four values (10, 1, .1,
    // .01) as the step-size buttons above, but laid out as a single vertical
    // column centered on the dpad's former Left-arrow x-position, and
    // controlling table-drag rotation sensitivity instead of pixel step.
    // Separate TextView instances from btnStep* above since a View cannot
    // be attached under two parents at once.
    private var sensStripContainer: LinearLayout? = null
    private var sensStripParams:    WindowManager.LayoutParams? = null
    private lateinit var btnSensPassthrough: TextView   // T — enables background-app touch passthrough over the pool region; sits above the 5 button
    private lateinit var btnSensLarge:  TextView   // 5 (AIMING-only value) — top of sens group
    private lateinit var btnSensNormal: TextView   // 1    — (default)
    private lateinit var btnSensSmall:  TextView   // 0.1
    private lateinit var btnSensTiny:   TextView   // 0.01 — bottom

    // When true, the AIMING full-table drag surface (tableDragView) is set
    // FLAG_NOT_TOUCHABLE so touches fall through to whatever app/window sits
    // beneath the overlay instead of being captured for the aim-drag
    // gesture. Toggled by btnSensPassthrough ('T'). Reset to false whenever
    // the AIMING overlay is torn down so re-entering AIMING always starts
    // with the pool region capturing touches as normal.
    private var tableDragPassthroughEnabled: Boolean = false

    // Table tap window
    private var tableTapView:   View?                       = null
    private var tableTapParams: WindowManager.LayoutParams? = null

    // AIMING-only full-table touch surface — drag-to-aim only (tapping has
    // no effect on the angle). Separate from tableTapView (which remains
    // dedicated to the locking-state marker-placement tap gesture).
    private var tableDragView:   View?                       = null
    private var tableDragParams: WindowManager.LayoutParams? = null

    // -----------------------------------------------------------------------
    // D-pad hold-repeat loop
    // -----------------------------------------------------------------------

    private val tickRunning = AtomicBoolean(false)
    private var heldDx = 0.0
    private var heldDy = 0.0

    // -----------------------------------------------------------------------
    // Capture gate
    // -----------------------------------------------------------------------

    private val captureThreadRunning = AtomicBoolean(false)

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    fun start() {
        if (started) { logW("start() already started — ignored"); return }
        logD("start() — begin")
        try {
            logD("start(): calling buildAllPersistentViews()")
            buildAllPersistentViews()
            logD("start(): buildAllPersistentViews() returned OK")
        } catch (t: Throwable) {
            logE("start(): buildAllPersistentViews() threw", t)
            throw t
        }
        try {
            logD("start(): calling transitionTo(IDLE)")
            transitionTo(State.IDLE)
            logD("start(): transitionTo(IDLE) returned OK")
        } catch (t: Throwable) {
            logE("start(): transitionTo(IDLE) threw", t)
            throw t
        }
        started = true
        logD("start() — complete, started=true")
    }

    /**
     * Tears down all overlays and native state.
     *
     * Change 1: calls QeightJNI.destroyIndirectSolver() alongside the
     * existing clearIndirectShot()/renderOverlay() calls, closing the
     * native solver leak that occurred when switching back to pipeline mode.
     *
     * Change 2: if a gesture was just fired via
     * IndirectGestureService.instance?.fireForceDrag(...) and hasn't had
     * time to complete, waits up to GESTURE_DRAIN_TIMEOUT_MS (500 ms) in
     * short GESTURE_DRAIN_POLL_MS (16 ms) intervals before proceeding.
     * The check uses IndirectGestureService.instance?.isGestureInFlight()
     * if the service exposes that predicate; otherwise it falls back to the
     * local gestureInFlight flag set immediately before fireForceDrag() is
     * called and cleared on completion or timeout. The wait is performed on
     * a background thread so as not to block the main thread; all subsequent
     * teardown is then posted back to the main thread.
     *
     * All other teardown behaviour is identical to the original.
     */
    fun stop() {
        logD("stop() — called. Caller stack:\n" +
                Thread.currentThread().stackTrace
                    .drop(1) // drop getStackTrace() itself
                    .take(12)
                    .joinToString("\n") { "    at $it" })

        // Change 2: if a gesture dispatch is in flight, drain it on a
        // background thread (bounded wait) before tearing down native state.
        // This prevents the accessibility gesture from being left dangling
        // across the mode switch back to pipeline.
        val needsDrain = gestureInFlight ||
                (IndirectGestureService.instance?.isGestureInFlight() == true)

        if (needsDrain) {
            logD("stop: gesture in flight — waiting up to ${GESTURE_DRAIN_TIMEOUT_MS}ms")
            Thread {
                val deadline = System.currentTimeMillis() + GESTURE_DRAIN_TIMEOUT_MS
                while (System.currentTimeMillis() < deadline) {
                    // Re-check both the local flag and the service's own
                    // in-flight predicate on every poll tick.
                    val serviceInFlight =
                        IndirectGestureService.instance?.isGestureInFlight() == true
                    if (!gestureInFlight && !serviceInFlight) {
                        logD("stop: gesture drained before timeout")
                        break
                    }
                    try { Thread.sleep(GESTURE_DRAIN_POLL_MS) }
                    catch (e: InterruptedException) {
                        Thread.currentThread().interrupt()
                        break
                    }
                }
                // Always clear the flag after the drain attempt, whether
                // the gesture completed normally or we hit the timeout.
                gestureInFlight = false
                logD("stop: proceeding with teardown (gestureInFlight cleared)")
                // Post actual teardown back to main thread — all WM
                // operations require it.
                mainHandler.post { performTeardown() }
            }.also { it.isDaemon = true; it.name = "IndirectGestureDrain" }.start()
        } else {
            // No in-flight gesture — tear down immediately on the calling
            // thread (which must already be the main thread for WM calls).
            performTeardown()
        }
    }

    /**
     * Performs the actual teardown of all views and native state.
     * Extracted from stop() so it can be called either directly (no gesture
     * in flight) or via mainHandler.post (after gesture drain completes).
     * All logic is identical to the original stop() body.
     *
     * Change 1: QeightJNI.destroyIndirectSolver() is called here, alongside
     * clearIndirectShot() and renderOverlay(), to release the native solver
     * allocation that was previously leaked on every mode exit.
     */
    private fun performTeardown() {
        stopDpadHold()
        captureThreadRunning.set(false)
        aimingMagnifierFocus = null
        removeDpadCluster()
        removeSensitivityStrip()
        removeTableDragView()
        removeMagnifierPanel()
        removeTableTapWindow()
        addedViews.toList().forEach { v ->
            try { if (v.isAttachedToWindow) windowManager.removeView(v) }
            catch (e: Exception) { logE("stop: removeView failed", e) }
        }
        addedViews.clear()
        // indirectPathView / forceBarView are covered by the addedViews
        // cleanup pass above (added via addOverlayView()); reset flags so a
        // subsequent start() re-attaches fresh on the next aim.
        // sensStripContainer / tableDragView are likewise covered by the
        // same pass and use plain isAttachedToWindow checks (no separate
        // *Added boolean), so nothing further to reset for them here.
        indirectPathViewAdded = false
        forceBarViewAdded = false

        // Hide the path view's own held data too (same as onResetClicked()),
        // so a stale path can't flash on screen if start() re-attaches this
        // same view instance before the next updateAimLive() call.
        indirectPathView?.setPathData(IndirectPathData(visible = false))

        // Clear native solver state and force a re-render, exactly like
        // onResetClicked() does — without this, exiting the mode removes
        // this app's own overlay views but leaves the last-rendered path
        // still drawn on screen via QeightJNI's own rendering surface,
        // since that surface is independent of the WindowManager views
        // above and is never cleared just by detaching them.
        try { QeightJNI.clearIndirectShot() }    catch (e: Exception) { logE("stop: clearIndirectShot", e) }
        try { QeightJNI.renderOverlay() }         catch (e: Exception) { logE("stop: renderOverlay", e) }

        // Change 1: release the native indirect solver allocation. This was
        // previously leaked on every call to stop() (i.e. every mode switch
        // back to pipeline), because clearIndirectShot() resets solver state
        // but does not free the solver object itself — only
        // destroyIndirectSolver() does. Called after clearIndirectShot() /
        // renderOverlay() so the solver is still valid for those calls.
        try { QeightJNI.destroyIndirectSolver() } catch (e: Exception) { logE("stop: destroyIndirectSolver", e) }

        started = false
    }

    /**
     * Sets the magnifier's zoom level. Safe to call at any time, including
     * before start()/buildAllPersistentViews() has run (e.g. OverlayService
     * calls this immediately before start() on every launchIndirectMode()).
     * The value is always remembered in pendingMagnifierZoom and re-applied
     * by buildMagnifierPanel() when the view is (re)built, so a zoom set
     * before the view exists is never silently dropped.
     */
    fun setMagnifierZoom(zoom: Float) {
        pendingMagnifierZoom = zoom
        magnifierView?.zoomFactor = zoom
    }

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------

    private fun loadPoolTableGeometry() {
        tableLeft     = poolTableLeft.toDouble()
        tableTop      = poolTableTop.toDouble()
        tableRight    = poolTableRight.toDouble()
        tableBottom   = poolTableBottom.toDouble()
        pocketNsShift = poolPocketNsShift.toDouble()
    }

    // -----------------------------------------------------------------------
    // View construction helpers
    // -----------------------------------------------------------------------

    private fun makeThemedCircleButton(label: String, diamPx: Int, textSizeSp: Float = 13f) =
        TextView(context).apply {
            text = label
            setTextColor(Color.parseColor("#FF6D00"))
            typeface = android.graphics.Typeface.create(
                android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, textSizeSp)
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(diamPx, diamPx)
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL; setColor(Color.BLACK)
            }
        }

    private fun makeThemedSquareButton(label: String, sidePx: Int, textSizeSp: Float = 13f) =
        TextView(context).apply {
            text = label
            setTextColor(Color.parseColor("#FF6D00"))
            typeface = android.graphics.Typeface.create(
                android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, textSizeSp)
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(sidePx, sidePx)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE; setColor(Color.BLACK)
            }
        }

    private fun themedActiveSquareBackground()   = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE; setColor(Color.parseColor("#FF6D00"))
    }
    private fun themedInactiveSquareBackground() = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE; setColor(Color.BLACK)
    }

    // -----------------------------------------------------------------------
    // Build all views
    // -----------------------------------------------------------------------

    private fun buildAllPersistentViews() {
        val steps: List<Pair<String, () -> Unit>> = listOf(
            "buildAddResetCluster"   to { buildAddResetCluster() },
            "buildMarkers"           to { buildMarkers() },
            "buildIndirectPathView"  to { buildIndirectPathView() },
            "buildForceBarView"      to { buildForceBarView() },
            "buildMagnifierPanel"    to { buildMagnifierPanel() },
            "buildDpadCluster"       to { buildDpadCluster() },
            "buildSensitivityStrip"  to { buildSensitivityStrip() },
            "buildTableTapWindow"    to { buildTableTapWindow() },
            "buildTableDragView"     to { buildTableDragView() }
        )
        for ((name, step) in steps) {
            try {
                logD("buildAllPersistentViews(): $name — start")
                step()
                logD("buildAllPersistentViews(): $name — OK")
            } catch (t: Throwable) {
                logE("buildAllPersistentViews(): $name — threw", t)
                throw t
            }
        }
    }

    // ── Add / Exit cluster (two independently-positioned buttons) ─────────
    //
    // btnAdd is dual-purpose (unchanged):
    //   • IDLE state   → shows "▶", click starts the flow (LOCKING_CUE_BALL)
    //   • Any other    → shows "↺", click runs the full reset/restart logic
    // Pinned near the TOP screen edge (right side).
    //
    // btnExit is the sole exit path, pinned near the BOTTOM-LEFT screen corner.
    //
    // Both are sized to match the dpad buttons (crossSize = DPAD_SIZE / 3)
    // rather than floatingButtonSizePx.

    private fun buildAddResetCluster() {
        val btnSize = DPAD_SIZE / 3
        val edgeMarginPx = dp(8)   // small gap so the button doesn't get clipped by the screen edge

        btnAdd  = makeThemedCircleButton("▶", btnSize)
        btnExit = makeThemedCircleButton("✕", btnSize)

        // btnAdd: own window, touching the upper boundary of the screen.
        startBtnContainer = FrameLayout(context).apply {
            setBackgroundColor(Color.TRANSPARENT)
            addView(btnAdd, FrameLayout.LayoutParams(btnSize, btnSize))
        }
        startBtnParams = baseParams(
            btnSize, btnSize,
            Gravity.TOP or Gravity.START,
            screenWidth - floatingButtonRightMarginPx - btnSize,
            edgeMarginPx
        )
        addOverlayView(startBtnContainer, startBtnParams)

        // btnExit: own window, pinned to the bottom-left corner of the screen.
        resetBtnContainer = FrameLayout(context).apply {
            setBackgroundColor(Color.TRANSPARENT)
            addView(btnExit, FrameLayout.LayoutParams(btnSize, btnSize))
        }
        resetBtnParams = baseParams(
            btnSize, btnSize,
            Gravity.TOP or Gravity.START,
            edgeMarginPx,
            screenHeight - btnSize - edgeMarginPx
        )
        addOverlayView(resetBtnContainer, resetBtnParams)

        btnAdd.setOnClickListener  {
            logD("btnAdd.onClick fired — state=$state")
            onAddClicked()
        }
        btnExit.setOnClickListener {
            logD("btnExit.onClick fired — state=$state — calling stop()+onExit()")
            stop()
            onExit()
        }
    }

    // ── Ring markers ─────────────────────────────────────────────────────

    private fun buildMarkers() {
        val size = dp(32)
        cueBallMarker = DashedRingView(context, Color.WHITE)
        cueBallMarkerParams = baseParams(size, size,
            Gravity.TOP or Gravity.START, 0, 0, touchable = false)

        targetBallMarker = DashedRingView(context, 0xFFFFD600.toInt())
        targetBallMarkerParams = baseParams(size, size,
            Gravity.TOP or Gravity.START, 0, 0, touchable = false)
    }

    // ── Indirect path overlay ─────────────────────────────────────────────
    // Full-screen, non-touchable overlay. Built eagerly but only added to
    // WindowManager once aiming begins (see updateAimLive()).

    private fun buildIndirectPathView() {
        indirectPathView = IndirectPathView(context)
        indirectPathViewParams = baseParams(
            screenWidth, screenHeight,
            Gravity.TOP or Gravity.START, 0, 0,
            touchable = false,
            noLimits  = true)
    }

    // ── Force-bar overlay ───────────────────────────────────────────────────
    // Vertical, touchable drag track. Built eagerly here (same pattern as
    // buildIndirectPathView()) but only attached/detached with AIMING state
    // (see showAimingOverlay/hideAimingOverlay).

    private fun buildForceBarView() {
        val barView = ForceBarView(context)
        forceBarView = barView
        forceBarViewParams = baseParams(
            scaledForceBarWidth, scaledForceBarBottomY - scaledForceBarTopY,
            Gravity.TOP or Gravity.START,
            scaledForceBarX - scaledForceBarWidth / 2, scaledForceBarTopY,
            touchable = true)

        barView.setOnTouchListener { _, ev -> onForceBarTouch(ev) }
    }

    // -----------------------------------------------------------------------
    // buildPanel — factory for the magnifier panel
    // -----------------------------------------------------------------------

    private data class PanelRefs(
        val panel:  LinearLayout,
        val params: WindowManager.LayoutParams
    )

    private fun buildPanel(
        x: Int,
        y: Int,
        w: Int,
        h: Int,
        contentBuilder: (LinearLayout) -> Unit
    ): PanelRefs {
        val panel = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.TRANSPARENT)
            setPadding(0, 0, 0, 0)
        }
        contentBuilder(panel)
        val params = baseParams(
            w, h,
            Gravity.TOP or Gravity.START,
            x, y,
            touchable = true,
            noLimits  = true
        )
        return PanelRefs(panel, params)
    }

    // ── Magnifier panel ───────────────────────────────────────────────────

    private fun buildMagnifierPanel() {
        val x = -40
        val y = 10
        magnifierPanelW = 500
        magnifierPanelH = 500

        val refs = buildPanel(
            x = x, y = y, w = magnifierPanelW, h = magnifierPanelH,
            contentBuilder = { panel ->
                val magView = MagnifierView(context).apply {
                    // Apply whatever zoom was last requested via
                    // setMagnifierZoom() (e.g. from the floating panel's
                    // spinner) rather than always resetting to the
                    // MAGNIFIER_ZOOM default — otherwise every (re)build of
                    // this view (each fresh start() call) would silently
                    // discard the user's chosen zoom.
                    zoomFactor   = pendingMagnifierZoom
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f).also {
                        it.setMargins(0, 0, 0, 0)
                    }
                }
                magnifierView = magView
                panel.addView(magView)
            }
        )
        magnifierPanelView   = refs.panel
        magnifierPanelParams = refs.params
    }

    // ── D-pad cluster ─────────────────────────────────────────────────────
    // Used only in LOCKING_CUE_BALL / LOCKING_TARGET_BALL. Fully hidden
    // (dpadContainer detached) during AIMING — replaced there by the
    // vertical sensitivity strip + table-drag aiming (see
    // buildSensitivityStrip / buildTableDragView).
    //
    //        ┌──────┐  ┌──────┐
    //        │ .01  │  │ .1   │   ← step-size row (above Up) — px/step
    //        └──────┘  └──────┘
    //            ┌────┐
    //            │ UP │
    //        ┌────┬────┬────┐
    //        │ LT │LOCK│ RT │
    //        └────┴────┴────┘
    //            │ DN │
    //            └────┘
    //        ┌──────┐  ┌──────┐
    //        │  1   │  │ 10   │   ← step-size row (below Down) — 1 is default
    //        └──────┘  └──────┘

    private fun buildDpadCluster() {
        val crossSize   = DPAD_SIZE / 3
        val stepBtnSize = crossSize
        val stepGapPx   = dp(STEP_BTN_GAP_PX)
        val rowMarginPx = dp(16)

        // Distance from the cross's center (the Lock button) to the center
        // of each directional button. Requested: 50px center-to-center,
        // independent of button size — buttons stay crossSize wide/tall,
        // they just sit farther apart than the old adjacent-cell layout.
        val armOffsetPx = dp(DPAD_ARM_OFFSET_PX)

        // Overall footprint now derives from armOffsetPx rather than the old
        // 3-cell grid, since Up/Down/Left/Right can sit farther from center
        // than a single cell width would allow.
        val crossSpan    = armOffsetPx * 2 + crossSize
        val stepRowWidth = stepBtnSize * 2 + stepGapPx
        val containerW   = maxOf(crossSpan, stepRowWidth)
        val containerH   = crossSpan + 2 * (stepBtnSize + rowMarginPx)
        val crossLeft    = (containerW - crossSpan) / 2
        val crossTop     = stepBtnSize + rowMarginPx
        val stepRowLeft  = (containerW - stepRowWidth) / 2

        // Center of the cross (where the Lock button sits), in container coords.
        val crossCenterX = crossLeft + crossSpan / 2
        val crossCenterY = crossTop  + crossSpan / 2

        val container = FrameLayout(context).apply {
            setBackgroundColor(Color.TRANSPARENT)
        }

        // Places a cross button centered at (crossCenterX + dCol*armOffsetPx,
        // crossCenterY + dRow*armOffsetPx) — i.e. dCol/dRow of 0 = center
        // (Lock), ±1 = one arm's length out along that axis.
        fun placeInCross(view: View, dCol: Int, dRow: Int) {
            view.layoutParams = FrameLayout.LayoutParams(crossSize, crossSize).also {
                it.leftMargin = crossCenterX + dCol * armOffsetPx - crossSize / 2
                it.topMargin  = crossCenterY + dRow * armOffsetPx - crossSize / 2
            }
            container.addView(view)
        }

        fun placeStepBtn(view: View, slot: Int, aboveCross: Boolean) {
            view.layoutParams = FrameLayout.LayoutParams(stepBtnSize, stepBtnSize).also {
                it.leftMargin = stepRowLeft + slot * (stepBtnSize + stepGapPx)
                it.topMargin  = if (aboveCross) 0 else crossTop + crossSpan + rowMarginPx
            }
            container.addView(view)
        }

        // ── Directional cross + lock ───────────────────────────────────────
        btnDpadUp    = makeThemedSquareButton("▲", crossSize, 18f)
        btnDpadDown  = makeThemedSquareButton("▼", crossSize, 18f)
        btnDpadLeft  = makeThemedSquareButton("◀", crossSize, 18f)
        btnDpadRight = makeThemedSquareButton("▶", crossSize, 18f)
        btnDpadLock  = makeThemedSquareButton("🔒", crossSize, 18f)

        placeInCross(btnDpadUp,    0,  -1)
        placeInCross(btnDpadLeft,  -1, 0)
        placeInCross(btnDpadLock,  0,  0)
        placeInCross(btnDpadRight, 1,  0)
        placeInCross(btnDpadDown,  0,  1)

        wireDpadButton(btnDpadUp,    dx = 0.0,  dy = -1.0)
        wireDpadButton(btnDpadDown,  dx = 0.0,  dy = 1.0)
        wireDpadButton(btnDpadLeft,  dx = -1.0, dy = 0.0)
        wireDpadButton(btnDpadRight, dx = 1.0,  dy = 0.0)

        btnDpadLock.setOnClickListener { onLockClicked() }

        // ── Step-size buttons ───────────────────────────────────────────────
        btnStepTiny   = makeThemedSquareButton(".01", stepBtnSize, 11f)
        btnStepSmall  = makeThemedSquareButton(".1",  stepBtnSize, 12f)
        btnStepNormal = makeThemedSquareButton("1",   stepBtnSize, 13f)
        btnStepLarge  = makeThemedSquareButton("10",  stepBtnSize, 13f)

        placeStepBtn(btnStepTiny,   slot = 0, aboveCross = true)
        placeStepBtn(btnStepSmall,  slot = 1, aboveCross = true)
        placeStepBtn(btnStepNormal, slot = 0, aboveCross = false)
        placeStepBtn(btnStepLarge,  slot = 1, aboveCross = false)

        wireStepButton(btnStepTiny,   STEP_TINY)
        wireStepButton(btnStepSmall,  STEP_SMALL)
        wireStepButton(btnStepNormal, STEP_NORMAL)
        wireStepButton(btnStepLarge,  STEP_LARGE)

        applyStepSizeUi()   // highlight the default (.1) selection

        dpadContainer = container
        dpadParams = baseParams(
            containerW, containerH,
            Gravity.TOP or Gravity.START,
            DPAD_X - crossLeft, DPAD_Y - crossTop,
            touchable = true,
            noLimits  = true
        )

        // Absolute screen x-center of the dpad's Left-arrow button — the
        // vertical sensitivity strip built in buildSensitivityStrip() is
        // centered on this exact x so it visually "replaces" the Left arrow
        // during AIMING. Left's center is now armOffsetPx left of the cross
        // center, which is crossSpan/2 right of DPAD_X (the cross's left edge).
        dpadLeftArrowCenterX = DPAD_X + crossSpan / 2 - armOffsetPx
    }

    // ── AIMING sensitivity strip ────────────────────────────────────────────
    // Vertical column of the same four values (10, 1, .1, .01) used by the
    // locking-state step buttons, but reused here as the table-drag
    // rotation-sensitivity scale (see onTableDragTouch). Centered on
    // dpadLeftArrowCenterX so it sits exactly where the dpad's Left arrow
    // button appears in the locking states.

    private fun buildSensitivityStrip() {
        val btnSize    = DPAD_SIZE / 3
        val gapPx      = dp(8)
        // Must match buildDpadCluster()'s crossSpan so this strip stays
        // vertically centered on the cross now that the arms sit
        // DPAD_ARM_OFFSET_PX apart instead of directly adjacent.
        val crossSpan  = dp(DPAD_ARM_OFFSET_PX) * 2 + btnSize

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.TRANSPARENT)
        }

        btnSensPassthrough = makeThemedSquareButton("T", btnSize, 13f)
        btnSensLarge  = makeThemedSquareButton("5",   btnSize, 13f)
        btnSensNormal = makeThemedSquareButton("1",   btnSize, 13f)
        btnSensSmall  = makeThemedSquareButton(".1",  btnSize, 12f)
        btnSensTiny   = makeThemedSquareButton(".01", btnSize, 11f)

        fun LinearLayout.addSensBtn(v: View, last: Boolean) {
            addView(v, LinearLayout.LayoutParams(btnSize, btnSize).also {
                if (!last) it.bottomMargin = gapPx
            })
        }
        // 5-button layout: passthrough toggle on top, then the four
        // sensitivity buttons.
        container.addSensBtn(btnSensPassthrough, last = false)
        container.addSensBtn(btnSensLarge,  last = false)
        container.addSensBtn(btnSensNormal, last = false)
        container.addSensBtn(btnSensSmall,  last = false)
        container.addSensBtn(btnSensTiny,   last = true)

        wireSensButton(btnSensLarge,  SENS_LARGE_AIMING_VALUE)
        wireSensButton(btnSensNormal, STEP_NORMAL)
        wireSensButton(btnSensSmall,  STEP_SMALL)
        wireSensButton(btnSensTiny,   STEP_TINY)
        wirePassthroughButton()

        applySensStripUi()   // highlight the default (.1) selection

        // Strip now holds 5 buttons with 4 gaps between them.
        val stripH = btnSize * 5 + gapPx * 4
        sensStripContainer = container
        sensStripParams = baseParams(
            btnSize, stripH,
            Gravity.TOP or Gravity.START,
            dpadLeftArrowCenterX - btnSize / 2, DPAD_Y - (stripH - crossSpan) / 2,
            touchable = true,
            noLimits  = true
        )
    }

    // Wires a sensitivity-strip button: tapping selects that sensitivity
    // scale and highlights it. Shares dpadStepPx with the locking-state
    // step buttons (same underlying field, dual meaning depending on state).
    private fun wireSensButton(btn: TextView, size: Double) {
        btn.setOnClickListener {
            dpadStepPx = size
            applySensStripUi()
        }
    }

    // Wires the passthrough ('T') button: tapping toggles whether the
    // AIMING full-table drag surface (tableDragView) captures touches or
    // lets them fall through to the background app/window beneath it —
    // effectively flipping FLAG_NOT_TOUCHABLE on that window's layer while
    // still aiming. Does not affect dpadStepPx.
    private fun wirePassthroughButton() {
        btnSensPassthrough.setOnClickListener {
            tableDragPassthroughEnabled = !tableDragPassthroughEnabled
            val dragView   = tableDragView
            val dragParams = tableDragParams
            if (dragView != null && dragParams != null) {
                // touchable = true means the pool region KEEPS capturing
                // touches (normal aiming); we want the inverse when
                // passthrough is enabled, so touches reach the app below.
                setWindowTouchable(dragView, dragParams, touchable = !tableDragPassthroughEnabled)
            }
            applySensStripUi()
        }
    }

    private fun applySensStripUi() {
        listOf(
            btnSensTiny   to STEP_TINY,
            btnSensSmall  to STEP_SMALL,
            btnSensNormal to STEP_NORMAL,
            btnSensLarge  to SENS_LARGE_AIMING_VALUE
        ).forEach { (btn, size) ->
            btn.background = if (dpadStepPx == size) themedActiveSquareBackground()
            else                    themedInactiveSquareBackground()
        }
        btnSensPassthrough.background = if (tableDragPassthroughEnabled) themedActiveSquareBackground()
        else                                 themedInactiveSquareBackground()
    }

    // Wires a step-size button: tapping selects that step size and highlights it.
    private fun wireStepButton(btn: TextView, size: Double) {
        btn.setOnClickListener {
            dpadStepPx = size
            applyStepSizeUi()
        }
    }

    private fun applyStepSizeUi() {
        listOf(
            btnStepTiny   to STEP_TINY,
            btnStepSmall  to STEP_SMALL,
            btnStepNormal to STEP_NORMAL,
            btnStepLarge  to STEP_LARGE
        ).forEach { (btn, size) ->
            btn.background = if (dpadStepPx == size) themedActiveSquareBackground()
            else                    themedInactiveSquareBackground()
        }
    }

    // Wires a single directional button for tap (single step) and
    // press-and-hold (continuous repeated steps) behaviour.
    private fun wireDpadButton(btn: TextView, dx: Double, dy: Double) {
        btn.setOnTouchListener { _, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> { startDpadHold(dx, dy); true }
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> { stopDpadHold(); true }
                else -> false
            }
        }
    }

    // ── Table-tap window ──────────────────────────────────────────────────

    private fun buildTableTapWindow() {
        val tableW = (tableRight  - tableLeft).toInt()
        val tableH = (tableBottom - tableTop).toInt()
        val tapView = View(context)
        tableTapView = tapView

        tableTapParams = baseParams(
            tableW, tableH,
            Gravity.TOP or Gravity.START,
            tableLeft.toInt(), tableTop.toInt(),
            touchable = true
        )

        var tapDownX = 0f; var tapDownY = 0f
        tapView.setOnTouchListener { _, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> { tapDownX = ev.rawX; tapDownY = ev.rawY }
                MotionEvent.ACTION_UP   -> {
                    if (hypot(ev.rawX - tapDownX, ev.rawY - tapDownY) < TAP_SLOP_PX)
                        onTableTap(ev.rawX.toDouble(), ev.rawY.toDouble())
                }
            }
            true
        }
    }

    // ── Table-drag aiming window (AIMING state only) ────────────────────────
    // Same table-region footprint as tableTapView, but a distinct view/
    // listener dedicated to AIMING: only dragging the finger across the
    // table re-aims, using a sensitivity-scaled delta of the raw
    // angle-to-finger direction (see onTableDragTouch). Simply tapping the
    // table (touch down then up with no movement) has no effect on the aim.

    private fun buildTableDragView() {
        val tableW = (tableRight  - tableLeft).toInt()
        val tableH = (tableBottom - tableTop).toInt()
        val dragView = View(context)
        tableDragView = dragView

        tableDragParams = baseParams(
            tableW, tableH,
            Gravity.TOP or Gravity.START,
            tableLeft.toInt(), tableTop.toInt(),
            touchable = true
        )

        dragView.setOnTouchListener { _, ev -> onTableDragTouch(ev) }
    }

    // -----------------------------------------------------------------------
    // Show / hide overlays
    // -----------------------------------------------------------------------

    // Shows dpadContainer, magnifier and table-tap window for both locking states.
    // Unchanged from original.
    private fun showLockingOverlays(isCueBall: Boolean) {
        logD("showLockingOverlays isCueBall=$isCueBall")

        // dpadStepPx is shared with the AIMING sensitivity strip and may
        // still hold a different value left over from a prior AIMING
        // session. Reset to the default step size (.1) so the locking-state
        // step buttons start in a known, correctly-highlighted state.
        dpadStepPx = STEP_SMALL
        applyStepSizeUi()

        if (tableTapView?.isAttachedToWindow    == false) addOverlayView(tableTapView!!,        tableTapParams!!)
        if (magnifierPanelView?.isAttachedToWindow == false) addOverlayView(magnifierPanelView!!, magnifierPanelParams!!)
        if (dpadContainer?.isAttachedToWindow      == false) addOverlayView(dpadContainer!!,      dpadParams!!)

        // Full dpad cross (Up/Down/Left/Right/Lock) is used for XY marker
        // movement in locking states.
        btnDpadUp.visibility   = View.VISIBLE
        btnDpadDown.visibility = View.VISIBLE
        btnDpadLock.visibility = View.VISIBLE

        virtualX = (tableLeft + tableRight)  / 2.0
        virtualY = (tableTop  + tableBottom) / 2.0

        if (isCueBall) {
            placeRingMarker(cueBallMarker, cueBallMarkerParams, virtualX, virtualY, cueBallMarkerAdded)
            cueBallMarkerAdded = true
        } else {
            placeRingMarker(targetBallMarker, targetBallMarkerParams, virtualX, virtualY, targetBallMarkerAdded)
            targetBallMarkerAdded = true
        }

        captureThreadRunning.set(true)
    }

    private fun hideLockingOverlays() {
        logD("hideLockingOverlays")
        stopDpadHold()
        captureThreadRunning.set(false)
        removeTableTapWindow()
        removeMagnifierPanel()
        removeDpadCluster()
    }

    // Shows the sensitivity strip + table-drag aiming surface + force-bar
    // overlay for AIMING state. The dpad cross is fully hidden (not shown
    // at all) — Up/Down/Left/Right/Lock are all gone during AIMING, replaced
    // by drag-to-aim on the table (onTableDragTouch) and the vertical
    // sensitivity strip (scales drag rotation, not a rotate button).
    // tableTapView and magnifierPanelView are NOT shown during aiming.
    private fun showAimingOverlay() {
        logD("showAimingOverlay")

        // dpadStepPx is shared with the locking-state step buttons and may
        // still hold a locking-state value left over from before the AIMING
        // transition. Reset to the default sensitivity (.1) so the strip
        // starts in a known, correctly-highlighted state.
        dpadStepPx = STEP_SMALL
        applySensStripUi()

        val stripView   = sensStripContainer
        val stripParams = sensStripParams
        if (stripView != null && stripParams != null && !stripView.isAttachedToWindow) {
            addOverlayView(stripView, stripParams)
        }

        val dragView   = tableDragView
        val dragParams = tableDragParams
        // Passthrough ('T') is session-scoped: always start a fresh AIMING
        // entry with the pool region capturing touches normally, regardless
        // of how the previous AIMING session was left.
        tableDragPassthroughEnabled = false
        dragParams?.let { it.flags = it.flags and WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE.inv() }
        if (dragView != null && dragParams != null && !dragView.isAttachedToWindow) {
            addOverlayView(dragView, dragParams)
        }

        val barView   = forceBarView
        val barParams = forceBarViewParams
        if (barView != null && barParams != null && !forceBarViewAdded) {
            addOverlayView(barView, barParams)
            forceBarViewAdded = true
        }
    }

    // Removes the sensitivity strip, table-drag surface, and force-bar
    // overlay, and stops any pending drag state, whether leaving AIMING via
    // LOCKED or a reset back to IDLE.
    private fun hideAimingOverlay() {
        logD("hideAimingOverlay")
        removeSensitivityStrip()
        removeTableDragView()
        removeForceBarView()
        currentDragPower = 0.0
        latchedPotPower = null
        lastRawDragAngleRad = null
        tableDragPassthroughEnabled = false
    }

    private fun removeForceBarView() {
        forceBarView?.let { if (it.isAttachedToWindow) removeOverlayView(it) }
        forceBarViewAdded = false
    }

    private fun removeSensitivityStrip() { sensStripContainer?.let { if (it.isAttachedToWindow) removeOverlayView(it) } }
    private fun removeTableDragView()    { tableDragView?.let      { if (it.isAttachedToWindow) removeOverlayView(it) } }

    private fun removeDpadCluster()    { dpadContainer?.let    { if (it.isAttachedToWindow) removeOverlayView(it) } }
    private fun removeMagnifierPanel() { magnifierPanelView?.let { if (it.isAttachedToWindow) removeOverlayView(it) } }
    private fun removeTableTapWindow() { tableTapView?.let        { if (it.isAttachedToWindow) removeOverlayView(it) } }

    // ── AIMING-only auto-focused magnifier (force-bar drag) ────────────────
    //
    // Shows the magnifier panel, auto-focused on the first-ever bounce ghost
    // circle (see aimingMagnifierFocus / updateAimLive), for as long as the
    // force-bar drag gesture is held. Distinct from the locking-state
    // magnifier usage (which follows virtualX/Y via the dpad cursor) — both
    // share the same magnifierView/onCleanFrame plumbing, just different
    // focus sources.

    private fun showAimingMagnifier() {
        logD("showAimingMagnifier")
        aimingMagnifierFocus = null   // cleared until the first updateAimLive() sample lands
        captureThreadRunning.set(true)
        val panel  = magnifierPanelView
        val params = magnifierPanelParams
        if (panel != null && params != null && !panel.isAttachedToWindow) {
            addOverlayView(panel, params)
        }
    }

    private fun hideAimingMagnifier() {
        logD("hideAimingMagnifier")
        captureThreadRunning.set(false)
        aimingMagnifierFocus = null
        removeMagnifierPanel()
    }

    // -----------------------------------------------------------------------
    // D-pad logic
    // -----------------------------------------------------------------------

    private fun startDpadHold(dx: Double, dy: Double) {
        heldDx = dx; heldDy = dy
        tickRunning.set(true)
        mainHandler.removeCallbacks(dpadRepeatRunnable)
        stepDpad(speedMultiplier = 1.0)   // immediate single step at normal speed — covers tap-to-move
        mainHandler.postDelayed(dpadRepeatRunnable, DPAD_REPEAT_MS)
    }

    private fun stopDpadHold() {
        tickRunning.set(false)
        mainHandler.removeCallbacks(dpadRepeatRunnable)
    }

    private val dpadRepeatRunnable = object : Runnable {
        override fun run() {
            if (!tickRunning.get()) return
            // Continuous hold moves at HOLD_SPEED_MULTIPLIER (2x) the rate of
            // a single tap, per request — only repeat ticks are sped up, not
            // the initial tap-step fired from startDpadHold().
            stepDpad(speedMultiplier = HOLD_SPEED_MULTIPLIER)
            mainHandler.postDelayed(this, DPAD_REPEAT_MS)
        }
    }

    /**
     * Branches on current state:
     *
     * - LOCKING_CUE_BALL / LOCKING_TARGET_BALL:
     *     Standard XY-nudge behaviour — moves the active ring marker by
     *     (heldDx * dpadStepPx * speedMultiplier, heldDy * dpadStepPx *
     *     speedMultiplier) pixels, clamped to table. speedMultiplier is 1.0
     *     for a single tap/click and HOLD_SPEED_MULTIPLIER (2x) for each
     *     tick while the button is held down.
     *
     * - Any other state: no-op. The dpad cross is never shown outside the
     *     locking states (AIMING uses table-drag aiming instead — see
     *     onTableDragTouch — and the dpad cross is fully hidden there).
     */
    private fun stepDpad(speedMultiplier: Double = 1.0) {
        when (state) {
            State.LOCKING_CUE_BALL, State.LOCKING_TARGET_BALL -> {
                val effectiveStep = dpadStepPx * speedMultiplier
                virtualX = clampToTableD(virtualX + heldDx * effectiveStep, horizontal = true)
                virtualY = clampToTableD(virtualY + heldDy * effectiveStep, horizontal = false)

                val isCueBallStep = (state == State.LOCKING_CUE_BALL)
                if (isCueBallStep)
                    placeRingMarker(cueBallMarker,   cueBallMarkerParams,   virtualX, virtualY, cueBallMarkerAdded)
                else
                    placeRingMarker(targetBallMarker, targetBallMarkerParams, virtualX, virtualY, targetBallMarkerAdded)
            }

            else -> { /* no-op — dpad cross is not visible outside locking states */ }
        }
    }

    // -----------------------------------------------------------------------
    // Live aim update (AIMING state)
    // -----------------------------------------------------------------------

    /**
     * Calls the native solver with the current angle and power, and updates
     * the path overlay immediately. Reuses the exact attach/update pattern
     * already present in this file for the path overlay — no new mechanism
     * invented. Vibrates briefly when the native solver reports a potential
     * pot.
     *
     * @param power        power fraction [0.0, 1.0] to evaluate at.
     * @param isLiveDrag   true when this call originates from an active
     *                     force-bar drag sample (ACTION_DOWN/ACTION_MOVE);
     *                     false for dpad-rotation or at-rest calls. Only
     *                     drag-originated calls may latch latchedPotPower.
     */
    private fun updateAimLive(power: Double = 1.0, isLiveDrag: Boolean = false) {
        if (aimUpdateInFlight) return
        aimUpdateInFlight = true
        try {
            val pots = try {
                QeightJNI.updateIndirectAim(
                    cueBallX.roundToInt(),    // boundary (b)
                    cueBallY.roundToInt(),
                    targetBallX.roundToInt(),
                    targetBallY.roundToInt(),
                    BALL_RADIUS_PX.roundToInt(),
                    // currentAngleRad is passed as-is, in this file's screen-
                    // pixel-space / Y-down convention. The native side
                    // (QeightJNI.cpp's updateIndirectAim) is responsible for
                    // converting it to the solver's Y-up/CCW-from-+X convention
                    // — it already performs the equivalent Y-axis mirroring on
                    // cueBallX/Y and targetBallX/Y via pxToCm, and now applies
                    // the same mirroring to thetaRad before calling
                    // evaluateAngle(), so the angle and the positions it's
                    // paired with stay in a consistent coordinate convention.
                    currentAngleRad.toFloat(),
                    power.toFloat()
                )
            } catch (e: Exception) { logE("updateIndirectAim threw", e); false }

            if (pots) {
                vibrate()
                if (isLiveDrag) latchedPotPower = power
            }

            val stitched = try { QeightJNI.getIndirectPathPx() } catch (e: Exception) { null }
            val points = stitched?.toPointList() ?: emptyList()
            val pathData = IndirectPathData(
                segments = points,
                visible  = true
            )
            indirectPathView?.setPathData(pathData)

            // Keep the magnifier auto-focused on the first-ever bounce ghost
            // circle (segments[1] — the first point after the cue ball; see
            // IndirectPathOverlay.kt) for the duration of either AIMING drag
            // gesture — table-drag aiming (onTableDragTouch, isLiveDrag =
            // false) as well as the force-bar power drag (isLiveDrag =
            // true). toPointList() returns screen-space points, so convert
            // back to capture-scale space to match onCleanFrame's crop math.
            if (points.size >= 2) {
                val (fx, fy) = points[1]
                aimingMagnifierFocus = Pair(
                    (fx * captureScale).roundToInt(),
                    (fy * captureScale).roundToInt()
                )
            }

            // Attach/update path overlay — reuses the same pattern as onVariationTapped().
            val view   = indirectPathView
            val params = indirectPathViewParams
            if (view != null && params != null) {
                if (!indirectPathViewAdded) {
                    addOverlayView(view, params)
                    indirectPathViewAdded = true
                } else if (view.isAttachedToWindow) {
                    try { windowManager.updateViewLayout(view, params) }
                    catch (e: Exception) { logE("updateAimLive: updateViewLayout failed", e) }
                }
            }

            try { QeightJNI.renderOverlay() } catch (e: Exception) { logE("renderOverlay", e) }
        } finally {
            aimUpdateInFlight = false
        }
    }

    // -----------------------------------------------------------------------
    // Lock button (centre of d-pad cross)
    // -----------------------------------------------------------------------

    private fun onLockClicked() {
        when (state) {
            State.LOCKING_CUE_BALL -> {
                // Commit cue ball position (capture-scale coordinates).
                cueBallX = virtualX * captureScale.toDouble()
                cueBallY = virtualY * captureScale.toDouble()
                logD("Cue locked screen=(%.2f,%.2f) cap=(%.2f,%.2f)"
                    .format(virtualX, virtualY, cueBallX, cueBallY))
                hideLockingOverlays()
                transitionTo(State.LOCKING_TARGET_BALL)
            }

            State.LOCKING_TARGET_BALL -> {
                // Commit target ball position (capture-scale coordinates).
                targetBallX = virtualX * captureScale.toDouble()
                targetBallY = virtualY * captureScale.toDouble()
                logD("Target locked screen=(%.2f,%.2f) cap=(%.2f,%.2f)"
                    .format(virtualX, virtualY, targetBallX, targetBallY))
                hideLockingOverlays()

                // Initialise aim angle from pixel-space direction cue→target
                // (Y-down convention, matching the coordinate system used
                // throughout this file and passed to the JNI layer).
                currentAngleRad = atan2(
                    targetBallY - cueBallY,
                    targetBallX - cueBallX
                )

                transitionTo(State.AIMING)

                // Fire one immediate update at full power so the default
                // 100%-power ghost path is visible before any drag occurs.
                updateAimLive(power = 1.0, isLiveDrag = false)
            }

            State.IDLE,
            State.AIMING,
            State.LOCKED -> { /* no-op guard — the lock button is hidden
                                  during AIMING; shot commit now happens
                                  exclusively via the force-bar release
                                  gesture (see onForceBarTouch). */ }
        }
    }

    // -----------------------------------------------------------------------
    // Force-bar touch handling (AIMING state — live power drag)
    // -----------------------------------------------------------------------

    /**
     * Computes the drag power fraction [0.0, 1.0] from a raw touch Y
     * position relative to the scaled force-bar track's top/bottom bounds.
     * Top of track = 0.0 = no power; bottom = 1.0 = full power.
     */
    private fun powerFromTouchY(rawY: Float): Double {
        val top    = scaledForceBarTopY.toDouble()
        val bottom = scaledForceBarBottomY.toDouble()
        if (bottom <= top) return 0.0
        return ((rawY - top) / (bottom - top)).coerceIn(0.0, 1.0)
    }

    private fun onForceBarTouch(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                latchedPotPower = null
                currentDragPower = powerFromTouchY(ev.rawY)
                forceBarView?.setPower(currentDragPower)
                showAimingMagnifier()
                updateAimLive(power = currentDragPower, isLiveDrag = true)
            }

            MotionEvent.ACTION_MOVE -> {
                currentDragPower = powerFromTouchY(ev.rawY)
                forceBarView?.setPower(currentDragPower)
                updateAimLive(power = currentDragPower, isLiveDrag = true)
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val potPower = latchedPotPower
                if (potPower != null) {
                    stopDpadHold()

                    // Always start the synthetic gesture from the top of the
                    // force-bar track (scaledForceBarTopY = minimum/zero power),
                    // stroking down to the Y position corresponding to the
                    // achieved pot power. This produces a fixed, repeatable
                    // downward stroke from min-power to the target power,
                    // regardless of where the user's finger actually touched down.
                    val endY = scaledForceBarTopY +
                            ((scaledForceBarBottomY - scaledForceBarTopY) * potPower).roundToInt()

                    // Remove EVERY touchable overlay window that could sit on
                    // top of the in-game force bar before dispatching the
                    // synthetic gesture — not after. tableDragView
                    // (buildTableDragView()) is a touchable window spanning
                    // the FULL pool table rect (tableLeft..tableRight /
                    // tableTop..tableBottom), and FORCE_BAR_X (203 of a 2400
                    // reference width) sits well inside that horizontal
                    // range for any normal table layout — so tableDragView
                    // was very likely still covering the force-bar column at
                    // dispatch time and silently absorbing/blocking the
                    // synthetic touch before it ever reached the game.
                    // sensStripContainer is removed too since it's the other
                    // touchable AIMING overlay and its exact footprint isn't
                    // guaranteed clear of the dispatch column either. Each
                    // remove*() is safe to call twice; hideAimingOverlay()'s
                    // later calls become no-ops via their isAttachedToWindow
                    // guards.
                    removeForceBarView()
                    removeTableDragView()
                    removeSensitivityStrip()
                    hideAimingMagnifier()

                    // NOTE: startBtnContainer sits at the top-right edge and
                    // resetBtnContainer sits at the bottom-left corner; both
                    // are nowhere near the force-bar dispatch column
                    // (x=203ish), so neither needs to be made non-touchable
                    // for this dispatch.

                    // Change 2: mark gesture as in-flight immediately before
                    // dispatching it, so stop() can detect an incomplete
                    // gesture dispatch and wait for it to drain rather than
                    // tearing down native state beneath it.
                    gestureInFlight = true

                    hideAimingOverlay()
                    transitionTo(State.LOCKED)

                    // Match the working sweep debug path exactly: overlays
                    // are torn down and state transitions to LOCKED FIRST,
                    // synchronously, then the actual gesture dispatch is
                    // scheduled afterward via postDelayed — not called inline
                    // in the same stack frame as the teardown.
                    mainHandler.postDelayed({
                        val service = IndirectGestureService.instance
                        if (service != null) {
                            val dispatchX = scaledForceBarX
                            logD("fireForceDrag: dispatching at x=$dispatchX " +
                                    "(forceBarCenterX=$scaledForceBarX barWidth=$scaledForceBarWidth) " +
                                    "topY=$scaledForceBarTopY endY=$endY potPower=$potPower serviceConnected=true")

                            service.fireForceDrag(
                                dispatchX,
                                scaledForceBarTopY,
                                endY,
                                durationMs = 1000L,
                                barWidth = scaledForceBarWidth,
                                onComplete = {
                                    gestureInFlight = false
                                }
                            )
                        } else {
                            gestureInFlight = false
                            logW("gesture service not connected — force drag skipped " +
                                    "(x=$scaledForceBarX topY=$scaledForceBarTopY endY=$endY potPower=$potPower)")
                        }
                    }, 100L)
                } else {
                    // No pot latched during this drag — stay in AIMING and
                    // fall back to showing the default full-power path.
                    hideAimingMagnifier()
                    updateAimLive(power = 1.0, isLiveDrag = false)
                }
            }
        }
        return true
    }

    // -----------------------------------------------------------------------
    // Table-drag aiming (AIMING state — drag-to-aim only)
    // -----------------------------------------------------------------------

    /**
     * Computes the raw (unscaled) aim angle from the cue ball toward a
     * screen point, in the same pixel-space / Y-down convention used
     * everywhere else in this file (matches the cue→target initialisation
     * in onLockClicked).
     */
    private fun rawAngleToPoint(screenX: Double, screenY: Double): Double =
        atan2(
            screenY * captureScale - cueBallY,
            screenX * captureScale - cueBallX
        )

    /** Normalizes an angle to (-PI, PI]. */
    private fun normalizeAngle(angleRad: Double): Double = when {
        angleRad >  PI  -> angleRad - 2 * PI
        angleRad <= -PI -> angleRad + 2 * PI
        else            -> angleRad
    }

    /**
     * Handles drag-to-aim over the pool table region during AIMING. Tapping
     * the table has no effect on the aim angle — only movement while the
     * finger is down rotates the shot:
     *
     * - ACTION_DOWN: does NOT touch currentAngleRad. Only starts delta
     *   tracking (records the raw angle-to-finger direction at the touch-
     *   down point) so a subsequent drag has a baseline to compute from.
     * - ACTION_MOVE: continuous drag-to-aim — computes the raw angle toward
     *   the current finger position, takes its delta from the previous
     *   sample's raw angle, scales that delta by dpadStepPx (the selected
     *   sensitivity), and applies the scaled delta to currentAngleRad.
     * - ACTION_UP/ACTION_CANCEL: clears delta tracking; nothing else to do
     *   here since firing only happens via the force-bar release gesture.
     *
     * Every ACTION_MOVE re-runs updateAimLive() at full power so the
     * stitched path refreshes in real time. ACTION_DOWN does not call
     * updateAimLive() since the angle hasn't changed yet.
     */
    private fun onTableDragTouch(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                // Record the baseline raw angle only — the aim angle itself
                // is left untouched until the finger actually moves.
                lastRawDragAngleRad = rawAngleToPoint(ev.rawX.toDouble(), ev.rawY.toDouble())

                // Surface the magnifier for the aiming drag too, same as the
                // force-bar power drag — previously showAimingMagnifier()
                // was only wired into onForceBarTouch, so the magnifier only
                // ever appeared while dragging power, never while aiming.
                showAimingMagnifier()
            }

            MotionEvent.ACTION_MOVE -> {
                val raw = rawAngleToPoint(ev.rawX.toDouble(), ev.rawY.toDouble())
                val previousRaw = lastRawDragAngleRad
                if (previousRaw != null) {
                    // Shortest-path delta between two raw angles, so a
                    // wrap-around near ±PI doesn't produce a huge jump.
                    var delta = raw - previousRaw
                    if (delta >  PI) delta -= 2 * PI
                    if (delta <= -PI) delta += 2 * PI

                    currentAngleRad = normalizeAngle(currentAngleRad + delta * dpadStepPx)
                }
                lastRawDragAngleRad = raw
                updateAimLive(power = 1.0, isLiveDrag = false)
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                lastRawDragAngleRad = null
                hideAimingMagnifier()
            }
        }
        return true
    }

    // -----------------------------------------------------------------------
    // Table tap
    // -----------------------------------------------------------------------

    private fun onTableTap(rawX: Double, rawY: Double) {
        virtualX = clampToTableD(rawX, horizontal = true)
        virtualY = clampToTableD(rawY, horizontal = false)
        logD("onTableTap virtualX=%.2f virtualY=%.2f".format(virtualX, virtualY))

        val isCueBallStep = (state == State.LOCKING_CUE_BALL)
        if (isCueBallStep)
            placeRingMarker(cueBallMarker,   cueBallMarkerParams,   virtualX, virtualY, cueBallMarkerAdded)
        else
            placeRingMarker(targetBallMarker, targetBallMarkerParams, virtualX, virtualY, targetBallMarkerAdded)
    }

    // -----------------------------------------------------------------------
    // Clamping
    // -----------------------------------------------------------------------

    private fun clampToTableD(v: Double, horizontal: Boolean): Double =
        if (horizontal) v.coerceIn(tableLeft, tableRight)
        else            v.coerceIn(tableTop,  tableBottom)

    // -----------------------------------------------------------------------
    // Ring marker placement — boundary (a)
    // -----------------------------------------------------------------------

    private fun placeRingMarker(
        marker: DashedRingView,
        params: WindowManager.LayoutParams,
        screenX: Double,
        screenY: Double,
        alreadyAttached: Boolean
    ) {
        val size = params.width
        params.x = (screenX - size / 2.0).roundToInt()   // boundary (a)
        params.y = (screenY - size / 2.0).roundToInt()   // boundary (a)

        if (alreadyAttached && marker.isAttachedToWindow) {
            try { windowManager.updateViewLayout(marker, params) }
            catch (e: Exception) { logE("placeRingMarker updateViewLayout failed", e) }
        } else {
            addOverlayView(marker, params)
        }
    }

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------

    private fun transitionTo(newState: State) {
        logD("transitionTo $state → $newState")
        state = newState
        applyStateUi()
    }

    private fun applyStateUi() {
        logD("applyStateUi(): posting UI update for state=$state")
        mainHandler.post {
            try {
                logD("applyStateUi(): posted block running, state=$state")
                when (state) {
                    State.IDLE                -> {
                        logD("applyStateUi(): calling applyIdleUi()")
                        applyIdleUi()
                        logD("applyStateUi(): applyIdleUi() returned OK")
                    }
                    State.LOCKING_CUE_BALL    -> showLockingOverlays(isCueBall = true)
                    State.LOCKING_TARGET_BALL -> showLockingOverlays(isCueBall = false)
                    State.AIMING              -> showAimingOverlay()
                    State.LOCKED              -> { /* path stays visible; dpad already hidden */ }
                }
                logD("applyStateUi(): calling updateAddButtonUi()")
                // Always sync the btnAdd label to reflect current state.
                updateAddButtonUi()
                logD("applyStateUi(): updateAddButtonUi() returned OK — posted block complete")
            } catch (t: Throwable) {
                logE("applyStateUi(): posted block threw for state=$state", t)
                throw t
            }
        }
    }

    private fun applyIdleUi() {
        logD("applyIdleUi(): begin, cueBallMarkerAdded=$cueBallMarkerAdded targetBallMarkerAdded=$targetBallMarkerAdded btnAddInitialized=${::btnAdd.isInitialized}")
        if (cueBallMarkerAdded)    { removeOverlayViewIfAttached(cueBallMarker);    cueBallMarkerAdded    = false }
        if (targetBallMarkerAdded) { removeOverlayViewIfAttached(targetBallMarker); targetBallMarkerAdded = false }
        btnAdd.isEnabled = true
        logD("applyIdleUi(): end")
    }

    // -----------------------------------------------------------------------
    // btnAdd dual-purpose UI sync
    // -----------------------------------------------------------------------

    /**
     * Sets btnAdd label based on current state:
     *   IDLE  → "▶" (start the flow)
     *   other → "↺" (restart / clear and begin again)
     */
    private fun updateAddButtonUi() {
        logD("updateAddButtonUi(): btnAddInitialized=${::btnAdd.isInitialized} state=$state")
        btnAdd.text = if (state == State.IDLE) "▶" else "↺"
        logD("updateAddButtonUi(): done")
    }

    // -----------------------------------------------------------------------
    // Click handlers
    // -----------------------------------------------------------------------

    /**
     * btnAdd click handler — dual-purpose depending on state:
     *   IDLE       → start the flow (LOCKING_CUE_BALL)
     *   any other  → run full reset logic (same as old onResetClicked)
     */
    private fun onAddClicked() {
        if (state == State.IDLE) {
            transitionTo(State.LOCKING_CUE_BALL)
        } else {
            onResetClicked()
        }
    }

    /**
     * Full clear/restart — invoked from btnAdd when state != IDLE.
     * Clears all native state, resets the path overlay, clears locally-held
     * ball/angle state, defensively removes any lingering overlays, and
     * returns to IDLE.
     */
    private fun onResetClicked() {
        logD("onResetClicked")

        // Clear native solver state.
        try { QeightJNI.clearIndirectShot() } catch (e: Exception) { logE("clearIndirectShot", e) }

        // Hide and reset the path overlay.
        indirectPathView?.setPathData(IndirectPathData(visible = false))
        if (indirectPathViewAdded) {
            indirectPathView?.let { removeOverlayViewIfAttached(it) }
            indirectPathViewAdded = false
        }
        try { QeightJNI.renderOverlay() } catch (e: Exception) { logE("renderOverlay", e) }

        // Reset all locally-held ball/angle state.
        cueBallX = 0.0; cueBallY = 0.0
        targetBallX = 0.0; targetBallY = 0.0
        currentAngleRad = 0.0
        virtualX = 0.0; virtualY = 0.0

        // Defensively remove any lingering overlays.
        stopDpadHold()
        captureThreadRunning.set(false)
        aimingMagnifierFocus = null
        removeDpadCluster()
        removeForceBarView()
        removeSensitivityStrip()
        removeTableDragView()
        removeMagnifierPanel()
        removeTableTapWindow()
        // Defensive: guarantee the start/exit button windows are never left
        // stuck non-touchable if a reset happens mid-dispatch (e.g. gesture
        // never completes and onComplete never fires).
        if (::startBtnParams.isInitialized) setWindowTouchable(startBtnContainer, startBtnParams, touchable = true)
        if (::resetBtnParams.isInitialized) setWindowTouchable(resetBtnContainer, resetBtnParams, touchable = true)
        currentDragPower = 0.0
        latchedPotPower = null
        lastRawDragAngleRad = null

        // Remove ring markers if still attached.
        if (cueBallMarkerAdded)    { removeOverlayViewIfAttached(cueBallMarker);    cueBallMarkerAdded    = false }
        if (targetBallMarkerAdded) { removeOverlayViewIfAttached(targetBallMarker); targetBallMarkerAdded = false }

        transitionTo(State.IDLE)
    }

    // -----------------------------------------------------------------------
    // onCleanFrame — feeds the magnifier panel during locking states, and
    // during an active AIMING force-bar drag (auto-focused on the first
    // bounce ghost circle — see aimingMagnifierFocus).
    // -----------------------------------------------------------------------

    fun onCleanFrame(buffer: ByteBuffer) {
        if (!captureThreadRunning.get()) return

        val mv = magnifierView ?: return
        val cropSize = mv.requiredCropSizePx

        // During an active AIMING drag, follow the first-bounce point;
        // otherwise (locking states) follow the ring-marker cursor as before.
        val focus = aimingMagnifierFocus
        val cx: Int
        val cy: Int
        if (focus != null) {
            cx = focus.first
            cy = focus.second
        } else {
            cx = (virtualX * captureScale).roundToInt()
            cy = (virtualY * captureScale).roundToInt()
        }

        val half  = cropSize / 2
        val x0    = (cx - half).coerceIn(0, (captureWidth  - cropSize).coerceAtLeast(0))
        val y0    = (cy - half).coerceIn(0, (captureHeight - cropSize).coerceAtLeast(0))
        val cropW = cropSize.coerceAtMost(captureWidth  - x0)
        val cropH = cropSize.coerceAtMost(captureHeight - y0)

        val cropPixels = IntArray(cropW * cropH)
        val buf = buffer.duplicate().also { it.order(ByteOrder.nativeOrder()) }

        for (row in 0 until cropH) {
            val srcRow = ((y0 + row) * captureWidth + x0) * 4
            val dstRow = row * cropW
            for (col in 0 until cropW) {
                val pos = srcRow + col * 4
                val r = buf.get(pos    ).toInt() and 0xFF
                val g = buf.get(pos + 1).toInt() and 0xFF
                val b = buf.get(pos + 2).toInt() and 0xFF
                val a = buf.get(pos + 3).toInt() and 0xFF
                cropPixels[dstRow + col] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val fw = cropW; val fh = cropH
        mainHandler.post { magnifierView?.updateFrame(cropPixels, fw, fh) }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /**
     * Converts a flat [x0,y0,x1,y1,...] FloatArray (as returned by
     * getIndirectPathPx) into the List<Pair<Float,Float>> shape
     * IndirectPathData expects, in screen-pixel space.
     *
     * cueBallX/Y and targetBallX/Y are sent into updateIndirectAim in
     * capture-scale coordinates (virtualX * captureScale — see
     * onLockClicked), so the stitched path native solves and returns via
     * getIndirectPathPx() comes back in that same capture-scale space.
     * IndirectPathView draws on a full-screen overlay in screen-pixel
     * space — the same space virtualX/virtualY and the
     * cueBallMarker/targetBallMarker ring placements use directly (see
     * placeRingMarker). Dividing by captureScale here converts the path
     * back to screen space so it lines up with those markers instead of
     * being drawn shrunk toward the table's top-left corner.
     *
     * An odd trailing coordinate (malformed array) is dropped rather than
     * crashing.
     */
    private fun FloatArray.toPointList(): List<Pair<Float, Float>> {
        val pairCount = size / 2
        return (0 until pairCount).map { i ->
            Pair(this[i * 2] / captureScale, this[i * 2 + 1] / captureScale)
        }
    }
}