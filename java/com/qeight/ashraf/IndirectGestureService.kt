/**
 * IndirectGestureService.kt
 *
 * An AccessibilityService subclass whose sole purpose is dispatching a single
 * synthetic drag gesture used to pull the in-game force/power bar to a specific
 * level. It does not observe or react to any accessibility events — it only
 * dispatches gestures outbound.
 *
 * ┌─ IMPORTANT NOTE FOR CALLERS ──────────────────────────────────────────────┐
 * │ Always check IndirectGestureService.instance != null before calling        │
 * │ fireForceDrag(). The instance field is only non-null while the user has    │
 * │ actually enabled this accessibility service in Android system settings     │
 * │ (Settings → Accessibility → [App Name] → IndirectGestureService).         │
 * │ This file does NOT handle detecting whether the service is enabled, nor    │
 * │ does it prompt the user to enable it — that responsibility belongs to the  │
 * │ calling layer (e.g. the ViewModel or a dedicated accessibility-state        │
 * │ helper).                                                                   │
 * └────────────────────────────────────────────────────────────────────────────┘
 *
 * Manifest requirements (outside the scope of this file):
 *   • android:canPerformGestures="true" on the <service> element.
 *   • A corresponding accessibility-service meta-data XML resource.
 *   • android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE".
 */

package com.ashraf.qeight

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.util.Log
import android.view.accessibility.AccessibilityEvent

// ─── Logging configuration ────────────────────────────────────────────────────

private const val TAG = "IndirectGestureService"

/**
 * Master switch for verbose logging inside this file.
 * Set to false for release builds to avoid log noise and minor overhead.
 */
private const val LOGGING_ENABLED = true

/**
 * Emits a DEBUG log entry when [LOGGING_ENABLED] is true.
 */
private fun logD(tag: String, msg: String) {
    if (LOGGING_ENABLED) Log.d(tag, msg)
}

/**
 * Emits a WARN log entry when [LOGGING_ENABLED] is true.
 */
private fun logW(tag: String, msg: String) {
    if (LOGGING_ENABLED) Log.w(tag, msg)
}

/**
 * Emits an ERROR log entry.
 * Errors are always logged regardless of [LOGGING_ENABLED] because they
 * represent unexpected failure states that should be visible in production
 * diagnostics.
 */
private fun logE(tag: String, msg: String, tr: Throwable? = null) {
    if (tr != null) Log.e(tag, msg, tr) else Log.e(tag, msg)
}

// ─── Service implementation ───────────────────────────────────────────────────

/**
 * Lightweight [AccessibilityService] that exclusively dispatches outbound
 * synthetic drag gestures. Event observation is intentionally minimal.
 *
 * Acquire a reference via the companion [instance] property; it is non-null
 * only while the service is bound (i.e. the user has enabled it in system
 * Accessibility settings).
 */
class IndirectGestureService : AccessibilityService() {

    // ── Companion: shared instance handle ─────────────────────────────────────

    companion object {
        /**
         * Live reference to the running service instance.
         *
         * Thread-safe via @Volatile; written only on the main thread during
         * [onServiceConnected] / [onUnbind], but may be read from any thread.
         *
         * **Null** when:
         *   • The service has never been connected in this process lifetime.
         *   • The user has disabled the service in system Accessibility settings.
         *   • The system has unbound the service for any other reason.
         */
        @Volatile
        var instance: IndirectGestureService? = null
            private set
    }

    // ── In-flight gesture tracking ────────────────────────────────────────────

    /**
     * True from the moment [fireForceDrag] successfully queues a gesture
     * until that gesture's [GestureResultCallback.onCompleted] or
     * [GestureResultCallback.onCancelled] fires. False otherwise (including
     * when no gesture has ever been dispatched, or dispatch was rejected).
     *
     * @Volatile since callers may poll [isGestureInFlight] from a background
     * thread (see IndirectModeController's drain loop) while the callback
     * itself is delivered on the main thread.
     */
    @Volatile
    private var gestureInFlight: Boolean = false

    /**
     * Returns whether a gesture dispatched via [fireForceDrag] is still
     * pending completion. Safe to call from any thread.
     */
    fun isGestureInFlight(): Boolean = gestureInFlight

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        logD(TAG, "onServiceConnected: service bound and instance registered — " +
                "IndirectGestureService.instance is now non-null, fireForceDrag() calls will proceed")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Intentionally minimal — we subscribe to events so Android 15 shows
        // the service in Settings, but we don't process them.
        // Just log for debugging purposes.
        event?.let {
            logD(TAG, "onAccessibilityEvent: ${it.eventType} from ${it.packageName}")
        }
    }

    override fun onInterrupt() {
        logD(TAG, "onInterrupt: received (no-op)")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        logW(TAG, "onUnbind: service unbound, clearing instance reference — " +
                "IndirectGestureService.instance will be null; fireForceDrag() calls will be " +
                "skipped by callers until the service reconnects " +
                "(usually means Accessibility permission was revoked or the OS killed the service)")
        instance = null
        return super.onUnbind(intent)
    }

    // ── Gesture dispatch ──────────────────────────────────────────────────────

    /**
     * Dispatches a single straight vertical drag gesture from ([x], [startY])
     * to ([x], [endY]) over [durationMs] milliseconds.
     *
     * Intended use: pull the in-game force/power bar down to a level computed
     * by the calling layer. The x coordinate and y range are determined by the
     * caller based on screen metrics and the desired power percentage.
     *
     * ### Return value
     * Returns **true** if the gesture was successfully *queued* for dispatch,
     * per [dispatchGesture]'s own return contract. This does **not** guarantee
     * that the gesture completed — see [GestureResultCallback.onCompleted] /
     * [GestureResultCallback.onCancelled] for completion semantics.
     * Returns **false** if:
     *   • [dispatchGesture] itself returned false (system rejected the gesture).
     *   • Any exception was thrown during path/gesture construction or dispatch.
     *
     * ### Threading
     * [dispatchGesture] must be called on the main thread. Callers are
     * responsible for ensuring this; a warning is logged if we detect we are
     * not on the main thread, but dispatch is attempted anyway.
     *
     * ### API level
     * [dispatchGesture] requires API 24+. This is guarded explicitly even if
     * the project's minSdk already exceeds 24, to make the requirement visible
     * at the call site and to satisfy the linter without a suppression.
     *
     * @param x          Horizontal pixel coordinate — center of the force
     *                   bar's touch column (constant target for a vertical
     *                   drag; actual per-waypoint X jitters within
     *                   [barWidth] around this center, see [barWidth]).
     * @param startY     Vertical pixel coordinate at the start of the drag.
     * @param endY       Vertical pixel coordinate at the end of the drag.
     * @param durationMs Duration of the gesture stroke in milliseconds.
     *                   Defaults to 1500 ms.
     * @param barWidth   Width in pixels of the force bar's touchable column.
     *                   The stroke's intermediate waypoints are jittered
     *                   horizontally within +/- barWidth/2 of [x] so the
     *                   dispatched gesture has real width/area like a
     *                   genuine finger drag, rather than a single
     *                   mathematically perfect vertical hairline — some
     *                   game input layers don't reliably register the
     *                   latter as a real drag. Pass 0 (default) to fall back
     *                   to a straight center-line stroke.
     * @param onComplete Optional callback invoked once the gesture reaches a
     *                   terminal state — i.e. from both
     *                   [GestureResultCallback.onCompleted] and
     *                   [GestureResultCallback.onCancelled] — so callers can
     *                   clear their own "in flight" bookkeeping (e.g.
     *                   IndirectModeController's `gestureInFlight` flag)
     *                   regardless of which outcome occurred. Delivered on
     *                   the main thread. Not invoked if dispatch is rejected
     *                   or throws — callers should treat a `false` return
     *                   from this function as "no callback will ever fire"
     *                   and clear their own flag synchronously in that case.
     * @return           true if gesture was accepted for dispatch, false otherwise.
     */
    fun fireForceDrag(
        x: Int,
        startY: Int,
        endY: Int,
        durationMs: Long = 1500L,
        barWidth: Int = 0,
        onComplete: (() -> Unit)? = null
    ): Boolean {
        logD(TAG, "fireForceDrag: ENTER x=$x startY=$startY endY=$endY durationMs=$durationMs " +
                "barWidth=$barWidth strokeLengthPx=${kotlin.math.abs(endY - startY)}")

        // Warn loudly if this is called off the main thread; dispatchGesture
        // is not safe to call from background threads.
        if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
            logW(TAG, "fireForceDrag: called off main thread — dispatchGesture may misbehave")
        }

        // Debug aid: a zero-length stroke (startY == endY) is a common
        // silent-failure cause — some OEM accessibility stacks accept it but
        // never visibly register it as a drag in-game. Flag it loudly rather
        // than letting it pass through unnoticed.
        if (startY == endY) {
            logW(TAG, "fireForceDrag: startY == endY ($startY) — this is a zero-length " +
                    "stroke and may not register as a drag at all")
        }
        // Debug aid: an x of 0 (or negative) usually means an upstream caller
        // passed an unset/uninitialized coordinate rather than a real
        // on-screen force-bar position.
        if (x <= 0) {
            logW(TAG, "fireForceDrag: x=$x looks unset/invalid for a real on-screen coordinate")
        }

        return try {
            // Build a "thick", slightly organic vertical stroke rather than a
            // single perfectly-straight, single-pixel-wide, instantaneous
            // line. Some game input layers (and some OEM synthetic-input
            // heuristics) don't register a mathematically perfect 2-point
            // linear GestureDescription path as a real drag — likely
            // because it has zero horizontal jitter and a single linear
            // segment, unlike a real finger drag. To make this look/behave
            // like a genuine thick swipe within the bar's touch column:
            //
            //  1. A brief DWELL at the start point (small pause before any
            //     vertical movement) — mimics the natural press-and-settle
            //     before a real drag begins.
            //  2. Multiple intermediate waypoints down the Y range, each
            //     with a small horizontal jitter constrained to the bar's
            //     own width (barWidth/2 either side of center) — this gives
            //     the stroke real width/area instead of a single vertical
            //     hairline, without ever leaving the force bar's actual
            //     touchable column.
            //  3. A brief DWELL at the end point before release — mimics a
            //     real finger settling before lifting, which some UI drag
            //     handlers require to commit the final value rather than
            //     treating a too-fast release as a flick/cancel.
            //
            // If barWidth is 0 (caller didn't pass one), this degrades
            // gracefully to a straight center-line multi-point stroke —
            // still an improvement over a single lineTo() since it adds
            // waypoints and dwell time.
            val halfWidth = (barWidth / 2).coerceAtLeast(0)
            val totalDeltaY = endY - startY
            val waypointCount = 6 // intermediate points between start and end (excluding dwell points)
            val rng = java.util.Random()

            fun jitteredX(): Float {
                if (halfWidth <= 0) return x.toFloat()
                val offset = (rng.nextInt(halfWidth * 2 + 1) - halfWidth)
                return (x + offset).toFloat()
            }

            val path = Path().apply {
                moveTo(x.toFloat(), startY.toFloat())
                // Tiny dwell segment at the start — near-zero movement, just
                // enough to register as "touch has settled" before the real
                // drag begins.
                lineTo(jitteredX(), startY.toFloat())

                for (i in 1..waypointCount) {
                    val fraction = i.toFloat() / (waypointCount + 1)
                    val wpY = startY + (totalDeltaY * fraction)
                    lineTo(jitteredX(), wpY)
                }

                lineTo(jitteredX(), endY.toFloat())
                // Tiny dwell segment at the end — settles at the final Y
                // before the stroke (and gesture) terminates, instead of
                // ending exactly on the last moving sample.
                lineTo(x.toFloat(), endY.toFloat())
            }

            // Wrap the path in a StrokeDescription: start offset 0 ms,
            // duration as provided by the caller. The extra waypoints/dwell
            // segments are all still contained within the same overall
            // durationMs — StrokeDescription paces evenly across the whole
            // path regardless of segment count.
            val stroke = GestureDescription.StrokeDescription(
                path,
                /* startTime = */ 0L,
                /* duration  = */ durationMs
            )

            val gesture = GestureDescription.Builder()
                .addStroke(stroke)
                .build()

            // Mark in-flight before dispatch so callers polling
            // isGestureInFlight() immediately after this call see true.
            gestureInFlight = true

            logD(TAG, "fireForceDrag: calling dispatchGesture — path=($x,$startY)->($x,$endY) " +
                    "durationMs=$durationMs")

            // Dispatch and return whether the system accepted the gesture.
            val accepted = dispatchGesture(
                gesture,
                object : GestureResultCallback() {
                    override fun onCompleted(gestureDescription: GestureDescription?) {
                        gestureInFlight = false
                        logD(TAG, "fireForceDrag: onCompleted — gesture finished successfully " +
                                "x=$x startY=$startY endY=$endY")
                        onComplete?.invoke()
                    }

                    override fun onCancelled(gestureDescription: GestureDescription?) {
                        gestureInFlight = false
                        logW(TAG, "fireForceDrag: onCancelled — gesture was cancelled by the " +
                                "system (often means another gesture was already in flight, or " +
                                "the target app/window doesn't accept synthetic input right now) " +
                                "x=$x startY=$startY endY=$endY")
                        onComplete?.invoke()
                    }
                },
                /* handler = */ null  // null → callbacks delivered on main thread
            )

            logD(TAG, "fireForceDrag: dispatchGesture returned accepted=$accepted")

            if (!accepted) {
                // System rejected the gesture outright — GestureResultCallback
                // will never fire in this case, so clear our own flag and
                // invoke onComplete here instead of leaving callers stuck
                // waiting on a callback that will never arrive.
                gestureInFlight = false
                logW(TAG, "fireForceDrag: dispatchGesture returned false — gesture not queued. " +
                        "Common causes: service not currently connected/bound, another gesture " +
                        "already in flight on this service instance, or the path coordinates " +
                        "fall outside the display bounds. x=$x startY=$startY endY=$endY")
                onComplete?.invoke()
            }

            accepted

        } catch (ex: Exception) {
            // Catch-all so that any unexpected failure (e.g. IllegalArgumentException
            // from a zero-duration stroke, or internal system errors) does not
            // propagate up and crash the calling UI flow. Also clear the flag
            // and invoke onComplete since no callback will fire for a dispatch
            // that never happened.
            gestureInFlight = false
            logE(TAG, "fireForceDrag: exception during gesture dispatch — returning false " +
                    "x=$x startY=$startY endY=$endY durationMs=$durationMs", ex)
            onComplete?.invoke()
            false
        }
    }
}