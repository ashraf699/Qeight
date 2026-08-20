/**
 * IndirectPathOverlay.kt
 *
 * Full-screen, non-touchable overlay that draws the solved indirect-shot
 * path: the sequence of cushion contact points, the ghost-ball position,
 * and the target pocket point — all connected by line segments / markers.
 *
 * This view is purely a renderer — it holds no solving logic. The
 * controller is responsible for populating it with an IndirectPathData
 * once QeightJNI.solveIndirectShot() plus its per-candidate path getters
 * (getIndirectCandidatePathPx) have produced results. The controller shows
 * one candidate at a time via repeated setPathData() calls — this view
 * remains a single-path renderer and does not manage a list of candidates
 * itself. The controller is also responsible for populating
 * IndirectPathData.ghostRadiusPx from QeightJNI.getLastIndirectGhostRadiusPx()
 * — this view never hardcodes or independently derives that radius.
 *
 * The entire path is rendered with a single uniform stroke style — white,
 * solid, same thickness throughout — so it reads as one continuous line
 * from launch through to the pocket.
 */
package com.ashraf.qeight

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

// ---------------------------------------------------------------------------
// IndirectPathData — plain data holder for everything IndirectPathView draws
// ---------------------------------------------------------------------------

/**
 * @param segments      Ordered list of (x, y) points describing the full
 *                      path — cue ball -> each cushion contact -> ghost
 *                      ball -> pocket. Drawn as connected line segments in
 *                      order. A ghost circle of radius [ghostRadiusPx] is
 *                      also drawn at every point except the first.
 * @param contacts      Cushion contact points (subset/overlay markers drawn
 *                      on top of the path at each cushion bounce).
 * @param pocket        Target pocket point, or null if not yet known.
 * @param visible       Master visibility switch — when false, onDraw
 *                      renders nothing (used to "hide" the overlay without
 *                      detaching it from the WindowManager).
 * @param ghostRadiusPx Ghost-circle rendering radius in device pixels. The
 *                      controller populates this from
 *                      QeightJNI.getLastIndirectGhostRadiusPx() (the same
 *                      canonical native radius direct mode uses) after each
 *                      successful updateIndirectAim() call, so this view
 *                      never hardcodes its own value. Defaults to 22.023f
 *                      (matching the native default) for the rare case a
 *                      caller constructs IndirectPathData before any native
 *                      call has run.
 */
data class IndirectPathData(
    val segments:      List<Pair<Float, Float>> = emptyList(),
    val contacts:      List<Pair<Float, Float>> = emptyList(),
    val pocket:        Pair<Float, Float>?      = null,
    val visible:       Boolean                  = false,
    val ghostRadiusPx: Float                    = 22.023f
)

// ---------------------------------------------------------------------------
// IndirectPathView — visual-only, not touchable
// ---------------------------------------------------------------------------

class IndirectPathView(context: Context) : View(context) {

    private val pathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 5f
        color       = Color.WHITE
    }

    private val contactPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFFFFD600.toInt()
    }

    private val ghostPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 4f
        color       = Color.WHITE
    }

    private val pocketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FF6D00")
    }

    private var data = IndirectPathData()

    init { setWillNotDraw(false) }

    /** Replace the drawn path data and trigger a redraw. */
    fun setPathData(newData: IndirectPathData) {
        if (data.segments.size > 10 && newData.segments.size < 2) {
            android.util.Log.w("IndirectPathView",
                "setPathData: segments dropped sharply " +
                        "(${data.segments.size} -> ${newData.segments.size}) — " +
                        "possible stale/truncated path from a concurrent update")
        }
        data = newData
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!data.visible) return

        if (data.segments.size >= 2) {
            for (i in 0 until data.segments.size - 1) {
                val (x1, y1) = data.segments[i]
                val (x2, y2) = data.segments[i + 1]
                canvas.drawLine(x1, y1, x2, y2, pathPaint)
            }

            // Ghost/reflection circle at every point except the first
            // (cue-ball start) — i.e. each cushion-bounce point plus the
            // final contact point. Radius comes from data.ghostRadiusPx,
            // which the controller sources from
            // QeightJNI.getLastIndirectGhostRadiusPx() — the same canonical
            // radius direct mode uses — rather than a hardcoded literal, so
            // indirect-mode ghost circles always track the native value.
            for (i in 1 until data.segments.size) {
                val (gx, gy) = data.segments[i]
                canvas.drawCircle(gx, gy, data.ghostRadiusPx, ghostPaint)
            }
        }

        data.contacts.forEach { (cx, cy) ->
            canvas.drawCircle(cx, cy, 8f, contactPaint)
        }

        data.pocket?.let { (px, py) ->
            canvas.drawCircle(px, py, 10f, pocketPaint)
        }
    }
}