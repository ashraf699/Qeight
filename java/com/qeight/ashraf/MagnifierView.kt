package com.ashraf.qeight

import android.content.Context
import android.graphics.*
import android.view.View
import kotlin.math.roundToInt

/**
 * MagnifierView
 *
 * Displays a live, cropped, zoomed-in preview of the most recent raw RGBA8888 capture
 * frame, centered on a caller-supplied point, with a dotted alignment circle at the
 * exact view center.
 *
 * True optical zoom: as [zoomFactor] increases, the *source* region the caller should
 * crop shrinks (see [requiredCropSizePx]), and that smaller region is stretched to
 * fill the full view. The caller (IndirectShotController.onCleanFrame) must crop
 * [requiredCropSizePx] × [requiredCropSizePx] source pixels — NOT a fixed CROP_SIZE —
 * for zoomFactor to have any visible effect. Passing a fixed-size crop regardless of
 * zoomFactor (the old behavior) makes zoom a no-op once the view's own pixel
 * dimensions are the binding constraint on scale.
 *
 * Zoom is controlled entirely via the public [zoomFactor] property; there is no
 * in-view slider. Touch events are not consumed — all touches pass through to whatever
 * is behind/above this view in the window stack.
 *
 * No references to OverlayService, ScreenCaptureManager, or QeightJNI are made here.
 */
class MagnifierView(context: Context) : View(context) {

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Current zoom level, clamped to [MIN_ZOOM, MAX_ZOOM]. Setting it triggers
     * invalidate() and recomputes [requiredCropSizePx] — callers should re-read
     * [requiredCropSizePx] after changing this and use it for the next crop.
     * [onZoomChanged] is invoked whenever this value changes.
     */
    var zoomFactor: Float = 10f
        set(value) {
            val clamped = value.coerceIn(MIN_ZOOM, MAX_ZOOM)
            if (clamped != field) {
                field = clamped
                // The crop size required for the new zoom differs from whatever is
                // currently in cropBitmap. Drop the stale frame immediately so onDraw
                // shows the neutral placeholder instead of stretching an old-zoom
                // image until the next updateFrame() call lands — otherwise the
                // change can look like it "didn't apply" until the next frame.
                hasFrame = false
                validContentW = 0
                validContentH = 0
                onZoomChanged?.invoke(clamped)
                invalidate()
            }
        }

    /**
     * The source-buffer side length (in raw capture pixels) the caller should crop
     * around the target point for the current [zoomFactor]. Shrinks as zoomFactor
     * increases, e.g. at zoomFactor=1 this equals [BASE_CROP_SIZE]; at zoomFactor=10
     * it's BASE_CROP_SIZE/10. Always >= [MIN_CROP_SIZE] so the crop never degenerates
     * to nothing at MAX_ZOOM.
     */
    val requiredCropSizePx: Int
        get() = (BASE_CROP_SIZE / zoomFactor).roundToInt().coerceAtLeast(MIN_CROP_SIZE)

    /**
     * The reference radius (in source-buffer pixels) that the dotted alignment circle
     * represents. Callers should keep this in sync with whatever ghost-ball radius
     * constant is used elsewhere in the app. Changing this triggers invalidate().
     */
    var ghostRadiusPx: Float = 22f
        set(value) {
            if (value != field) {
                field = value
                invalidate()
            }
        }

    /**
     * Invoked whenever [zoomFactor] changes, so the caller can persist or react to the
     * new value. Left in place as available (if currently unused) API.
     */
    var onZoomChanged: ((Float) -> Unit)? = null

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    companion object {
        /** Source crop side length (px) used at zoomFactor = 1 (no zoom). */
        const val BASE_CROP_SIZE      = 300
        /** Backwards-compatible alias — max bitmap buffer size, sized for BASE_CROP_SIZE. */
        const val CROP_SIZE           = BASE_CROP_SIZE
        /** Floor on the source crop side length so it never shrinks to nothing at MAX_ZOOM. */
        private const val MIN_CROP_SIZE = 20
        private const val MIN_ZOOM    = 1f
        private const val MAX_ZOOM    = 15f
        private const val DASH_ON_DP  = 6f
        private const val DASH_OFF_DP = 4f
    }

    // -------------------------------------------------------------------------
    // Pixel-density helper (computed once)
    // -------------------------------------------------------------------------

    private val density: Float = context.resources.displayMetrics.density

    private fun dp(value: Float): Float = value * density

    // -------------------------------------------------------------------------
    // Persistent bitmap + pixel buffer (300 × 300, reused across frames)
    // -------------------------------------------------------------------------

    private val cropBitmap: Bitmap =
        Bitmap.createBitmap(BASE_CROP_SIZE, BASE_CROP_SIZE, Bitmap.Config.ARGB_8888)

    /** Actual valid (non-padding) content size within cropBitmap, set by updateFrame(). */
    private var validContentW: Int = BASE_CROP_SIZE
    private var validContentH: Int = BASE_CROP_SIZE

    /** True once at least one frame has been supplied via updateFrame(). */
    private var hasFrame: Boolean = false

    // -------------------------------------------------------------------------
    // Paint objects – built once as fields
    // -------------------------------------------------------------------------

    /** Paints the cropped bitmap onto the canvas. */
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Background fill when no frame has arrived yet. */
    private val placeholderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF1A1A2E.toInt()
        style = Paint.Style.FILL
    }

    /** Dotted alignment circle. */
    private val circlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color       = 0xFFFFFFFF.toInt()
        style       = Paint.Style.STROKE
        strokeWidth = 2f
        pathEffect  = DashPathEffect(
            floatArrayOf(dp(DASH_ON_DP), dp(DASH_OFF_DP)),
            0f
        )
    }

    // -------------------------------------------------------------------------
    // Reused drawing objects – allocated once, mutated in onDraw
    // -------------------------------------------------------------------------

    private val srcRect       = Rect()
    private val bitmapDstRect = RectF()

    // -------------------------------------------------------------------------
    // Public frame-update method
    // -------------------------------------------------------------------------

    /**
     * Accepts a pre-cropped, pre-converted ARGB_8888 IntArray that was prepared
     * entirely on the capture thread before ScreenCaptureManager.releaseFrame() was
     * called. No ByteBuffer, Image, or ScreenCaptureManager reference is touched here,
     * so this method cannot race with releaseFrame() by construction.
     *
     * Must be called on the main thread (e.g. via Handler.post).
     *
     * The caller should crop [requiredCropSizePx] × [requiredCropSizePx] source pixels
     * (not a fixed size) so that [zoomFactor] has a visible effect — see class doc.
     *
     * @param cropPixels  Row-major IntArray of size [cropWidth] × [cropHeight] in
     *                    Android ARGB_8888 format (each Int = 0xAARRGGBB). May be
     *                    smaller than requiredCropSizePx × requiredCropSizePx when the
     *                    crop window was clamped near a frame boundary; pixels outside
     *                    that region are filled with transparent black.
     * @param cropWidth   Width of the supplied crop in pixels (≤ BASE_CROP_SIZE).
     * @param cropHeight  Height of the supplied crop in pixels (≤ BASE_CROP_SIZE).
     */
    fun updateFrame(cropPixels: IntArray, cropWidth: Int, cropHeight: Int) {
        val w = cropWidth.coerceIn(0, BASE_CROP_SIZE)
        val h = cropHeight.coerceIn(0, BASE_CROP_SIZE)

        // The valid (actually-captured) region may be smaller than requiredCropSizePx
        // near frame edges; track it so onDraw scales only the real content to fill
        // the view, rather than stretching padding along with it.
        validContentW = w
        validContentH = h

        if (w > 0 && h > 0) {
            cropBitmap.setPixels(cropPixels, 0, w, 0, 0, w, h)
        }
        // Clear columns to the right of the valid region.
        if (w < BASE_CROP_SIZE && h > 0) {
            val clear = IntArray((BASE_CROP_SIZE - w) * h)
            cropBitmap.setPixels(clear, 0, BASE_CROP_SIZE - w, w, 0, BASE_CROP_SIZE - w, h)
        }
        // Clear rows below the valid region.
        if (h < BASE_CROP_SIZE) {
            val clear = IntArray(BASE_CROP_SIZE * (BASE_CROP_SIZE - h))
            cropBitmap.setPixels(clear, 0, BASE_CROP_SIZE, 0, h, BASE_CROP_SIZE, BASE_CROP_SIZE - h)
        }

        hasFrame = true
        invalidate()
    }

    // -------------------------------------------------------------------------
    // Drawing
    // -------------------------------------------------------------------------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        if (!hasFrame) {
            // Neutral dark placeholder when no frame has arrived yet.
            canvas.drawRect(0f, 0f, viewW, viewH, placeholderPaint)
            return
        }

        // ---- 1. Scale the valid captured sub-region to fill the full view area ----
        //
        // The caller crops requiredCropSizePx × requiredCropSizePx source pixels (which
        // shrinks as zoomFactor rises) and that region — validContentW × validContentH,
        // stored at the top-left of cropBitmap — is stretched to fill the view exactly.
        // This is where the actual optical zoom happens: a smaller source region
        // stretched over the same view area looks more zoomed in. Unlike the old
        // fixed-crop approach, this is NOT re-clamped by zoomFactor here — the capture
        // side already did the zoom by choosing how much source to crop.
        if (validContentW <= 0 || validContentH <= 0) {
            canvas.drawRect(0f, 0f, viewW, viewH, placeholderPaint)
            return
        }

        srcRect.set(0, 0, validContentW, validContentH)
        bitmapDstRect.set(0f, 0f, viewW, viewH)

        canvas.drawBitmap(cropBitmap, srcRect, bitmapDstRect, bitmapPaint)

        // NOTE: No separate alignment circle is drawn here. The captured crop already
        // contains the real on-screen dotted ring marker (DashedRingView), so drawing
        // a second circle here would duplicate it. See ghostRadiusPx/circlePaint below
        // — retained as available API/fields but intentionally unused in onDraw.
    }
}