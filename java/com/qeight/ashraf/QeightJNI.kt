package com.ashraf.qeight

import android.content.res.AssetManager
import android.view.Surface
import java.nio.ByteBuffer

/**
 * QeightJNI — Kotlin singleton that declares all native function bindings.
 *
 * Lives in its own file so every Kotlin file in the package can import it
 * without circular dependency. The companion `init` block loads the single
 * shared library that contains the entire native pipeline.
 *
 * Every `external` function here has a matching C implementation in QeightJNI.cpp
 * following the Java_com_ashraf_qeight_QeightJNI_* naming convention.
 */
object QeightJNI {

    init {
        // Load the single shared library containing the entire native pipeline.
        // The library name matches the CMakeLists.txt target: qeight_pipeline → libqeight_pipeline.so
        System.loadLibrary("qeight_pipeline")
    }

    // ── Pipeline lifecycle ──────────────────────────────────────────────────

    /**
     * Initialize the image processing pipeline with capture-buffer and ROI dimensions.
     *
     * [screenW] and [screenH] are the dimensions of the downscaled capture buffer
     * (i.e. the pixel dimensions of the ImageReader frame), NOT the native device
     * display resolution.
     *
     * [captureScale] is the ratio (captureHeight / nativeHeight, always in the
     * range (0, 1]) used internally by the native pipeline to scale geometric
     * constants (e.g. ball radius, pocket size) that were originally calibrated
     * against native-resolution pixels.
     */
    external fun initPipeline(
        screenW: Int, screenH: Int,
        captureScale: Float,
        roiX1: Int, roiY1: Int, roiX2: Int, roiY2: Int
    )

    /**
     * Historical no-op kept only for backward compatibility with existing call sites.
     * There is no Vulkan compute subsystem, no GPU-side processing, and no shader
     * pipeline anywhere in the current native code. The native implementation does
     * nothing.
     */
    external fun initVulkan(assetManager: AssetManager)

    /**
     * Constructs the native overlay renderer — a CPU/software pixel rasterizer, not
     * a GPU renderer — targeting the ANativeWindow from [surface].
     *
     * [screenW] and [screenH] must match the capture-buffer dimensions used everywhere
     * else (the same ones passed to [initPipeline] and to every [processFrame] buffer).
     * The renderer uses them to scale AT-strip and ghost-circle coordinates up to the
     * physical display surface. There is no NDC math or surface pre-rotation transform
     * involved — coordinates are scaled directly in pixel space.
     */
    external fun initRenderer(surface: Surface, screenW: Int, screenH: Int)

    // ── Per-frame processing ────────────────────────────────────────────────

    /**
     * Process one captured RGBA8888 frame.
     * directBuffer is the DirectByteBuffer from ImageReader — zero-copy.
     * Returns true on a successful call; false if the pipeline has not been
     * initialized yet or the supplied buffer fails validation.
     */
    external fun processFrame(directBuffer: ByteBuffer): Boolean

    /**
     * Render the current AT strips and overlays synchronously to the ANativeWindow buffer.
     * This function performs a complete render directly to the surface and returns only
     * once the frame has been fully drawn and posted to the compositor — no separate
     * wait call is needed or available. Calling code may immediately proceed to the
     * next acquire-process iteration upon return.
     */
    external fun renderOverlay()

    // ── Parameter setters ───────────────────────────────────────────────────

    /** Update the ROI bounding box after calibration save. */
    external fun setRoi(x1: Int, y1: Int, x2: Int, y2: Int)

    /** Set the number of CBC (cue ball cut) cushion reflections, range 0–8. */
    external fun setCbcReflections(count: Int)

    /** Set the number of TGT (target ball) cushion reflections, range 0–8. */
    external fun setTgtReflections(count: Int)

    /** Set the trajectory power as a percentage (0–100). */
    external fun setTrajectoryPower(percent: Int)

    /** Set the cue force stat value. */
    external fun setCueForce(stat: Int)

    /** Set the cue spin stat value. */
    external fun setCueSpin(stat: Int)

    /**
     * Enable or disable cushion shots mode.
     * When false, reflection counts are forced to 0.
     */
    external fun setCushionShots(enabled: Boolean)

    /** Set the overlay line/ghost circle color as RGBA byte components (0–255 each). */
    external fun setOverlayColor(r: Int, g: Int, b: Int, a: Int)

    /** Set overlay line thickness in pixels, range 1–8. */
    external fun setLineThickness(thickness: Int)

    /** Toggles the translucent fill band drawn between the dotted guide rails. [alpha] is in [0.0, 1.0]. */
    external fun setParallelLinesFill(enabled: Boolean, alpha: Float)

    /**
     * Controls visibility of the dotted parallel guide lines (the guide rails).
     * When fill is enabled, the dotted lines are typically hidden to avoid visual clutter.
     */
    external fun setParallelLinesVisible(visible: Boolean)

    /** Update pocket geometry parameters (radius 5–200, nsShift 0–300). */
    external fun setPocketParams(radius: Int, nsShift: Int)

    /**
     * Sets the pool table bounds separately from the processing ROI.
     * These bounds define where reflection walls and pockets are rendered.
     * All coordinates are in screen pixels (landscape space, same as ROI coordinates).
     */
    external fun setPoolTableBounds(x1: Int, y1: Int, x2: Int, y2: Int)

    /**
     * Switch renderer between calibration mode (ROI lines + pockets only)
     * and normal aim-assist mode (AT strips + rays + HUD).
     */
    external fun setCalibrationMode(enabled: Boolean)

    /**
     * Sets the native-side screen mode gate.
     *
     * [mode] == 0 → PIPELINE (default on startup): [processFrame] and [renderOverlay]
     * are permitted; [updateIndirectAim] is suppressed.
     *
     * [mode] == 1 → INDIRECT: [updateIndirectAim] is permitted; [processFrame] and
     * [renderOverlay] are suppressed so that the pipeline does not overwrite indirect-mode
     * overlay content and does not waste CPU time processing capture frames that will
     * not be rendered.
     *
     * Any unrecognised value is silently ignored by the native layer so that future
     * additions to this set of constants do not crash an older native library.
     *
     * This is a belt-and-suspenders guard — even if a Kotlin call site forgets to check
     * the current screen mode before calling a pipeline or indirect function, the native
     * layer refuses to execute it in the wrong mode.
     */
    external fun setScreenMode(mode: Int)

    // ── Indirect mode ───────────────────────────────────────────────────────
    //
    // NOTE: This is the single-angle "live aim" API — the caller (e.g.
    // IndirectModeController) evaluates one angle at a time as the user
    // rotates the cue direction, rather than requesting ranked candidates
    // up front. There is no solveIndirectShot / getIndirectCandidate* API;
    // those were removed because no corresponding native implementation
    // exists in QeightJNI.cpp.

    /**
     * Returns the ghost/reflection circle rendering radius, in capture-resolution
     * pixels, for the most recent successful [updateIndirectAim] call.
     *
     * This mirrors the same canonical radius direct mode uses — computed
     * natively as gr = max(4f, 22.023f * captureScale) — and is also the
     * exact value used to derive the physics ball radius for that call
     * (ballRadiusCm = gr / pxPerCmAvg), so it scales with captureScale and
     * is not a fixed, rendering-only constant.
     *
     * Returns 0f if no successful [updateIndirectAim] call has been made yet.
     */
    external fun getLastIndirectGhostRadiusPx(): Float

    /**
     * Clears the cached indirect-shot path (from the most recent [updateIndirectAim]
     * call) and resets the renderer's indirect path to invisible. Safe to call even
     * if no path was ever evaluated.
     */
    external fun clearIndirectShot()

    /**
     * Evaluates a single indirect-shot angle and caches the resulting path for
     * Kotlin-side retrieval via [getIndirectPathPx].
     * Called on every rotation tick as the user sweeps the cue direction.
     *
     * [cueBallX]/[cueBallY] and [targetBallX]/[targetBallY] are capture-resolution
     * pixel coordinates. [ballRadiusPx] is the ball radius in capture-resolution
     * pixels. [thetaRad] is the launch angle to evaluate, in radians.
     *
     * [powerFraction] is the fraction of full cue power to use for this angle
     * evaluation, in the range 0.0 to 1.0. It scales the launch speed used for
     * the physics evaluation. There is no default — callers must always pass
     * an explicit value.
     *
     * The stitched path is stored unconditionally (even when the shot doesn't
     * pot) so the caller can render a cushion-only path while the user is
     * still rotating and hasn't yet aligned with the target ball.
     *
     * Returns true if the evaluated angle results in a pot; false if no pot, the
     * pipeline is uninitialized, or on any error.
     */
    external fun updateIndirectAim(
        cueBallX: Int, cueBallY: Int,
        targetBallX: Int, targetBallY: Int,
        ballRadiusPx: Int,
        thetaRad: Float,
        powerFraction: Float
    ): Boolean

    /**
     * Returns the stitched path from the most recent [updateIndirectAim] call as a
     * flat array of capture-resolution pixel coordinates [x0, y0, x1, y1, …, xn, yn].
     *
     * Returns null if no successful [updateIndirectAim] call has been made yet or
     * the cached path is empty.
     */
    external fun getIndirectPathPx(): FloatArray?

    // ── Destruction ─────────────────────────────────────────────────────────

    /** Destroy the PipelineEngine singleton and release CPU resources. */
    external fun destroyPipeline()

    /**
     * Historical no-op kept only for backward compatibility with existing call sites.
     * There is no VulkanCompute singleton and no GPU compute resources anywhere in
     * the current native code.
     */
    external fun destroyVulkan()

    /**
     * Destroys the native overlay renderer and releases its resources.
     * OverlayRenderer is a CPU/software rasterizer, so there are no GPU graphics
     * resources involved.
     */
    external fun destroyRenderer()

    /**
     * Destroys the IndirectShotSolver and IndirectPhysics native singletons and
     * releases their memory. Should be called when leaving indirect mode.
     *
     * These objects are constructed lazily on the first [updateIndirectAim] call
     * and reused for subsequent calls in the same indirect-mode session. Without
     * an explicit destroy call they would leak for the process lifetime.
     *
     * Also clears all cached indirect-shot path data ([getIndirectPathPx] will
     * return null after this call) so that stale path data from the previous
     * indirect session is not returned after the solver has been destroyed.
     *
     * A subsequent [updateIndirectAim] call will re-trigger lazy construction,
     * so this may be called and followed by a new indirect session without
     * requiring [initPipeline] to be called again.
     *
     * Safe to call even if [updateIndirectAim] was never called (i.e. the
     * singletons were never constructed).
     */
    external fun destroyIndirectSolver()
}