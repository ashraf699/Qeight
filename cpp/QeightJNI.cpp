/**
 * QeightJNI.cpp — The sole JNI bridge between Kotlin and the native C++ pipeline.
 *
 * All Java_com_ashraf_qeight_* functions are defined here and only here.
 * Each function delegates to the appropriate subsystem:
 *   PipelineEngine   : performs ball detection, strip detection, AT-strip construction,
 *                      ghost-ball correction, and shot ray-casting
 *   OverlayRenderer  : a CPU/software pixel rasterizer that writes AT-strip ray segments
 *                      and ghost circles directly into the ANativeWindow surface buffer —
 *                      no GPU or Vulkan involvement, and no HUD text is drawn
 *
 * Every function wraps its body in try/catch and propagates C++ exceptions
 * to Kotlin as Java RuntimeExceptions, so Kotlin callers see a meaningful
 * exception message rather than a silent native crash.
 *
 * NOTE: All JNI functions use jobject as the second parameter because the
 *       native methods are declared as instance methods in Kotlin (not static/companion).
 *       The JVM passes the receiver object (this) as a jobject for instance natives.
 *
 * SIMPLIFICATIONS APPLIED (renderFrame is now synchronous, no render thread):
 *
 *   1. g_renderAllowed atomic removed entirely — the use-after-free window it guarded
 *      only existed because of the asynchronous render thread. With renderFrame()
 *      synchronous and all functions holding g_lifecycleMutex for their full body,
 *      there is no window between a pointer copy and its use.
 *
 *   2. clearOverlay() removed — clearing is now performed inside renderFrame() itself.
 *
 *   3. waitForRender() removed — renderFrame() is synchronous and returns only after
 *      the frame has been fully drawn and posted to the ANativeWindow surface.
 *
 *   4. destroyRenderer() simplified — the 50 ms drain sleep is gone. The function
 *      simply acquires g_lifecycleMutex (which serialises it against any concurrent
 *      processFrame / renderOverlay call that also holds the full lock), then deletes
 *      and nulls the pointer.
 *
 *   5. Full-lock pattern applied uniformly — every JNI function (processFrame,
 *      renderOverlay, and all parameter setters) holds g_lifecycleMutex for its
 *      entire body. The brief-lock helper functions (acquirePipelinePtr et al.) are
 *      removed since they are no longer needed. Because renderFrame() is now fast
 *      and synchronous, holding the mutex during its call is safe and eliminates the
 *      former use-after-free risk without any additional atomic flags or sleeps.
 *
 * BUG FIXES RETAINED:
 *   1. copyLastResult() (not getLastResult()) — returns PipelineResult by value,
 *      avoiding unsafe const-reference lifetime across the JNI boundary.
 *
 *   2. DirectByteBuffer capacity validation — capacity <= 0 check prevents -1 from
 *      being cast to SIZE_MAX and bypassing validation.
 *
 * SANITY CHECKS ADDED:
 *   1. initPipeline() buffer size validation — warns if screenW×screenH×4 exceeds
 *      typical capture buffer threshold (1600×720×4 bytes), catching native-vs-capture
 *      dimension mixups that would cause incorrect geometric scaling.
 *
 * SCREEN MODE GATE ADDED:
 *   g_screenMode (std::atomic<ScreenMode>) acts as a native-side rendering isolation
 *   gate so that pipeline rendering and indirect-mode evaluation cannot fire during
 *   the wrong screen state, even if a Kotlin call site forgets to check.
 *
 *   - processFrame()      only executes when g_screenMode == PIPELINE.
 *   - renderOverlay()     only draws    when g_screenMode == PIPELINE.
 *   - updateIndirectAim() only evaluates when g_screenMode == INDIRECT.
 *
 *   setScreenMode(0) → PIPELINE, setScreenMode(1) → INDIRECT.
 *
 * LEAK FIX ADDED:
 *   destroyIndirectSolver() deletes g_indirectSolver and g_indirectPhysics under
 *   g_lifecycleMutex. Previously these were constructed lazily on the first
 *   updateIndirectAim() call but never destroyed, leaking memory for the process
 *   lifetime. Kotlin should call this when leaving indirect mode.
 */

#include <jni.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

#include "PipelineEngine.h"
#include "TrajectoryPhysicsEngine.h"
#include "OverlayRenderer.h"
#include "IndirectShotSolver.h"

#define LOG_TAG "QeightJNI"
// LOGE/LOGW/LOGD are gated by kLoggingEnabled (declared in OverlayRenderer.h,
// included above), the same shared switch used by OverlayRenderer.cpp.
// When disabled, calls compile away to nothing — no functional change,
// only logging output is suppressed.
#define LOGE(...) do { if (kLoggingEnabled) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGW(...) do { if (kLoggingEnabled) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__); } while (0)
#define LOGD(...) do { if (kLoggingEnabled) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// Screen mode gate
// ---------------------------------------------------------------------------

/**
 * Two-state enum representing which rendering subsystem is currently active.
 *
 * PIPELINE : Normal aim-assist mode. processFrame() and renderOverlay() are
 *            permitted. updateIndirectAim() is suppressed.
 *
 * INDIRECT : Indirect-shot evaluation mode. updateIndirectAim() is permitted.
 *            processFrame() and renderOverlay() are suppressed so that the
 *            pipeline does not overwrite indirect-mode overlay content and
 *            does not waste CPU time processing capture frames that will not
 *            be rendered.
 *
 * The enum is stored in a std::atomic so that reads in any thread are
 * sequentially consistent without requiring g_lifecycleMutex for the read
 * itself. setScreenMode() still acquires g_lifecycleMutex for consistency
 * with the full-lock pattern used everywhere else, and to ensure the mode
 * transition is serialised against any in-flight processFrame() /
 * renderOverlay() / updateIndirectAim() call that is already holding the
 * mutex.
 */
enum class ScreenMode {
    PIPELINE = 0,  ///< Normal pipeline + overlay rendering mode (default).
    INDIRECT = 1   ///< Indirect-shot evaluation mode.
};

/**
 * Native-side screen mode gate. Default is PIPELINE so that all existing
 * behaviour is preserved for callers that never call setScreenMode().
 *
 * Atomic: reads in the guard expressions of processFrame(), renderOverlay(),
 * and updateIndirectAim() need no mutex. The store in setScreenMode() is
 * also atomic and is additionally serialised by g_lifecycleMutex against
 * any concurrent JNI call that is already inside the lock.
 */
static std::atomic<ScreenMode> g_screenMode{ScreenMode::PIPELINE};

// ---------------------------------------------------------------------------
// Global singleton instances — one per process lifetime.
// ALL reads and writes (including inside processFrame, renderOverlay, and every
// parameter setter) are performed while holding g_lifecycleMutex, so no
// additional atomic flags or helper functions are needed.
// ---------------------------------------------------------------------------
static PipelineEngine*  g_pipeline = nullptr;
static OverlayRenderer* g_renderer = nullptr;

// g_indirectPhysics / g_indirectSolver are declared here (rather than further
// below, next to the rest of the indirect-mode singleton machinery) so that
// initPipeline() and destroyPipeline() can also tear them down whenever
// g_pipeline is replaced or destroyed. They hold a const PoolPhysicsEngine&
// reference obtained from g_pipeline's internals, so any g_pipeline
// replacement/destruction must invalidate them to avoid a dangling reference.
static IndirectPhysics*    g_indirectPhysics = nullptr;
static IndirectShotSolver* g_indirectSolver  = nullptr;

// Single mutex held for the full body of every JNI function that accesses
// g_pipeline or g_renderer, whether that is a lifecycle operation, frame
// processing call, render call, or parameter setter.
static std::mutex g_lifecycleMutex;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Propagates a C++ std::exception to the JVM as a java.lang.RuntimeException.
 * After calling this, the JNI function must return immediately —
 * the JVM will throw the exception when control returns to Java/Kotlin.
 */
static void throwJavaException(JNIEnv* env, const char* message)
{
    jclass rteClass = env->FindClass("java/lang/RuntimeException");
    if (rteClass != nullptr) {
        env->ThrowNew(rteClass, message);
        env->DeleteLocalRef(rteClass);
    }
}

// ============================================================================
// Pipeline lifecycle  —  FULL locks (pointer mutation)
// ============================================================================

/**
 * Initializes the image processing pipeline with capture-buffer and ROI dimensions.
 *
 * screenW / screenH are the downscaled capture-buffer dimensions (i.e. the pixel
 * dimensions of the ImageReader frame delivered to processFrame), NOT the native
 * display resolution.
 *
 * captureScale is the ratio used internally by PipelineEngine to derive
 * native-pixel-calibrated geometric constants (e.g. ball radius, pocket size)
 * from the downscaled capture-buffer coordinates. For example, if the device
 * native display is 1080 px wide and the capture buffer is 540 px wide,
 * captureScale should be 0.5f.
 *
 * Must be called after the surface is ready (initRenderer succeeds).
 * Creates the PipelineEngine singleton and pins the worker thread to big cores.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_initPipeline(
        JNIEnv* env, jobject /*thiz*/,
        jint screenW, jint screenH,
        jfloat captureScale,
        jint roiX1, jint roiY1, jint roiX2, jint roiY2)
{
    LOGD("initPipeline: screenW=%d screenH=%d captureScale=%.4f roi=(%d,%d,%d,%d)",
         screenW, screenH, (float)captureScale, roiX1, roiY1, roiX2, roiY2);

    try {
        // Sanity check: warn if the implied RGBA8888 buffer size suggests
        // the caller may have passed native display dimensions instead of
        // downscaled capture-buffer dimensions. This catches the common
        // regression where initPipeline is accidentally given the native
        // device resolution (e.g. 2340×1080) instead of the downscaled
        // ImageReader dimensions (e.g. 540×243).
        const size_t impliedBufferSize = (size_t)screenW * (size_t)screenH * 4;
        const size_t typicalMaxCaptureSize = 1600 * 720 * 4;  // 4,608,000 bytes

        if (impliedBufferSize > typicalMaxCaptureSize) {
            LOGW("initPipeline: screenW=%d screenH=%d implies RGBA8888 buffer size %zu bytes, "
                 "which exceeds typical capture buffer threshold (%zu bytes). "
                 "Did you pass native display dimensions instead of capture buffer dimensions? "
                 "Capture buffers are typically ≤1600×720 (downscaled for performance). "
                 "Pipeline will proceed but geometric scaling may be incorrect if this is a mixup.",
                 screenW, screenH, impliedBufferSize, typicalMaxCaptureSize);
        }

        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        // g_indirectSolver/g_indirectPhysics (if already constructed) hold a
        // const PoolPhysicsEngine& reference into the g_pipeline instance we're
        // about to delete below. Deleting g_pipeline without invalidating them
        // would leave that reference dangling into freed memory. Delete solver
        // first (it holds a reference into g_indirectPhysics), then physics,
        // then null both so the next updateIndirectAim() call lazily
        // reconstructs them against the new g_pipeline.
        delete g_indirectSolver;
        g_indirectSolver = nullptr;
        delete g_indirectPhysics;
        g_indirectPhysics = nullptr;

        delete g_pipeline;
        g_pipeline = new PipelineEngine(
                (int)screenW, (int)screenH,
                (float)captureScale,
                (int)roiX1,   (int)roiY1,
                (int)roiX2,   (int)roiY2);
        LOGD("initPipeline: PipelineEngine created successfully");
    } catch (const std::exception& e) {
        LOGE("initPipeline: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("initPipeline: Unknown exception");
        throwJavaException(env, "PipelineEngine: unknown fatal error");
    }
}

/**
 * Historical no-op kept only so existing Kotlin call sites continue to link.
 * There is no Vulkan-based renderer, no GPU compute subsystem, and nothing is
 * initialized by this function.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_initVulkan(
        JNIEnv* /*env*/, jobject /*thiz*/, jobject /*assetManager*/)
{
    LOGD("initVulkan: historical no-op — nothing to initialize");
}

/**
 * Constructs OverlayRenderer — a CPU/software pixel rasterizer, not a GPU
 * renderer — targeting the ANativeWindow obtained from the SurfaceView [surface].
 * No swapchain, render pass, shader, or vertex buffer is involved.
 * Must be called when the SurfaceView's surface is created.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_initRenderer(
        JNIEnv* env, jobject /*thiz*/, jobject surface, jint screenW, jint screenH)
{
    LOGD("initRenderer: Initializing software overlay renderer screenW=%d screenH=%d",
         (int)screenW, (int)screenH);
    try {
        // ANativeWindow_fromSurface calls into the JVM; do it before the lock.
        ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
        if (!window) {
            throw std::runtime_error("ANativeWindow_fromSurface returned null");
        }

        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        delete g_renderer;
        g_renderer = new OverlayRenderer(window, (int)screenW, (int)screenH);
        LOGD("initRenderer: OverlayRenderer created successfully");
    } catch (const std::exception& e) {
        LOGE("initRenderer: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("initRenderer: Unknown exception");
        throwJavaException(env, "OverlayRenderer: unknown fatal error");
    }
}

/**
 * Destroys the PipelineEngine singleton and releases all CPU-side resources.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_destroyPipeline(
        JNIEnv* /*env*/, jobject /*thiz*/)
{
    LOGD("destroyPipeline: Destroying PipelineEngine");
    std::lock_guard<std::mutex> lock(g_lifecycleMutex);

    // g_indirectSolver/g_indirectPhysics (if already constructed) hold a
    // const PoolPhysicsEngine& reference into the g_pipeline instance being
    // destroyed here. Leaving them alive would leave that reference dangling
    // into freed memory. Delete solver first (it holds a reference into
    // g_indirectPhysics), then physics, then null both.
    delete g_indirectSolver;
    g_indirectSolver = nullptr;
    delete g_indirectPhysics;
    g_indirectPhysics = nullptr;

    delete g_pipeline;
    g_pipeline = nullptr;
}

/**
 * GPU compute subsystem has been removed. Kotlin still calls destroyVulkan() and
 * it must not crash. This function returns silently — there is nothing to destroy.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_destroyVulkan(
        JNIEnv* /*env*/, jobject /*thiz*/)
{
    LOGD("destroyVulkan: GPU compute removed — nothing to destroy");
}

/**
 * Destroys the OverlayRenderer singleton and releases its software-rendering
 * resources (there are no GPU resources to release — OverlayRenderer is CPU-only).
 *
 * With renderFrame() now synchronous and every JNI function holding g_lifecycleMutex
 * for its full body, destroyRenderer() simply acquires the mutex and deletes the
 * object. Any concurrent processFrame() or renderOverlay() call will complete and
 * release the mutex before destruction begins — no atomic drain flag or sleep needed.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_destroyRenderer(
        JNIEnv* /*env*/, jobject /*thiz*/)
{
    LOGD("destroyRenderer: Destroying OverlayRenderer");
    std::lock_guard<std::mutex> lock(g_lifecycleMutex);
    delete g_renderer;
    g_renderer = nullptr;
}

// ============================================================================
// Screen mode setter  —  FULL lock
// ============================================================================

/**
 * Sets the native-side screen mode gate.
 *
 * mode == 0  →  ScreenMode::PIPELINE  (default on startup)
 * mode == 1  →  ScreenMode::INDIRECT
 *
 * Any unrecognised value is silently ignored so that future Kotlin-side enum
 * additions do not crash an older native library.
 *
 * g_lifecycleMutex is held for the full body so that the mode transition is
 * serialised against any in-flight processFrame(), renderOverlay(), or
 * updateIndirectAim() call that is already inside the lock, preventing a race
 * between a mode store and a concurrent JNI call that reads g_screenMode and
 * then acts on g_pipeline / g_renderer.
 *
 * The underlying g_screenMode atomic store is sequentially consistent on its
 * own, but holding the mutex ensures the transition is also ordered with
 * respect to all non-atomic state (g_pipeline, g_renderer, the path caches)
 * touched inside the lock by the guarded functions.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setScreenMode(
        JNIEnv* env, jobject /*thiz*/, jint mode)
{
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        switch (static_cast<int>(mode)) {
            case static_cast<int>(ScreenMode::PIPELINE):
                g_screenMode.store(ScreenMode::PIPELINE, std::memory_order_seq_cst);
                LOGD("setScreenMode: → PIPELINE");
                break;
            case static_cast<int>(ScreenMode::INDIRECT):
                g_screenMode.store(ScreenMode::INDIRECT, std::memory_order_seq_cst);
                LOGD("setScreenMode: → INDIRECT");
                break;
            default:
                LOGW("setScreenMode: unrecognised mode %d — ignored", static_cast<int>(mode));
                break;
        }
    } catch (const std::exception& e) {
        LOGE("setScreenMode: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("setScreenMode: Unknown exception");
        throwJavaException(env, "setScreenMode: unknown error");
    }
}

// ============================================================================
// Frame processing  —  FULL lock
// ============================================================================

/**
 * Processes one captured frame through the full image processing pipeline.
 *
 * directBuffer is the RGBA8888 pixel data from ImageReader — passed zero-copy.
 * Always returns JNI_TRUE on success. pipeline->processFrame() returns void;
 * renderOverlay() is called unconditionally by the Kotlin loop after this returns.
 *
 * MODE GATE: Returns JNI_FALSE immediately — without acquiring g_lifecycleMutex
 * or touching any shared state — when g_screenMode != PIPELINE. This prevents
 * the pipeline from consuming CPU time processing capture frames that will not
 * be rendered while indirect mode is active. The atomic load is sequentially
 * consistent, so the guard reliably sees any mode store that completed before
 * this call on any thread.
 *
 * g_lifecycleMutex is held for the full body once the mode gate passes.
 * Because renderFrame() is now synchronous and the Kotlin capture loop is
 * single-threaded, the mutex is never held for longer than one processFrame()
 * call (~50 ms) without yielding to the loop thread, so no cross-thread stall
 * occurs.
 *
 * DirectByteBuffer validation is performed inside the lock — GetDirectBufferAddress
 * and GetDirectBufferCapacity are direct C calls and do not block on JVM locks.
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_ashraf_qeight_QeightJNI_processFrame(
        JNIEnv* env, jobject /*thiz*/, jobject directBuffer)
{
    // ── Mode gate (pre-lock, atomic read) ────────────────────────────────────
    // Checked before acquiring g_lifecycleMutex so that the common fast-path
    // (wrong mode → return immediately) never blocks on the mutex at all.
    if (g_screenMode.load(std::memory_order_seq_cst) != ScreenMode::PIPELINE) {
        LOGD("processFrame: suppressed — g_screenMode != PIPELINE");
        return JNI_FALSE;
    }

    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        // Re-check after acquiring the lock: setScreenMode() holds the same
        // mutex, so if a mode transition completed before we locked, the
        // store is visible here. If it raced and won the lock first, we see
        // the updated mode now. Either way we do not process a frame during
        // a mode transition.
        if (g_screenMode.load(std::memory_order_seq_cst) != ScreenMode::PIPELINE) {
            LOGD("processFrame: suppressed after lock — g_screenMode != PIPELINE");
            return JNI_FALSE;
        }

        if (!g_pipeline) {
            LOGE("processFrame: Pipeline not initialized");
            return JNI_FALSE;
        }

        void* pixels = env->GetDirectBufferAddress(directBuffer);
        if (!pixels) {
            LOGE("processFrame: GetDirectBufferAddress returned null — not a direct buffer");
            return JNI_FALSE;
        }

        // BUG FIX: GetDirectBufferCapacity returns -1 for invalid/non-direct
        // buffers; casting -1 to size_t yields SIZE_MAX, bypassing validation.
        jlong capacity = env->GetDirectBufferCapacity(directBuffer);
        if (capacity <= 0) {
            LOGE("processFrame: Invalid buffer capacity %lld — not a valid direct ByteBuffer",
                 (long long)capacity);
            return JNI_FALSE;
        }

        g_pipeline->processFrame(pixels, (size_t)capacity);
        return JNI_TRUE;

    } catch (const std::exception& e) {
        LOGE("processFrame: Exception: %s", e.what());
        throwJavaException(env, e.what());
        return JNI_FALSE;
    } catch (...) {
        LOGE("processFrame: Unknown exception");
        throwJavaException(env, "processFrame: unknown error");
        return JNI_FALSE;
    }
}

/**
 * Renders the current AT strips, ghost circles, and reflection rays directly into
 * the ANativeWindow surface buffer via the CPU/software OverlayRenderer.
 * Synchronous — returns only after the frame has been fully drawn and posted to
 * the ANativeWindow surface.
 *
 * copyLastResult() copies m_result under m_resultMutex (internally) and returns
 * a snapshot by value. renderFrame() then draws and presents that snapshot
 * synchronously before returning.
 *
 * MODE GATE: After the existing null checks, returns immediately without drawing
 * when g_screenMode != PIPELINE. This is belt-and-suspenders isolation — even if
 * a Kotlin call site forgets to gate on the current screen mode, the native
 * renderer refuses to draw pipeline content while indirect mode is flagged active.
 * The check is performed inside the lock so it is ordered with respect to any
 * concurrent setScreenMode() call.
 *
 * g_lifecycleMutex is held for the full body, serialising this call against
 * destroyRenderer() and eliminating the former use-after-free race.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_renderOverlay(
        JNIEnv* env, jobject /*thiz*/)
{
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        // Existing null checks — unchanged.
        if (!g_pipeline || !g_renderer) {
            LOGE("renderOverlay: Pipeline or Renderer not initialized");
            return;
        }

        // ── Mode gate (inside lock) ───────────────────────────────────────────
        // Positioned after the null checks per spec. Inside the lock so the
        // mode read is ordered with respect to setScreenMode() which also holds
        // g_lifecycleMutex, ensuring we see the latest committed mode value.
        if (g_screenMode.load(std::memory_order_seq_cst) != ScreenMode::PIPELINE) {
            LOGD("renderOverlay: suppressed — g_screenMode != PIPELINE");
            return;
        }

        // copyLastResult() acquires m_resultMutex internally, copies m_result
        // by value, and returns the snapshot — safe to call under g_lifecycleMutex.
        PipelineResult result = g_pipeline->copyLastResult();
        g_renderer->renderFrame(result);

    } catch (const std::exception& e) {
        LOGE("renderOverlay: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("renderOverlay: Unknown exception");
        throwJavaException(env, "renderOverlay: unknown error");
    }
}

// ============================================================================
// Parameter setters  —  FULL lock
//
// Each setter holds g_lifecycleMutex for its entire body. Because the setters
// are called only from the Kotlin main thread (panel interactions) and the
// capture loop is single-threaded, contention is negligible — setters execute
// in microseconds and are never blocked behind a long processFrame() call from
// a separate thread.
// ============================================================================

/**
 * Updates the ROI bounding box after calibration is saved.
 * The pipeline adjusts its crop rectangle immediately — no restart needed.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setRoi(
        JNIEnv* env, jobject /*thiz*/,
        jint x1, jint y1, jint x2, jint y2)
{
    LOGD("setRoi: (%d,%d,%d,%d)", x1, y1, x2, y2);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) { LOGE("setRoi: Pipeline not initialized"); return; }
        g_pipeline->setRoi((int)x1, (int)y1, (int)x2, (int)y2);
    } catch (const std::exception& e) {
        LOGE("setRoi: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the number of cushion-shot reflections for the CBC (cue ball cut) ray.
 * Range 0–8. Applied immediately to the next processFrame call.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setCbcReflections(
        JNIEnv* env, jobject /*thiz*/, jint count)
{
    LOGD("setCbcReflections: count=%d", count);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setCbcReflections((int)count);
    } catch (const std::exception& e) {
        LOGE("setCbcReflections: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the number of cushion-shot reflections for the TGT (target ball) ray.
 * Range 0–8. Applied immediately to the next processFrame call.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setTgtReflections(
        JNIEnv* env, jobject /*thiz*/, jint count)
{
    LOGD("setTgtReflections: count=%d", count);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setTgtReflections((int)count);
    } catch (const std::exception& e) {
        LOGE("setTgtReflections: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Enables or disables cushion shots mode.
 * When disabled (false), maxr_cbc and maxr_tgt are forced to 0 regardless
 * of the spinner values, suppressing all reflection extensions.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setCushionShots(
        JNIEnv* env, jobject /*thiz*/, jboolean enabled)
{
    LOGD("setCushionShots: enabled=%d", (int)enabled);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setCushionShots(enabled == JNI_TRUE);
    } catch (const std::exception& e) {
        LOGE("setCushionShots: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the overlay line/ghost circle color as RGBA byte components (0–255 each).
 * Stored on OverlayRenderer and read directly by its CPU rasterizer on the
 * next renderOverlay call — there is no shader or GPU uniform involved.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setOverlayColor(
        JNIEnv* env, jobject /*thiz*/,
        jint r, jint g, jint b, jint a)
{
    LOGD("setOverlayColor: r=%d g=%d b=%d a=%d", r, g, b, a);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_renderer) return;
        g_renderer->setOverlayColor(
                (float)r / 255.f,
                (float)g / 255.f,
                (float)b / 255.f,
                (float)a / 255.f);
    } catch (const std::exception& e) {
        LOGE("setOverlayColor: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the overlay line thickness in pixels (range 1–8).
 * Stored on OverlayRenderer and used directly by its CPU rasterizer on the
 * next renderOverlay call — there is no GPU pipeline or uniform involved.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setLineThickness(
        JNIEnv* env, jobject /*thiz*/, jint thickness)
{
    LOGD("setLineThickness: thickness=%d", thickness);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_renderer) return;
        g_renderer->setLineThickness((float)std::clamp((int)thickness, 1, 8));
    } catch (const std::exception& e) {
        LOGE("setLineThickness: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Switches the renderer between calibration mode and normal aim-assist mode.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setCalibrationMode(
        JNIEnv* env, jobject /*thiz*/, jboolean enabled)
{
    LOGD("setCalibrationMode: enabled=%d", (int)enabled);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_renderer) return;
        g_renderer->setCalibrationMode(enabled == JNI_TRUE);
    } catch (const std::exception& e) {
        LOGE("setCalibrationMode: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Toggles the translucent fill band drawn between the dotted guide rails.
 * [alpha] is in [0.0, 1.0] and is clamped inside OverlayRenderer.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setParallelLinesFill(
        JNIEnv* env, jobject /*thiz*/, jboolean enabled, jfloat alpha)
{
    LOGD("setParallelLinesFill: enabled=%d alpha=%.3f", (int)enabled, (float)alpha);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_renderer) return;
        g_renderer->setParallelLinesFill(enabled == JNI_TRUE, static_cast<float>(alpha));
    } catch (const std::exception& e) {
        LOGE("setParallelLinesFill: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Controls visibility of the dotted parallel guide lines (the guide rails).
 * When the translucent fill band is enabled, the dotted lines are typically
 * hidden to reduce visual clutter.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setParallelLinesVisible(
        JNIEnv* env, jobject /*thiz*/, jboolean visible)
{
    LOGD("setParallelLinesVisible: visible=%d", (int)visible);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_renderer) {
            LOGW("setParallelLinesVisible: renderer not initialized");
            return;
        }
        g_renderer->setParallelLinesVisible(visible == JNI_TRUE);
    } catch (const std::exception& e) {
        LOGE("setParallelLinesVisible: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("setParallelLinesVisible: Unknown exception");
        throwJavaException(env, "setParallelLinesVisible: unknown error");
    }
}

/**
 * Updates the pocket geometry parameters used by both the pipeline and renderer.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setPocketParams(
        JNIEnv* env, jobject /*thiz*/, jint radius, jint nsShift)
{
    LOGD("setPocketParams: radius=%d nsShift=%d", radius, nsShift);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setPocketParams((int)radius, (int)nsShift);
    } catch (const std::exception& e) {
        LOGE("setPocketParams: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the trajectory power as a percentage (0–100).
 * Applied immediately to the next processFrame call.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setTrajectoryPower(
        JNIEnv* env, jobject /*thiz*/, jint percent)
{
    LOGD("setTrajectoryPower: percent=%d", percent);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setTrajectoryPower((int)percent);
    } catch (const std::exception& e) {
        LOGE("setTrajectoryPower: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the cue force stat value.
 * Applied immediately to the next processFrame call.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setCueForce(
        JNIEnv* env, jobject /*thiz*/, jint stat)
{
    LOGD("setCueForce: stat=%d", stat);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setCueForce((int)stat);
    } catch (const std::exception& e) {
        LOGE("setCueForce: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets the cue spin stat value.
 * Applied immediately to the next processFrame call.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setCueSpin(
        JNIEnv* env, jobject /*thiz*/, jint stat)
{
    LOGD("setCueSpin: stat=%d", stat);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setCueSpin((int)stat);
    } catch (const std::exception& e) {
        LOGE("setCueSpin: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Sets pool table bounds separately from the processing ROI.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_setPoolTableBounds(
        JNIEnv* env, jobject /*thiz*/,
        jint x1, jint y1, jint x2, jint y2)
{
    LOGD("setPoolTableBounds: (%d,%d,%d,%d)", x1, y1, x2, y2);
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        if (!g_pipeline) return;
        g_pipeline->setPoolTableBounds((int)x1, (int)y1, (int)x2, (int)y2);
    } catch (const std::exception& e) {
        LOGE("setPoolTableBounds: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

// ============================================================================
// Indirect mode  —  FULL lock
//
// This section implements a live single-angle aim evaluator for indirect
// (cushion-first) shots.  updateIndirectAim() is called on every rotation
// tick from Kotlin as the user sweeps the cue angle; it evaluates exactly
// one angle per call and stores the resulting path data in single-path
// caches for Kotlin-side retrieval.
//
// There is no multi-candidate solve, no pocket-pinning logic, and no
// one-shot batch solve.  The caller supplies the angle (thetaRad) and
// retrieves the evaluated path via getIndirectPathPx().
// clearIndirectShot() resets the caches.
//
// All functions hold g_lifecycleMutex for their entire body, exactly as
// every other JNI function in this file does.  The IndirectShotSolver is
// lazily constructed on the first updateIndirectAim() call and cached for
// the process lifetime — same singleton pattern as g_pipeline / g_renderer.
// destroyIndirectSolver() must be called when leaving indirect mode to
// release the memory (no longer leaked for the process lifetime).
// ============================================================================

// Lazily constructed, destroyed by destroyIndirectSolver().
// Both are constructed together on the first updateIndirectAim() call and
// reused for all subsequent calls in the same indirect-mode session.
// Guarded by g_lifecycleMutex alongside g_pipeline and g_renderer.
// (Declared near g_pipeline/g_renderer above so initPipeline()/destroyPipeline()
// can also tear them down when g_pipeline is replaced or destroyed.)

// ── Single-path pixel-space cache ────────────────────────────────────────────
// g_lastIndirectPathPx        : flat [x,y] pairs for the stitched path
//                               (cue-ball launch → cushion contacts → pocket)
// g_lastIndirectTouchesTarget : true if the evaluated angle causes the cue ball
//                               to make contact with the target ball
// g_lastIndirectPots          : true if the evaluated angle results in a pot
// g_lastIndirectPocketHitPx   : pixel-space pocket hit position (valid when pots)
// g_lastIndirectGhostRadiusPx : ghost-circle radius, in capture-resolution
//                               pixels, set to the same canonical pixel
//                               radius gr = max(4.f, 22.023f *
//                               captureScale) that PipelineEngine computes
//                               at construction time for direct mode —
//                               used verbatim here for rendering and, via
//                               gr / pxPerCmAvg, to derive the physics ball
//                               radius in cm. One source of truth, no
//                               independent indirect-mode constants.
//
// All are guarded by g_lifecycleMutex.
static std::vector<std::pair<float,float>> g_lastIndirectPathPx;
static bool  g_lastIndirectTouchesTarget = false;
static bool  g_lastIndirectPots          = false;
static std::pair<float,float> g_lastIndirectPocketHitPx{0.f, 0.f};
static float g_lastIndirectGhostRadiusPx = 0.f;

// Legacy fixed radius value, in capture-resolution pixels, formerly used
// unconditionally for cushion-bounce and cue/target contact ghost circles
// in indirect-shot mode. No longer used to set g_lastIndirectGhostRadiusPx —
// that value is now set to gr, the same canonical pixel radius
// (max(4.f, 22.023f * captureScale)) that PipelineEngine computes at
// construction time and uses for both direct-mode physics and rendering.
// Left declared in case anything else references it.
static constexpr float INDIRECT_GHOST_RADIUS_PX = 22.023f;

/**
 * Evaluates a single indirect-shot angle and caches the resulting path for
 * Kotlin-side retrieval.  Called on every rotation tick as the user sweeps
 * the cue direction.
 *
 * cueBallX / cueBallY       : capture-resolution pixel coordinates of the cue ball.
 * targetBallX / targetBallY : capture-resolution pixel coordinates of the target ball.
 * ballRadiusPx              : ball radius in capture-resolution pixels.
 * thetaRad                  : the launch angle to evaluate, in radians.
 * powerFraction             : fraction of full cue power to simulate for this
 *                              evaluation, in [0.0, 1.0]. Clamped internally
 *                              since it originates from a UI drag gesture.
 *                              Scales launchSpeedCmS (derived from full
 *                              cueStats.maxPower) before the request is built.
 *
 * MODE GATE: Returns JNI_FALSE immediately — without acquiring g_lifecycleMutex
 * or touching any shared state — when g_screenMode != INDIRECT. This prevents
 * indirect evaluation from firing during normal pipeline operation, even if a
 * Kotlin call site forgets to gate on the current screen mode. The atomic load
 * is sequentially consistent, so the guard reliably sees any mode store that
 * completed before this call on any thread. A second check is performed after
 * acquiring the lock to handle the unlikely race where setScreenMode() won the
 * lock between the pre-lock check and the lock acquisition.
 *
 * The stitched path is stored unconditionally (even when touchesTarget is
 * false) so that the caller has a cushion-only path available while the user
 * is rotating and has not yet aligned with the target ball.
 *
 * Returns JNI_TRUE  if the evaluated angle results in a pot.
 * Returns JNI_FALSE if no pot (contact without potting does NOT return true),
 *                   if the pipeline is uninitialised, if the mode gate fires,
 *                   or on any error.
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_ashraf_qeight_QeightJNI_updateIndirectAim(
        JNIEnv* env, jobject /*thiz*/,
        jint cueBallX,    jint cueBallY,
        jint targetBallX, jint targetBallY,
        jint ballRadiusPx,
        jfloat thetaRad,
        jfloat powerFraction)
{
    LOGD("updateIndirectAim: cue=(%d,%d) tgt=(%d,%d) ballRadiusPx=%d thetaRad=%.4f powerFraction=%.4f",
         cueBallX, cueBallY, targetBallX, targetBallY, ballRadiusPx, (float)thetaRad, (float)powerFraction);

    // ── Mode gate (pre-lock, atomic read) ────────────────────────────────────
    // Checked before acquiring g_lifecycleMutex so that the common fast-path
    // (wrong mode → return immediately) never blocks on the mutex at all.
    if (g_screenMode.load(std::memory_order_seq_cst) != ScreenMode::INDIRECT) {
        LOGD("updateIndirectAim: suppressed — g_screenMode != INDIRECT");
        return JNI_FALSE;
    }

    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        // Re-check after acquiring the lock: setScreenMode() holds the same
        // mutex, so if a mode transition completed before we locked, the
        // store is visible here.
        if (g_screenMode.load(std::memory_order_seq_cst) != ScreenMode::INDIRECT) {
            LOGD("updateIndirectAim: suppressed after lock — g_screenMode != INDIRECT");
            return JNI_FALSE;
        }

        // Clamp defensively — powerFraction originates from a UI drag gesture
        // and should not be trusted as pre-clamped.
        const float clampedPowerFraction =
                std::min(1.0f, std::max(0.0f, static_cast<float>(powerFraction)));

        // ── Reset single-path caches before every evaluation ─────────────────
        g_lastIndirectPathPx.clear();
        g_lastIndirectTouchesTarget = false;
        g_lastIndirectPots          = false;
        g_lastIndirectPocketHitPx   = {0.f, 0.f};
        g_lastIndirectGhostRadiusPx = 0.f;

        if (!g_pipeline) {
            LOGE("updateIndirectAim: Pipeline not initialized");
            return JNI_FALSE;
        }

        // ── 1. Derive pixel↔cm conversion from pool table bounds ──────────────
        //
        // Replicates the exact math used by TrajectoryPhysicsEngine::
        // computeTrajectories.  Bounds are exposed via
        // PipelineEngine::getPoolTableBounds().
        int poolX1, poolY1, poolX2, poolY2;
        g_pipeline->getPoolTableBounds(poolX1, poolY1, poolX2, poolY2);

        const float tableWidthPx  = static_cast<float>(poolX2 - poolX1);
        const float tableHeightPx = static_cast<float>(poolY2 - poolY1);

        if (tableWidthPx <= 0.f || tableHeightPx <= 0.f) {
            LOGE("updateIndirectAim: pool table bounds are degenerate — evaluate aborted");
            return JNI_FALSE;
        }

        // Physical table dimensions in cm — single source of truth from PoolPhysicsEngine.
        // The pool table is rendered in landscape; the long axis maps to pixel width.
        const float pxPerCmX   = tableWidthPx  / PoolPhysicsEngine::TABLE_LENGTH_CM;
        const float pxPerCmY   = tableHeightPx / PoolPhysicsEngine::TABLE_WIDTH_CM;

        // Single source of truth shared by physics and rendering, matching
        // TrajectoryPhysicsEngine::computeTrajectories's derivation of the
        // direct-mode ghost radius.
        const float pxPerCmAvg  = std::sqrt(pxPerCmX * pxPerCmY);

        LOGD("updateIndirectAim: poolBounds=(%d,%d,%d,%d) pxPerCmX=%.6f pxPerCmY=%.6f",
             poolX1, poolY1, poolX2, poolY2, pxPerCmX, pxPerCmY);

        // Center of the pool table in pixel space (origin for cm conversion).
        const float centerX = static_cast<float>(poolX1 + poolX2) * 0.5f;
        const float centerY = static_cast<float>(poolY1 + poolY2) * 0.5f;

        // Lambda: pixel → cm (centered on table origin).
        // Y is negated because Android pixel space has Y increasing downward,
        // while PoolPhysicsEngine's convention is TOP = positive Y.
        auto pxToCm = [&](float px, float py) -> cv::Point2f {
            return cv::Point2f(
                    (px - centerX) / pxPerCmX,
                    -(py - centerY) / pxPerCmY);
        };

        // Lambda: cm → pixel (inverse of pxToCm).
        auto cmToPx = [&](float cx, float cy) -> cv::Point2f {
            return cv::Point2f(
                    cx * pxPerCmX + centerX,
                    -cy * pxPerCmY + centerY);
        };

        // ── 2. Convert ball positions to cm; ball radius is sourced from the
        //      same canonical pixel radius direct mode uses ─────────────────
        //
        // PipelineEngine's constructor computes one canonical pixel radius,
        // gr = max(4.f, 22.023f * captureScale), and uses it for both
        // physics (passed as ballRadiusPx into
        // TrajectoryPhysicsEngine::computeTrajectories, which converts to cm
        // via ballRadiusCm = ballRadiusPx / pxPerCmAvg) and rendering (stored
        // verbatim as PipelineResult::ghostRadius). Indirect mode replicates
        // that exact derivation here instead of using independent constants,
        // so its physics radius and its rendered ghost radius are tied to the
        // same single source of truth as direct mode.
        const float gr = std::max(4.f, 22.023f * g_pipeline->getCaptureScale());

        const cv::Point2f cueBallCm    = pxToCm(static_cast<float>(cueBallX),
                                                static_cast<float>(cueBallY));
        const cv::Point2f targetBallCm = pxToCm(static_cast<float>(targetBallX),
                                                static_cast<float>(targetBallY));
        // Matches TrajectoryPhysicsEngine::computeTrajectories's
        // ballRadiusCm = ballRadiusPx / pxPerCmAvg line exactly, including
        // the same floor clamp, for parity with direct mode.
        const float ballRadiusCm = std::max(0.5f, gr / pxPerCmAvg);
        (void)ballRadiusPx;  // no longer used for physics — direct mode doesn't
        // use a per-frame detected radius either; it uses
        // the constructor-time gr, same as here.

        // ── 3. Compute launch speed at full power ──────────────────────────────
        //
        // applyCueStats(cueForceStat, cueSpinStat).maxPower is used at full
        // power — this is unrelated to any real-world force percentage; it is
        // only used for the physics simulation.
        int cueForceStat, cueSpinStat;
        g_pipeline->getCueStats(cueForceStat, cueSpinStat);

        const CueStatResult cueStats =
                g_pipeline->getTrajectoryEngine().getPhysicsEngine()
                        .applyCueStats(cueForceStat, cueSpinStat);
        const float launchSpeedCmS = cueStats.maxPower * clampedPowerFraction;  // scaled by requested power fraction

        LOGD("updateIndirectAim: cueBallCm=(%.4f,%.4f) targetBallCm=(%.4f,%.4f) "
             "ballRadiusCm=%.4f launchSpeedCmS=%.4f",
             cueBallCm.x, cueBallCm.y,
             targetBallCm.x, targetBallCm.y,
             ballRadiusCm, launchSpeedCmS);

        // ── 4. Lazily construct / reuse IndirectPhysics + IndirectShotSolver ───
        if (!g_indirectPhysics || !g_indirectSolver) {
            const PoolPhysicsEngine& physicsEngine =
                    g_pipeline->getTrajectoryEngine().getPhysicsEngine();
            delete g_indirectSolver;
            g_indirectSolver = nullptr;
            delete g_indirectPhysics;
            g_indirectPhysics = new IndirectPhysics(physicsEngine);
            g_indirectSolver  = new IndirectShotSolver(physicsEngine, *g_indirectPhysics);
            LOGD("updateIndirectAim: IndirectPhysics + IndirectShotSolver constructed");
        }

        // ── 5. Build the request and evaluate the supplied angle ───────────────
        IndirectShotRequest req;
        req.cueBallPos     = cueBallCm;
        req.targetBallPos  = targetBallCm;
        req.ballRadiusCm   = ballRadiusCm;
        req.launchSpeedCmS = launchSpeedCmS;
        // No pocket field and no cue-stat fields on IndirectShotRequest.

        // Mirror the Y-axis convention flip already applied to the ball
        // positions via pxToCm (screen Y increases downward, solver cm-space
        // Y increases upward), so the launch angle matches the coordinate
        // convention of the positions it's paired with.
        const float thetaRadCm = -static_cast<float>(thetaRad);

        const IndirectShotEvaluation eval =
                g_indirectSolver->evaluateAngle(req, thetaRadCm);

        LOGD("updateIndirectAim: touchesTarget=%d pots=%d stitchedPathSize=%zu",
             (int)eval.touchesTarget, (int)eval.pots,
             eval.stitchedPath.size());

        // ── 6. Convert path to pixel space and cache — unconditionally ─────────
        //
        // Store even when touchesTarget is false so the caller has a
        // cushion-only path available while the user rotates and has not
        // yet aligned with the target ball.
        //
        // Defensive validation: truncate the path at the last in-bounds point
        // if a point ever falls outside the physical table (with a ball-radius
        // tolerance, since ball centers near the cushion legitimately sit within
        // one radius of the exact edge). This guards against solver edge cases
        // producing out-of-table points that would otherwise be returned to
        // the caller.
        const float halfLengthCm = PoolPhysicsEngine::TABLE_LENGTH_CM * 0.5f + ballRadiusCm;
        const float halfWidthCm  = PoolPhysicsEngine::TABLE_WIDTH_CM  * 0.5f + ballRadiusCm;
        auto isInBoundsCm = [&](const cv::Point2f& posCm) -> bool {
            return posCm.x >= -halfLengthCm && posCm.x <= halfLengthCm &&
                   posCm.y >= -halfWidthCm  && posCm.y <= halfWidthCm;
        };

        g_lastIndirectPathPx.reserve(eval.stitchedPath.size());
        for (const PathPoint& pt : eval.stitchedPath) {
            if (!isInBoundsCm(pt.position)) {
                break;
            }
            const cv::Point2f px = cmToPx(pt.position.x, pt.position.y);
            g_lastIndirectPathPx.emplace_back(px.x, px.y);
        }
        if (g_lastIndirectPathPx.size() < 2) {
            g_lastIndirectPathPx.clear();
        }

        // ── 7. Store scalar results ────────────────────────────────────────────
        g_lastIndirectTouchesTarget = eval.touchesTarget;
        g_lastIndirectPots          = eval.pots;

        if (eval.pots) {
            const cv::Point2f pocketPx = cmToPx(eval.pocketHit.x, eval.pocketHit.y);
            g_lastIndirectPocketHitPx = {pocketPx.x, pocketPx.y};
        }

        // The same canonical pixel radius (gr) used for physics above,
        // untouched — matching direct mode, which stores gr verbatim as
        // PipelineResult::ghostRadius with no reconstruction.
        g_lastIndirectGhostRadiusPx = gr;

        // Return true only when the shot results in a pot.
        // Contact without potting must NOT return true (Kotlin uses this to
        // decide whether to vibrate).
        return eval.pots ? JNI_TRUE : JNI_FALSE;

    } catch (const std::exception& e) {
        LOGE("updateIndirectAim: Exception: %s", e.what());
        throwJavaException(env, e.what());
        return JNI_FALSE;
    } catch (...) {
        LOGE("updateIndirectAim: Unknown exception");
        throwJavaException(env, "updateIndirectAim: unknown error");
        return JNI_FALSE;
    }
}

/**
 * Returns the stitched indirect-shot path from the most recent
 * updateIndirectAim() call as a flat jfloatArray [x0,y0,x1,y1,...,xn,yn]
 * in capture-resolution pixel coordinates.
 *
 * The path is stored unconditionally — it is non-null even when
 * touchesTarget is false, so the caller has a cushion-only path available
 * while the user rotates.
 *
 * Returns null if no evaluation has been run or the path is empty.
 * g_lifecycleMutex is held for the full body.
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ashraf_qeight_QeightJNI_getIndirectPathPx(
        JNIEnv* env, jobject /*thiz*/)
{
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        if (g_lastIndirectPathPx.empty()) return nullptr;

        const jsize floatCount = static_cast<jsize>(g_lastIndirectPathPx.size() * 2);
        jfloatArray result = env->NewFloatArray(floatCount);
        if (!result) {
            LOGE("getIndirectPathPx: NewFloatArray(%d) returned null — out of memory",
                 floatCount);
            return nullptr;
        }

        std::vector<jfloat> flat;
        flat.reserve(static_cast<size_t>(floatCount));
        for (const auto& pt : g_lastIndirectPathPx) {
            flat.push_back(pt.first);
            flat.push_back(pt.second);
        }
        env->SetFloatArrayRegion(result, 0, floatCount, flat.data());
        return result;

    } catch (const std::exception& e) {
        LOGE("getIndirectPathPx: Exception: %s", e.what());
        throwJavaException(env, e.what());
        return nullptr;
    } catch (...) {
        LOGE("getIndirectPathPx: Unknown exception");
        throwJavaException(env, "getIndirectPathPx: unknown error");
        return nullptr;
    }
}

/**
 * Returns the ghost-circle radius, in capture-resolution pixels, associated
 * with the most recent updateIndirectAim() call.
 *
 * This value equals gr = max(4.f, 22.023f * captureScale), the same
 * canonical pixel radius PipelineEngine computes once at construction time
 * and uses for both direct-mode physics and rendering (PipelineResult::
 * ghostRadius). Indirect mode now sources its radius from that identical
 * value rather than an independent constant.
 *
 * Returns 0.f if no evaluation has been run or the cache has been cleared.
 * g_lifecycleMutex is held for the full body.
 */
extern "C" JNIEXPORT jfloat JNICALL
Java_com_ashraf_qeight_QeightJNI_getLastIndirectGhostRadiusPx(
        JNIEnv* /*env*/, jobject /*thiz*/)
{
    std::lock_guard<std::mutex> lock(g_lifecycleMutex);
    return g_lastIndirectGhostRadiusPx;
}

/**
 * Clears all cached indirect-shot path data. Safe to call even if no
 * evaluation has ever been run.
 *
 * g_lifecycleMutex is held for the full body.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_clearIndirectShot(
        JNIEnv* env, jobject /*thiz*/)
{
    LOGD("clearIndirectShot");
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);
        g_lastIndirectPathPx.clear();
        g_lastIndirectTouchesTarget = false;
        g_lastIndirectPots          = false;
        g_lastIndirectPocketHitPx   = {0.f, 0.f};
        g_lastIndirectGhostRadiusPx = 0.f;
    } catch (const std::exception& e) {
        LOGE("clearIndirectShot: Exception: %s", e.what());
        throwJavaException(env, e.what());
    }
}

/**
 * Destroys the IndirectShotSolver and IndirectPhysics singletons and releases
 * their memory. Kotlin should call this when leaving indirect mode.
 *
 * Previously these objects were constructed lazily on the first
 * updateIndirectAim() call but never destroyed, leaking memory for the
 * process lifetime. This function closes that leak.
 *
 * Destruction order: g_indirectSolver is deleted first because it holds a
 * reference to *g_indirectPhysics (passed at construction); deleting
 * g_indirectPhysics first would leave g_indirectSolver holding a dangling
 * reference that its destructor might access. After deletion both pointers
 * are nulled so that a subsequent updateIndirectAim() call correctly
 * re-triggers lazy construction.
 *
 * The path caches are also cleared so that stale path data from the
 * previous indirect session is not returned by getIndirectPathPx() after
 * the solver has been destroyed.
 *
 * g_lifecycleMutex is held for the full body, serialising this call against
 * any concurrent updateIndirectAim() / getIndirectPathPx() / clearIndirectShot()
 * call that is already inside the lock.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_ashraf_qeight_QeightJNI_destroyIndirectSolver(
        JNIEnv* env, jobject /*thiz*/)
{
    LOGD("destroyIndirectSolver: Destroying IndirectShotSolver and IndirectPhysics");
    try {
        std::lock_guard<std::mutex> lock(g_lifecycleMutex);

        // Delete solver before physics: solver holds a reference to *physics.
        delete g_indirectSolver;
        g_indirectSolver = nullptr;

        delete g_indirectPhysics;
        g_indirectPhysics = nullptr;

        // Clear path caches so stale data is not returned after re-entry.
        g_lastIndirectPathPx.clear();
        g_lastIndirectTouchesTarget = false;
        g_lastIndirectPots          = false;
        g_lastIndirectPocketHitPx   = {0.f, 0.f};
        g_lastIndirectGhostRadiusPx = 0.f;

        LOGD("destroyIndirectSolver: IndirectShotSolver and IndirectPhysics destroyed");
    } catch (const std::exception& e) {
        LOGE("destroyIndirectSolver: Exception: %s", e.what());
        throwJavaException(env, e.what());
    } catch (...) {
        LOGE("destroyIndirectSolver: Unknown exception");
        throwJavaException(env, "destroyIndirectSolver: unknown error");
    }
}