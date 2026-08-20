#include "OverlayRenderer.h"

#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#define LOG_TAG "OverlayRenderer"
// LOGD/LOGE are gated by kLoggingEnabled (declared in OverlayRenderer.h).
// When disabled, calls compile away to nothing — no functional change,
// only logging output is suppressed.
#define LOGD(...) do { if (kLoggingEnabled) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGE(...) do { if (kLoggingEnabled) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Geometry math helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline cv::Point2f pt_scale(cv::Point2f p, float s) noexcept
{ return {p.x * s, p.y * s}; }

static inline cv::Point2f pt_add(cv::Point2f a, cv::Point2f b) noexcept
{ return {a.x + b.x, a.y + b.y}; }

static inline cv::Point2f pt_sub(cv::Point2f a, cv::Point2f b) noexcept
{ return {a.x - b.x, a.y - b.y}; }

static inline float pt_dot(cv::Point2f a, cv::Point2f b) noexcept
{ return a.x * b.x + a.y * b.y; }

static inline float pt_mag(cv::Point2f a) noexcept
{ return std::hypot(a.x, a.y); }

static inline cv::Point2f pt_norm(cv::Point2f a) noexcept {
    float m = pt_mag(a);
    return m > 1e-9f ? pt_scale(a, 1.f / m) : cv::Point2f{0.f, 0.f};
}

static inline cv::Point2f pt_perp(cv::Point2f a) noexcept
{ return {-a.y, a.x}; }

static constexpr float PI_F = 3.14159265f;

// ─────────────────────────────────────────────────────────────────────────────
// scalarToFloats
// cv::Scalar convention: s[0]=B, s[1]=G, s[2]=R, s[3]=A
// ─────────────────────────────────────────────────────────────────────────────
static void scalarToFloats(const cv::Scalar& s,
                           float& r, float& g, float& b, float& a) noexcept {
    b = static_cast<float>(s[0]) / 255.f;
    g = static_cast<float>(s[1]) / 255.f;
    r = static_cast<float>(s[2]) / 255.f;
    a = (s[3] > 0.0) ? static_cast<float>(s[3]) / 255.f : 1.f;
}

// ═════════════════════════════════════════════════════════════════════════════
// Construction / destruction
// ═════════════════════════════════════════════════════════════════════════════
OverlayRenderer::OverlayRenderer(ANativeWindow* window,
                                 int pipelineScreenW,
                                 int pipelineScreenH)
        : m_window(window),
          m_pipelineScreenW(pipelineScreenW),
          m_pipelineScreenH(pipelineScreenH)
{
    if (!window)
        throw std::runtime_error("OverlayRenderer: ANativeWindow is null");

    // WINDOW_FORMAT_RGBA_8888: R,G,B,A bytes in memory order.
    // On little-endian ARM, a uint32_t read of those 4 bytes is
    // (A<<24)|(B<<16)|(G<<8)|R.
    // 0,0 means keep existing surface dimensions.
    ANativeWindow_setBuffersGeometry(window, 0, 0, WINDOW_FORMAT_RGBA_8888);

    m_surfaceW = ANativeWindow_getWidth(window);
    m_surfaceH = ANativeWindow_getHeight(window);

    LOGD("OverlayRenderer: surface %dx%d  pipeline %dx%d",
         m_surfaceW, m_surfaceH, m_pipelineScreenW, m_pipelineScreenH);

    // Portrait surface + landscape pipeline → 90° CW rotation.
    m_rotMode = (m_surfaceW < m_surfaceH &&
                 m_pipelineScreenW > m_pipelineScreenH) ? 1 : 0;

    // Pre-clear ALL buffer slots (double/triple buffered) so neither slot
    // ever contains stale pixel data from a previous renderer instance.
    // Without this, alternating buffer slots show old content on the first
    // few frames — visible as flicker for non-white colors because white
    // 0xFFFFFFFF is immune to stale buffer content whereas colored pixels
    // with partial alpha are not.
    for (int i = 0; i < 3; ++i) {
        ANativeWindow_Buffer buf{};
        if (ANativeWindow_lock(window, &buf, nullptr) == 0) {
            const size_t count =
                    static_cast<size_t>(buf.stride) * static_cast<size_t>(buf.height);
            std::fill(static_cast<uint32_t*>(buf.bits),
                      static_cast<uint32_t*>(buf.bits) + count,
                      0u);
            ANativeWindow_unlockAndPost(window);
        }
    }

    LOGD("OverlayRenderer: rotMode=%d, buffer slots pre-cleared — renderer ready",
         m_rotMode);
}

OverlayRenderer::~OverlayRenderer() {
    LOGD("OverlayRenderer: destroyed");
    // ANativeWindow_fromSurface() (called once per initRenderer() in
    // QeightJNI.cpp, immediately before this object is constructed with the
    // resulting pointer) acquires a strong native reference on the window —
    // its acquire/release contract is symmetric with ANativeWindow_release().
    // Nothing else in this codebase ever released that reference: this
    // object is the sole owner of it for its lifetime, and destroyRenderer()
    // (the only caller that destroys an OverlayRenderer) is reliably invoked
    // on every surfaceDestroyed(). Previously this destructor assumed "the
    // caller manages it" and did nothing, which meant every surfaceCreated()
    // -> surfaceDestroyed() -> surfaceCreated() cycle (every rotation, every
    // screen-off/on, every renderer rebuild) permanently leaked one
    // ANativeWindow reference for the life of the process — the dead
    // Surface/BufferQueue behind it could never be fully torn down while a
    // native reference to it still existed, so old surfaces accumulated and
    // could contend with the live one. Releasing it here closes that leak.
    if (m_window) {
        ANativeWindow_release(m_window);
        m_window = nullptr;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Public parameter setters  (UI-thread safe via m_paramMtx)
// ═════════════════════════════════════════════════════════════════════════════
void OverlayRenderer::setOverlayColor(float r, float g, float b, float a) {
    std::lock_guard<std::mutex> lk(m_paramMtx);
    m_overlayColorR = r;
    m_overlayColorG = g;
    m_overlayColorB = b;
    m_overlayColorA = a;
}

void OverlayRenderer::setLineThickness(float t) {
    std::lock_guard<std::mutex> lk(m_paramMtx);
    m_lineThickness = std::clamp(t, 1.f, 8.f);
}

void OverlayRenderer::setCalibrationMode(bool e) {
    std::lock_guard<std::mutex> lk(m_paramMtx);
    m_calibrationMode = e;
}

void OverlayRenderer::setParallelLinesFill(bool enabled, float alpha) {
    std::lock_guard<std::mutex> lk(m_paramMtx);
    m_parallelFillEnabled = enabled;
    m_parallelFillAlpha   = std::clamp(alpha, 0.f, 1.f);
}

void OverlayRenderer::setParallelLinesVisible(bool visible) {
    std::lock_guard<std::mutex> lk(m_paramMtx);
    m_parallelLinesVisible = visible;
}

// ═════════════════════════════════════════════════════════════════════════════
// renderFrame — synchronous, unconditional render every call.
// lock → clear → generate → rasterize → post
// No change detection — every call produces one clean posted frame.
// ═════════════════════════════════════════════════════════════════════════════
void OverlayRenderer::renderFrame(const PipelineResult& result) {
    if (!m_window) {
        LOGE("renderFrame: m_window is null, skipping");
        return;
    }

    // Step 1: acquire surface buffer
    ANativeWindow_Buffer buf{};
    if (ANativeWindow_lock(m_window, &buf, nullptr) != 0) {
        LOGE("renderFrame: ANativeWindow_lock failed, skipping frame");
        return;
    }

    const int   w      = buf.width;
    const int   h      = buf.height;
    const int   stride = buf.stride;
    auto* const pixels = static_cast<uint32_t*>(buf.bits);

    // Step 2: clear to fully transparent (0x00000000)
    // Must clear every pixel we own — stride may be wider than width.
    if (stride == w) {
        std::fill(pixels,
                  pixels + static_cast<size_t>(stride) * static_cast<size_t>(h),
                  0u);
    } else {
        for (int y = 0; y < h; ++y) {
            std::fill(pixels + static_cast<size_t>(y) * static_cast<size_t>(stride),
                      pixels + static_cast<size_t>(y) * static_cast<size_t>(stride) + w,
                      0u);
        }
    }

    // Step 3: generate geometry
    std::vector<Seg> segs = generateGeometry(result);

    // Step 4: rasterize
    if (!segs.empty())
        rasterizeSegs(pixels, stride, w, h, segs);

    // Step 5: post to compositor
    ANativeWindow_unlockAndPost(m_window);
}

// ═════════════════════════════════════════════════════════════════════════════
// generateGeometry
// ═════════════════════════════════════════════════════════════════════════════
std::vector<OverlayRenderer::Seg>
OverlayRenderer::generateGeometry(const PipelineResult& result)
{
    std::vector<Seg> out;

    // Read all render parameters under a single lock
    float cr, cg, cb, ca, thick;
    bool  calibMode;
    bool  fillEnabled;
    float fillAlpha;
    bool  parallelLinesVisible;
    {
        std::lock_guard<std::mutex> lk(m_paramMtx);
        cr                   = m_overlayColorR;
        cg                   = m_overlayColorG;
        cb                   = m_overlayColorB;
        ca                   = m_overlayColorA;
        thick                = m_lineThickness;
        calibMode            = m_calibrationMode;
        fillEnabled          = m_parallelFillEnabled;
        fillAlpha            = m_parallelFillAlpha;
        parallelLinesVisible = m_parallelLinesVisible;
    }

    // Calibration mode: no overlay drawn
    if (calibMode) return out;

    // cv::Scalar is (B, G, R, A) — OpenCV convention
    const cv::Scalar color(cb * 255.f, cg * 255.f, cr * 255.f, ca * 255.f);

    const float fr   = result.ghostRadius;
    const float offX = static_cast<float>(result.roiX1);
    const float offY = static_cast<float>(result.roiY1);

    // ── drawSegment ───────────────────────────────────────────────────────────
    auto drawSegment = [&](std::vector<Seg>& sv,
                           cv::Point2f from,
                           cv::Point2f to,
                           cv::Point2f segDir,
                           bool        showCircle)
    {
        const cv::Point2f ext = pt_scale(segDir, fr);

        // Translucent fill band between the two dotted rails
        if (fillEnabled && fr > 0.5f) {
            const cv::Point2f fill_from = pt_add(from, ext);
            const cv::Point2f fill_to   = pt_sub(to,   ext);
            if (pt_mag(pt_sub(fill_to, fill_from)) > 0.5f &&
                pt_dot(pt_sub(fill_to, fill_from), segDir) > 0.f)
            {
                float r_f, g_f, b_f, a_f;
                scalarToFloats(color, r_f, g_f, b_f, a_f);
                sv.push_back({fill_from.x, fill_from.y,
                              fill_to.x,   fill_to.y,
                              r_f, g_f, b_f, fillAlpha, fr});
            }
        }

        // Solid centre line
        addLine(sv, from, to, color, thick);

        // Dotted guide rails at ±ghostRadius (only if visible)
        if (parallelLinesVisible) {
            const cv::Point2f perp = pt_perp(segDir);

            // Rail +perp
            const cv::Point2f s1 = pt_add(pt_add(from, pt_scale(perp,  fr)), ext);
            const cv::Point2f e1 = pt_sub(pt_add(to,   pt_scale(perp,  fr)), ext);
            if (pt_mag(pt_sub(e1, s1)) > 0.5f &&
                pt_dot(pt_sub(e1, s1), segDir) > 0.f)
                addDottedLine(sv, s1, e1, color, thick, 8, 6);

            // Rail −perp
            const cv::Point2f s2 = pt_add(pt_add(from, pt_scale(perp, -fr)), ext);
            const cv::Point2f e2 = pt_sub(pt_add(to,   pt_scale(perp, -fr)), ext);
            if (pt_mag(pt_sub(e2, s2)) > 0.5f &&
                pt_dot(pt_sub(e2, s2), segDir) > 0.f)
                addDottedLine(sv, s2, e2, color, thick, 8, 6);
        }

        // Ghost circle at cushion-reflection or pocket-entry endpoint
        if (showCircle && fr > 0.5f)
            addCircle(sv, to, fr, color, thick, 32);
    };

    // ── convertShot ───────────────────────────────────────────────────────────
    auto convertShot = [&](std::vector<Seg>& sv, const ShotEx& shot)
    {
        // ROI-local → pipeline-screen space
        const cv::Point2f from(shot.from.x + offX, shot.from.y + offY);
        const cv::Point2f to  (shot.to.x   + offX, shot.to.y   + offY);

        const cv::Point2f segDir = pt_norm(shot.dir);
        if (pt_mag(segDir) < 1e-9f) return;  // degenerate, skip

        const bool showCircle = (shot.wall >= 0) || shot.pocket_stop;
        drawSegment(sv, from, to, segDir, showCircle);
    };

    // ── Main path ─────────────────────────────────────────────────────────────
    if (!result.shots.empty()) {
        out.reserve(result.shots.size() * 70u);
        for (const auto& shot : result.shots)
            convertShot(out, shot);
    }

    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// Coordinate mapping
// pipeline/capture space → surface/display space
// ═════════════════════════════════════════════════════════════════════════════
void OverlayRenderer::pipelineToSurface(float px, float py,
                                        float& sx, float& sy) const noexcept {
    const float scaleX = static_cast<float>(m_surfaceW)
                         / static_cast<float>(m_pipelineScreenW);
    const float scaleY = static_cast<float>(m_surfaceH)
                         / static_cast<float>(m_pipelineScreenH);

    switch (m_rotMode) {
        case 1:  // 90° CW
            sx = py * scaleY;
            sy = (static_cast<float>(m_pipelineScreenW) - px) * scaleX;
            break;
        case 2:  // 180°
            sx = (static_cast<float>(m_pipelineScreenW) - px) * scaleX;
            sy = (static_cast<float>(m_pipelineScreenH) - py) * scaleY;
            break;
        case 3:  // 270° CW
            sx = (static_cast<float>(m_pipelineScreenH) - py) * scaleY;
            sy = px * scaleX;
            break;
        default: // identity
            sx = px * scaleX;
            sy = py * scaleY;
            break;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Pixel helpers
//
// WINDOW_FORMAT_RGBA_8888 on little-endian ARM:
//   memory byte layout : [ R ][ G ][ B ][ A ]  (byte address ascending)
//   as uint32_t word   : (A << 24) | (B << 16) | (G << 8) | R
//
// This is why white (1,1,1,1) → 0xFFFFFFFF was unaffected by any swap:
// all four bytes are 0xFF. Any other color requires the correct packing.
// ═════════════════════════════════════════════════════════════════════════════
uint32_t OverlayRenderer::toRGBA(float r, float g, float b, float a) noexcept {
    const uint32_t A = static_cast<uint32_t>(std::clamp(a, 0.f, 1.f) * 255.f + 0.5f);
    const uint32_t R = static_cast<uint32_t>(std::clamp(r, 0.f, 1.f) * 255.f + 0.5f);
    const uint32_t G = static_cast<uint32_t>(std::clamp(g, 0.f, 1.f) * 255.f + 0.5f);
    const uint32_t B = static_cast<uint32_t>(std::clamp(b, 0.f, 1.f) * 255.f + 0.5f);
    // Correct little-endian RGBA_8888 word: A=byte[3]=MSB, R=byte[0]=LSB
    return (A << 24u) | (B << 16u) | (G << 8u) | R;
}

// blendRGBA: alpha-blend src over dst.
// Both words use the same (A<<24)|(B<<16)|(G<<8)|R layout so the
// per-channel shift arithmetic is layout-agnostic.
// The alpha channel is always in bits [31:24].
uint32_t OverlayRenderer::blendRGBA(uint32_t dst, uint32_t src) noexcept {
    const uint32_t sA = (src >> 24u) & 0xFFu;
    if (sA == 0u)   return dst;
    if (sA == 255u) return src;

    const uint32_t dA    = (dst >> 24u) & 0xFFu;
    const uint32_t invSA = 255u - sA;

    auto blendCh = [&](uint32_t shift) -> uint32_t {
        return (((src >> shift) & 0xFFu) * sA
                + ((dst >> shift) & 0xFFu) * invSA) / 255u;
    };

    const uint32_t outA = sA + (dA * invSA) / 255u;
    return (outA        << 24u)
           | (blendCh(16u) << 16u)
           | (blendCh( 8u) <<  8u)
           |  blendCh( 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Rasterizer
// ═════════════════════════════════════════════════════════════════════════════
void OverlayRenderer::rasterizeSegs(uint32_t* pixels,
                                    int stride, int surfW, int surfH,
                                    const std::vector<Seg>& segs) const {
    const float scaleX   = static_cast<float>(surfW)
                           / static_cast<float>(m_pipelineScreenW);
    const float scaleY   = static_cast<float>(surfH)
                           / static_cast<float>(m_pipelineScreenH);
    const float scaleAvg = (scaleX + scaleY) * 0.5f;

    for (const Seg& s : segs) {
        float sx0, sy0, sx1, sy1;
        pipelineToSurface(s.x0, s.y0, sx0, sy0);
        pipelineToSurface(s.x1, s.y1, sx1, sy1);

        const uint32_t color      = toRGBA(s.r, s.g, s.b, s.a);
        const float    scaledHalf = s.halfThick * scaleAvg;

        rasterizeLine(pixels, stride, surfW, surfH,
                      sx0, sy0, sx1, sy1, color, scaledHalf);
    }
}

void OverlayRenderer::rasterizeLine(uint32_t* pixels,
                                    int stride, int surfW, int surfH,
                                    float x0, float y0,
                                    float x1, float y1,
                                    uint32_t color, float halfThick) {
    const float margin = halfThick + 2.f;
    const float fW     = static_cast<float>(surfW);
    const float fH     = static_cast<float>(surfH);

    // Trivial reject
    if ((x0 < -margin && x1 < -margin) || (x0 > fW + margin && x1 > fW + margin)) return;
    if ((y0 < -margin && y1 < -margin) || (y0 > fH + margin && y1 > fH + margin)) return;

    const float dx  = x1 - x0;
    const float dy  = y1 - y0;
    const float len = std::hypot(dx, dy);

    // Degenerate: single pixel
    if (len < 0.5f) {
        const int ix = static_cast<int>(x0);
        const int iy = static_cast<int>(y0);
        if (ix >= 0 && ix < surfW && iy >= 0 && iy < surfH) {
            const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(stride) + ix;
            pixels[idx] = blendRGBA(pixels[idx], color);
        }
        return;
    }

    // Unit tangent and normal
    const float nx = dx / len;  // tangent x
    const float ny = dy / len;  // tangent y
    const float px = -ny;       // normal x
    const float py =  nx;       // normal y
    const float hw = halfThick;

    // Bounding box of the thick segment rectangle
    const float cx[4] = { x0 + px*hw, x0 - px*hw, x1 - px*hw, x1 + px*hw };
    const float cy[4] = { y0 + py*hw, y0 - py*hw, y1 - py*hw, y1 + py*hw };

    const int bbY0 = std::max(
            static_cast<int>(std::floor(std::min({cy[0], cy[1], cy[2], cy[3]}))), 0);
    const int bbY1 = std::min(
            static_cast<int>(std::ceil( std::max({cy[0], cy[1], cy[2], cy[3]}))), surfH - 1);
    const int bbX0 = std::max(
            static_cast<int>(std::floor(std::min({cx[0], cx[1], cx[2], cx[3]}))), 0);
    const int bbX1 = std::min(
            static_cast<int>(std::ceil( std::max({cx[0], cx[1], cx[2], cx[3]}))), surfW - 1);

    // Source alpha for coverage scaling (alpha is in bits [31:24])
    const float srcA = static_cast<float>((color >> 24u) & 0xFFu) / 255.f;

    for (int y = bbY0; y <= bbY1; ++y) {
        uint32_t* const row = pixels + static_cast<size_t>(y) * static_cast<size_t>(stride);
        for (int x = bbX0; x <= bbX1; ++x) {

            // Signed distance from pixel centre to nearest point on segment axis
            const float qx    = static_cast<float>(x) - x0;
            const float qy    = static_cast<float>(y) - y0;
            const float along = std::clamp(qx * nx + qy * ny, 0.f, len);
            const float cpx   = x0 + nx * along;
            const float cpy   = y0 + ny * along;
            const float dist  = std::hypot(static_cast<float>(x) - cpx,
                                           static_cast<float>(y) - cpy);

            if (dist > hw + 1.f) continue;

            const float coverage = std::clamp(hw - dist + 0.5f, 0.f, 1.f);

            uint32_t blendSrc = color;
            if (coverage < 1.f) {
                // Scale alpha by coverage; keep R,G,B bits unchanged.
                // Alpha is always in bits [31:24] for our RGBA packing.
                const uint32_t scaledA = static_cast<uint32_t>(
                        srcA * coverage * 255.f + 0.5f);
                blendSrc = (scaledA << 24u) | (color & 0x00FFFFFFu);
            }

            row[x] = blendRGBA(row[x], blendSrc);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Geometry primitives
// ═════════════════════════════════════════════════════════════════════════════
void OverlayRenderer::addLine(std::vector<Seg>& segs,
                              cv::Point2f p1, cv::Point2f p2,
                              const cv::Scalar& color, float thickness) {
    if (pt_mag(pt_sub(p2, p1)) < 1e-6f) return;
    float r, g, b, a;
    scalarToFloats(color, r, g, b, a);
    segs.push_back({p1.x, p1.y, p2.x, p2.y, r, g, b, a, thickness * 0.5f});
}

void OverlayRenderer::addDottedLine(std::vector<Seg>& segs,
                                    cv::Point2f p1, cv::Point2f p2,
                                    const cv::Scalar& color,
                                    float thickness,
                                    int dashLen, int gapLen) {
    const cv::Point2f d   = pt_sub(p2, p1);
    const float       tot = pt_mag(d);
    if (tot < 1e-6f) return;

    const cv::Point2f n = pt_scale(d, 1.f / tot);
    float t    = 0.f;
    bool  draw = true;

    while (t < tot) {
        const float step = draw ? static_cast<float>(dashLen)
                                : static_cast<float>(gapLen);
        const float te = std::min(t + step, tot);
        if (draw)
            addLine(segs,
                    pt_add(p1, pt_scale(n, t)),
                    pt_add(p1, pt_scale(n, te)),
                    color, thickness);
        t    = te;
        draw = !draw;
    }
}

void OverlayRenderer::addCircle(std::vector<Seg>& segs,
                                cv::Point2f center, float radius,
                                const cv::Scalar& color,
                                float thickness, int segments) {
    if (segments < 3) segments = 32;
    const float step = 2.f * PI_F / static_cast<float>(segments);
    for (int i = 0; i < segments; ++i) {
        const float a1 = step * static_cast<float>(i);
        const float a2 = step * static_cast<float>(i + 1);
        const cv::Point2f p1{ center.x + radius * std::cos(a1),
                              center.y + radius * std::sin(a1) };
        const cv::Point2f p2{ center.x + radius * std::cos(a2),
                              center.y + radius * std::sin(a2) };
        addLine(segs, p1, p2, color, thickness);
    }
}