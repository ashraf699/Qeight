#pragma once
#include <android/native_window.h>
#include "PipelineEngine.h"
#include <opencv2/core.hpp>
#include <cstdint>
#include <mutex>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Global logging switch.
// Shared by OverlayRenderer.cpp and QeightJNI.cpp (both include this header).
// Set to true to enable LOGD/LOGE/LOGW output; false to disable it.
// Default: OFF.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr bool kLoggingEnabled = false;

class OverlayRenderer {
public:
    OverlayRenderer(const OverlayRenderer&)            = delete;
    OverlayRenderer& operator=(const OverlayRenderer&) = delete;

    OverlayRenderer(ANativeWindow* window,
                    int pipelineScreenW = 2400,
                    int pipelineScreenH = 1080);
    ~OverlayRenderer();

    // Synchronously: lock → clear → generate → rasterize → post.
    void renderFrame(const PipelineResult& result);

    // Parameter setters — safe to call concurrently from the UI thread.
    void setOverlayColor(float r, float g, float b, float a);
    void setLineThickness(float thickness);
    void setCalibrationMode(bool enabled);
    void setParallelLinesFill(bool enabled, float alpha);
    void setParallelLinesVisible(bool visible);

private:
    // Line-segment primitive in pipeline/capture-buffer pixel space.
    struct Seg {
        float x0, y0, x1, y1;
        float r, g, b, a;
        float halfThick;
    };

    ANativeWindow* m_window          = nullptr;
    int            m_surfaceW        = 0;
    int            m_surfaceH        = 0;
    int            m_pipelineScreenW = 2400;
    int            m_pipelineScreenH = 1080;
    int            m_rotMode         = 0;

    // Overlay drawing parameters — guarded by m_paramMtx.
    mutable std::mutex m_paramMtx;
    float m_overlayColorR        = 0.f;
    float m_overlayColorG        = 200.f / 255.f;
    float m_overlayColorB        = 80.f  / 255.f;
    float m_overlayColorA        = 1.f;
    float m_lineThickness        = 2.f;
    bool  m_calibrationMode      = false;
    bool  m_parallelFillEnabled  = true;
    float m_parallelFillAlpha    = 0.15f;
    bool  m_parallelLinesVisible = true;

    // Geometry generation
    std::vector<Seg> generateGeometry(const PipelineResult& result);

    // Geometry primitives
    static void addLine(std::vector<Seg>& segs,
                        cv::Point2f p1, cv::Point2f p2,
                        const cv::Scalar& color, float thickness);
    static void addDottedLine(std::vector<Seg>& segs,
                              cv::Point2f p1, cv::Point2f p2,
                              const cv::Scalar& color, float thickness,
                              int dashLen, int gapLen);
    static void addCircle(std::vector<Seg>& segs,
                          cv::Point2f center, float radius,
                          const cv::Scalar& color, float thickness,
                          int segments = 32);

    // Rasterizer
    void rasterizeSegs(uint32_t* pixels, int stride, int surfW, int surfH,
                       const std::vector<Seg>& segs) const;
    static void rasterizeLine(uint32_t* pixels, int stride, int surfW, int surfH,
                              float x0, float y0, float x1, float y1,
                              uint32_t color, float halfThick);

    // Coordinate mapping and pixel helpers
    void pipelineToSurface(float px, float py,
                           float& sx, float& sy) const noexcept;

    // toRGBA packs float r,g,b,a into a uint32_t word matching
    // WINDOW_FORMAT_RGBA_8888 on little-endian ARM:
    //   memory byte order: R, G, B, A
    //   as 32-bit LE word: (A<<24)|(B<<16)|(G<<8)|R
    static uint32_t toRGBA(float r, float g, float b, float a) noexcept;
    static uint32_t blendRGBA(uint32_t dst, uint32_t src) noexcept;
};