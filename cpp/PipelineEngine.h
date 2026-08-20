/**
 * PipelineEngine — Orchestrates the full Qeight pool-shot analysis pipeline.
 *
 * Pipeline stages (in order):
 *   0. PSQ scene check     — samples 6 pocket-square brightness crops; rejects
 *                            frame if fewer than PSQ_MIN_PASSING pass. Runs
 *                            concurrently with Stage 1 via std::async.
 *   1. Hollow ball detect  — intensity gate → connectedComponentsWithStats →
 *                            morph-close "delta" scoring to find the component
 *                            most likely to contain a hole → flood-fill-based
 *                            exterior/hole isolation on that winning component
 *                            → component-count and aspect-ratio guards → the
 *                            padded cumulative hole rect's center/half-diagonal
 *                            become the detected ball center/radius. Operates
 *                            directly on the capture-resolution `roi` buffer
 *                            (no internal downscale).
 *   2. Single-pass detection — the detection block (stages 3–8 below) runs
 *                            exactly once per frame on the original, unexpanded
 *                            ROI. There is no two-pass / ROI-expansion mechanism.
 *   3. Full-region erase-pad — zero-erases only the hollow-ball circle from the
 *                            full ROI image. erase_radius is derived per-frame
 *                            from the detected ball radius. BGR output.
 *   4. Seed cluster match  — intensity gate (SEED_INTENSITY ± SEED_THRESHOLD) +
 *                            connected components filtered by minimum pixel count
 *                            to find bright line clusters.
 *   5. Strip centerline    — pure PCA-projection: projects cluster pixels onto
 *                            the PCA major axis to find endpoints; orders
 *                            medial_near/medial_far by distance to hollow_c.
 *   6. Gating              — search-circle + auto-tuned min-length/collinearity
 *                            grid search selects which ridges are valid and
 *                            determines n_valid_lines.
 *   7. Classification      — n_valid_lines branching:
 *                              0    : no output.
 *                              1 + near-edge: reflected ATStrip produced via
 *                                wall-contact ray cast; added to unified list.
 *                              2    : axis-angle test selects TGT + optional CBC.
 *                              3    : axis-angle test selects TGT + CBC (closer-
 *                                to-perpendicular ridge wins as CBC).
 *                              >3   : all ridges discarded.
 *   8. AT strips           — buildATStripFromRidge for TGT / CBC ridges.
 *   9. Coordinate restore  — no-op pass-through (restore_origin is always (0,0)).
 *  10. Ghost correction    — clamps AT strip origins to pool-table interior.
 *  11. Shot building       — ray-cast with reflections and pocket detection.
 *
 * All AT strips (TGT, CBC, near-edge reflected) are placed in a single unified
 * list and processed identically by downstream stages.
 *
 * Android-specific notes:
 *   - The capture buffer delivered to processFrame() is ALREADY downscaled to
 *     ~720 px height (captureScale = 720/nativeScreenHeight, clamped to
 *     [0.05, 1.0]). PipelineEngine never sees a native-resolution frame.
 *   - captureScreenW/captureScreenH passed to the constructor are the dimensions
 *     of that downscaled buffer — processFrame() expects exactly that size.
 *   - Callers must convert any native-screen-space ROI / UI coordinates into
 *     capture-resolution space (multiply by captureScale) before calling the
 *     constructor, setRoi(), or setPoolTableBounds().
 *   - processFrame() is expected to run on a performance-core-affined thread.
 *   - Pool-table boundaries are supplied at runtime via setPoolTableBounds() and
 *     are used to clamp line_maps-ball origins and build shot-ray bounds.
 *
 * SANITY CONTRACT (checked at construction time in the .cpp):
 *   If captureScale < 1.0 and either captureScreenW or captureScreenH exceeds
 *   CAPTURE_DIM_SANITY_LIMIT (2000 px), an LOGE warning is emitted because the
 *   caller almost certainly passed native-resolution dimensions instead of
 *   downscaled capture-resolution dimensions. The pipeline is NOT aborted, but
 *   byteCount validation in processFrame will reject every frame until the object
 *   is re-constructed with correct dimensions.
 */
#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Plain data structures shared across pipeline stages and the renderer.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Single line segment produced by the medial-axis fitter.
 * Carries pt1, pt2, length, and valid.
 */
struct EDLine {
    cv::Point2f pt1, pt2;
    float length = 0;
    bool  valid  = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Ghost-ball correction structures.
// ─────────────────────────────────────────────────────────────────────────────

/** Result of clamping a line_maps ball origin to remain inside the pool table. */
struct GhostTangentResult {
    cv::Point2f center;
    int  wall      = -1;
    [[maybe_unused]] bool corrected = false;
};

/** Result of the hollow-ball detector (Stage 1). */
struct HollowDetectResult {
    cv::Point2f hollow_center_orig;       ///< Hollow ball centre in ROI-local pixels.
    float       hollow_radius_orig = 0.f; ///< Detected dynamically each frame from the hole's bounding rect.
    bool        valid              = false;
};

/**
 * Seed cluster found by the intensity-gate + connected-components stage.
 * Represents one bright line/cluster of pixels in the erase-pad image.
 */
struct SeedCluster {
    std::vector<cv::Point> pixels;  ///< All pixel positions belonging to this cluster.
    cv::Point2f centroid;           ///< Mean pixel position.
    float major_axis_length = 0.f; ///< 2*sqrt(eigenvalue_max).
    float orientation_deg   = 0.f; ///< Orientation of major axis in degrees.
};

/**
 * Medial-axis centerline result for one SeedCluster, produced by
 * findStripCenterline (pure PCA-projection).
 */
struct RidgeResult {
    cv::Point2f medial_near, medial_far; ///< Medial endpoints ordered near-to-far from hollow ball.
    cv::Point2f extend_dir;              ///< Unit direction from near to far along the medial axis.
    EDLine      source_line{};           ///< EDLine built from the fitted medial segment.
    bool        valid       = false;
};

/**
 * Aiming-trajectory strip derived from one SeedCluster ridge or wall-reflection
 * geometry. TGT, CBC, and near-edge reflected strips all share this type and are
 * held in a single unified list.
 */
struct ATStrip {
    cv::Point2f origin, evec;
    [[maybe_unused]] float angle_deg   = 0;
    bool        is_cue_ball_cut = false;
    bool        valid           = false;
};

/** Inset bounding rectangle representing the valid play area. */
struct Bounds {
    float top = 0, bottom = 0, left = 0, right = 0;
    [[nodiscard]] bool valid() const { return top < bottom && left < right; }
};

/** A single pocket on the pool table. */
struct Pocket {
    cv::Point2f center;
    float       radius = 0;
};

/** One segment of a multi-reflection shot ray. */
struct ShotEx {
    cv::Point2f from, to;
    cv::Point2f dir;
    int         wall         = -1;
    bool        pocket_stop  = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Top-level pipeline result (renderer-facing snapshot).
// ─────────────────────────────────────────────────────────────────────────────

/** All data the renderer needs to draw the overlay for one processed frame. */
struct PipelineResult {
    std::vector<ATStrip> ats;     ///< Unified list: TGT, CBC, and reflected strips.
    std::vector<Pocket>  pockets;
    std::vector<ShotEx>  shots;
    [[maybe_unused]] int roiX1 = 0, roiY1 = 0, roiX2 = 0, roiY2 = 0;
    int poolX1 = 0, poolY1 = 0, poolX2 = 0, poolY2 = 0;
    float ghostRadius = 22.023f;
    [[maybe_unused]] int maxr_cbc = 0;
    int maxr_tgt = 1;
    [[maybe_unused]] int pocketR = 40, pocketNS = 30;
};

// ─────────────────────────────────────────────────────────────────────────────
// PipelineEngine
// ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPhysicsEngine;

class PipelineEngine {
public:
    /**
     * @param captureScreenW  Width  of the downscaled capture buffer (pixels).
     * @param captureScreenH  Height of the downscaled capture buffer (pixels).
     * @param captureScale    Native→capture downscale ratio in (0, 1]. Used only
     *                        during construction to derive scaled runtime constants.
     * @param roiX1/roiY1/roiX2/roiY2
     *                        Region of interest in capture-resolution coordinates.
     */
    PipelineEngine(int captureScreenW, int captureScreenH, float captureScale,
                   int roiX1, int roiY1, int roiX2, int roiY2);
    ~PipelineEngine();

    /** Runs all pipeline stages on one RGBA capture-resolution frame buffer. */
    void processFrame(void* pixelData, size_t byteCount);

    /**
     * Returns a by-value snapshot of the most recent pipeline result.
     * Thread-safe; copy is made while holding m_resultMutex.
     */
    PipelineResult copyLastResult() const {
        std::lock_guard<std::mutex> lock(m_resultMutex);
        return m_result;
    }

    // ── Parameter setters ─────────────────────────────────────────────────────
    /// All coordinate arguments must be in capture-resolution space.
    void setRoi(int x1, int y1, int x2, int y2);
    void setCbcReflections(int count);
    void setTgtReflections(int count);
    void setCushionShots(bool enabled);
    void setPocketParams(int radius, int nsShift);
    void setTrajectoryPower(int pct);
    void setCueForce(int stat);
    void setCueSpin(int stat);
    /// All coordinate arguments must be in capture-resolution space.
    void setPoolTableBounds(int x1, int y1, int x2, int y2);

    // ── Additive read-only getters (used by QeightJNI indirect-shot path) ────

    /**
     * Returns the four pool-table bounds currently stored on the engine
     * (capture-resolution pixel coordinates, same values set by
     * setPoolTableBounds). Used by solveIndirectShot to replicate the
     * px→cm conversion that TrajectoryPhysicsEngine::computeTrajectories uses.
     */
    void getPoolTableBounds(int& x1, int& y1, int& x2, int& y2) const {
        x1 = m_poolX1.load();
        y1 = m_poolY1.load();
        x2 = m_poolX2.load();
        y2 = m_poolY2.load();
    }

    /**
     * Returns the cue force and spin stat values currently stored on the engine
     * (last values set by setCueForce / setCueSpin). Used by solveIndirectShot
     * to compute full-power launch speed via PoolPhysicsEngine::applyCueStats.
     */
    void getCueStats(int& forceStat, int& spinStat) const {
        forceStat = m_cueForceStat.load();
        spinStat  = m_cueSpinStat.load();
    }

    /**
     * Exposes the TrajectoryPhysicsEngine owned by PipelineEngine so that
     * QeightJNI can obtain a const reference to its PoolPhysicsEngine instance
     * (via TrajectoryPhysicsEngine::getPhysicsEngine()) without constructing a
     * second, independent physics object with different calibration.
     * The reference is valid for the lifetime of this PipelineEngine instance.
     */
    const TrajectoryPhysicsEngine& getTrajectoryEngine() const {
        return *m_trajectoryPhysics;
    }

    /**
     * Returns the captureScale passed to the constructor (native→capture
     * downscale ratio). Stored once at construction time and never mutated,
     * so no locking is needed. Used by QeightJNI's indirect-shot path to
     * derive the same canonical pixel ball radius direct mode uses
     * (gr = max(4.f, 22.023f * captureScale), matching m_ghostRadius's
     * derivation below) instead of an independent constant.
     */
    float getCaptureScale() const {
        return m_captureScale;
    }

private:
    // ── Member variables ──────────────────────────────────────────────────────
    int m_screenW, m_screenH;
    std::atomic<int>  m_roiX1, m_roiY1, m_roiX2, m_roiY2;
    std::atomic<int>  m_poolX1{0}, m_poolY1{0}, m_poolX2{0}, m_poolY2{0};
    std::atomic<float>  m_ghostRadius{22.023f};
    std::atomic<int>  m_maxr_cbc{0}, m_maxr_tgt{1};
    std::atomic<int>  m_pocketR{40}, m_pocketNS{30};
    std::atomic<int>  m_trajectoryPowerPct{100};
    std::atomic<int>  m_cueForceStat{8};
    std::atomic<int>  m_cueSpinStat{8};
    std::atomic<bool> m_cushionShots{true};
    std::unique_ptr<TrajectoryPhysicsEngine> m_trajectoryPhysics;
    PipelineResult     m_result;
    mutable std::mutex m_resultMutex;
    uint64_t           m_frameCount = 0;

    // ── Runtime state ─────────────────────────────────────────────────────────
    std::chrono::high_resolution_clock::time_point m_constructTime;

    // ── Runtime-scaled constants (computed once in constructor) ───────────────

    /// captureScale as passed to the constructor. Stored verbatim (not
    /// atomic — written once here and never mutated again, same pattern as
    /// m_holePad et al. below) so getCaptureScale() can expose it without
    /// locking.
    float m_captureScale;

    /// Edge-proximity threshold for the near-edge CA-fallback test (Stage 7, n=1).
    /// Native calibration: 26.5 px.  Scaled: 26.5f * captureScale.
    float m_caFbEdgeThresh;

    // Pixel-space constants scaled from native px at construction time.
    // None of these are atomics — they are written once in the constructor and
    // never mutated again.
    int   m_holePad;             ///< HOLE_PAD * captureScale
    int   m_holeCcMinArea;       ///< HOLE_CC_MIN_AREA * captureScale^2 (area)
    int   m_holeDiagMin;         ///< HOLE_DIAG_MIN * captureScale
    int   m_holeDiagMax;         ///< HOLE_DIAG_MAX * captureScale
    int   m_psqCornerOffset;     ///< PSQ_CORNER_OFFSET * captureScale
    int   m_psqMidOffset;        ///< PSQ_MID_OFFSET * captureScale
    int   m_psqSize;             ///< PSQ_SIZE * captureScale
    float m_cornerPocketShift;   ///< CORNER_POCKET_SHIFT * captureScale

    // ── Plausibility limit for constructor-time sanity check ──────────────────
    static constexpr int CAPTURE_DIM_SANITY_LIMIT = 2000;

    // ── Seed cluster / strip-detection constants ──────────────────────────────
    static constexpr float SEED_INTENSITY     = 254.77f; ///< Target luminance for intensity gate.
    static constexpr int   SEED_THRESHOLD     =       2; ///< ±threshold around SEED_INTENSITY.
    static constexpr int   MIN_CLUSTER_PIXELS =      20; ///< Minimum pixel count to accept a seed cluster.

    // ── Hole-detection tuning (Stage 1) ──────────────────────────────────────
    static constexpr int   HOLE_PAD         =   2;   ///< bbox padding, native px
    static constexpr int   HOLE_CC_MIN_AREA = 300;   ///< min hole-component area, native px^2
    static constexpr int   HOLE_DIAG_MIN    =  25;   ///< min half-diagonal, native px
    static constexpr int   HOLE_DIAG_MAX    =  45;   ///< max half-diagonal, native px

    // ── Pocket-square scene validation (Stage 0) ──────────────────────────────
    static constexpr int   PSQ_CORNER_OFFSET =  30;   ///< native px
    static constexpr int   PSQ_MID_OFFSET    =  30;   ///< native px
    static constexpr int   PSQ_SIZE          =  20;   ///< native px, sample square side
    static constexpr float PSQ_BRIGHT_MIN    =   0.f;
    static constexpr float PSQ_BRIGHT_MAX    =  40.f;
    static constexpr int   PSQ_MIN_PASSING   =   4;   ///< out of 6 samples, NOT px-scaled

    // ── Pocket geometry (Stage 11) ────────────────────────────────────────────
    static constexpr int   CORNER_POCKET_SHIFT = 8;   ///< native px, diagonal inset on 4 corner pockets only

    // ── Logging control ────────────────────────────────────────────────────────
    static constexpr bool    LOGS_ENABLED    = false;  ///< default: OFF
    static constexpr int64_t LOG_AUTO_OFF_MS = 180000; ///< 3 minutes

    // ── Logging gate ──────────────────────────────────────────────────────────
    /**
     * Returns true if LOGS_ENABLED and the engine has been alive for less than
     * LOG_AUTO_OFF_MS milliseconds. Reads m_constructTime directly.
     */
    bool logsActive() const;

    // ── Stage 0: pocket-square scene validation ──────────────────────────────
    /**
     * Samples mean brightness in 6 small squares around the pool-table bounds
     * (4 corners offset diagonally by psq_corner_offset, 2 mid-rail points
     * offset by psq_mid_offset) and counts how many fall within
     * [psq_bright_min, psq_bright_max]. Returns the passing count (0-6).
     * Operates directly on the full ROI buffer in ROI-local coordinates.
     * Sample evaluation is split across two threads by sample index.
     */
    int countPassingPocketSquares(
            const cv::Mat& roi,
            int ptOX, int ptOY, int ptW, int ptH,
            int psq_corner_offset, int psq_mid_offset, int psq_size,
            float psq_bright_min, float psq_bright_max);

    // ── Stage 1: hollow-ball detector ────────────────────────────────────────
    /**
     * Detects the hollow ball via: intensity gate →
     * connectedComponentsWithStats → morph-close delta scoring → flood-fill
     * exterior isolation → hole component union → guards → padded rect center.
     * Uses m_holeCcMinArea, m_holePad, m_holeDiagMin, m_holeDiagMax.
     * Returns valid==false when no hollow ball candidate is found in roi.
     */
    HollowDetectResult detectHollowBall(const cv::Mat& roi);

    // ── Stage 3: erase-pad ────────────────────────────────────────────────────
    /**
     * Erases only the hollow-ball circle from a full-size copy of src.
     * erase_radius is derived per-frame from detectHollowBall's result:
     * erase_radius = max(2, round(detected_radius)).
     */
    static cv::Mat buildErasePadFull(const cv::Mat& src,
                                     cv::Point2f hollow_c,
                                     int erase_radius);

    // ── Stage 4: seed cluster detection ──────────────────────────────────────
    /**
     * Per-pixel luminance gate: intensity = 0.299R + 0.587G + 0.114B (BGR order).
     * Output mask pixel = 255 if |intensity - seed_intensity| <= threshold, else 0.
     * On ARM NEON targets an accelerated float-intrinsic path is used
     * automatically (row-parallel across worker threads); a scalar fallback
     * is used on non-NEON targets.
     */
    static cv::Mat computeIntensityGateMask(const cv::Mat& bgr,
                                            float seed_intensity,
                                            int   threshold);

    /**
     * Connected components (8-connectivity) filtered by minimum pixel count.
     * Returns clusters sorted descending by major_axis_length.
     */
    std::vector<SeedCluster> findSeedClusters(
            const cv::Mat& candidate_mask,
            int min_pixels);

    // ── Stage 5: strip centerline (pure PCA-projection) ───────────────────────
    /**
     * Projects all cluster pixels onto the PCA major axis to find extremal
     * endpoints, then orders medial_near/medial_far by distance to hollow_c.
     */
    RidgeResult findStripCenterline(
            const SeedCluster& cluster,
            cv::Point2f hollow_c);

    // ── Stage 8: AT strip construction ───────────────────────────────────────
    /**
     * Builds an ATStrip from a ridge. Origin is always hollow_c_crop clamped
     * to image bounds; evec comes from rr.extend_dir oriented away from the ball.
     */
    static ATStrip buildATStripFromRidge(const EDLine& ln, const RidgeResult& rr,
                                         cv::Point2f hollow_c_crop, int hollow_r,
                                         int iw, int ih, bool is_cbc);

    // ── Ghost ball correction ─────────────────────────────────────────────────
    static GhostTangentResult computeGhostTangentOrigin(cv::Point2f axis_origin,
                                                        cv::Point2f evec,
                                                        float gr,
                                                        float pool_x1, float pool_y1,
                                                        float pool_x2, float pool_y2);
    static cv::Point2f correctATGhostOrigin(cv::Point2f origin, cv::Point2f evec,
                                            float gr,
                                            float pool_x1, float pool_y1,
                                            float pool_x2, float pool_y2,
                                            int& tangent_wall_out);

    // ── Shot ray construction ─────────────────────────────────────────────────
    static std::vector<ShotEx> buildShots(const std::vector<ATStrip>& ats,
                                          const Bounds& ptBounds,
                                          const std::vector<Pocket>& pockets,
                                          int iw, int ih,
                                          int maxr_cbc, int maxr_tgt);

    /**
     * Builds 6 Pocket structs from pool-table bounds. Corner pockets are offset
     * diagonally by m_cornerPocketShift * (√2/2); mid-rail pockets use ns_shift
     * unmodified. Non-static: reads m_cornerPocketShift.
     */
    std::vector<Pocket> makePockets(int ox, int oy, int w, int h,
                                    float pr, float ns_shift);

    // ── Coordinate restore helper ─────────────────────────────────────────────
    /** Translates an ATStrip's origin by crop_origin (no-op when origin is (0,0)). */
    static ATStrip restoreAT(ATStrip strip, cv::Point2f crop_origin);

    // ── Geometry and ray-casting helpers ─────────────────────────────────────
    static float       normAng(float d);
    static Bounds      makeBounds(int x1, int y1, int x2, int y2, int inset);
    static cv::Point2f reflDir(cv::Point2f d, int w);
    static cv::Point2f wallNorm(int w);
    static int         rayHit(cv::Point2f o, cv::Point2f d, const Bounds& b,
                              int skip, float& tout, cv::Point2f& gout);
    static bool        inPocket(cv::Point2f p, const std::vector<Pocket>& pockets);
    static float       rayPocketT(cv::Point2f o, cv::Point2f d,
                                  const std::vector<Pocket>& pockets, float maxT);
};