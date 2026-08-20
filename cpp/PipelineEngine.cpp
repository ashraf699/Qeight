/**
 * PipelineEngine.cpp — Seed Cluster Pipeline
 *
 * Stages:
 *  0. PSQ scene check       — samples 6 pocket-square brightness crops; rejects
 *                             frame if fewer than PSQ_MIN_PASSING pass. Runs
 *                             concurrently with Stage 1 via std::async.
 *  1. detectHollowBall      — intensity gate → connectedComponentsWithStats →
 *                             morph-close delta scoring to find the component
 *                             with a hole → flood-fill exterior isolation →
 *                             hole-component union → component-count and
 *                             aspect-ratio guards → padded-rect center/radius.
 *                             Operates directly on the capture-resolution
 *                             `roi` buffer (no internal downscale).
 *  2. Single-pass detection — the detection block (stages 3-9 below) runs
 *                             exactly once per frame on the original,
 *                             unexpanded ROI. There is no two-pass /
 *                             ROI-expansion mechanism.
 *  3. buildErasePadFull     — BGR erase-pad on full proc_roi; erase radius is
 *                             derived per-frame from the detected ball radius
 *                             (erase_radius = max(2, round(detected_radius))).
 *  4. computeIntensityGateMask + findSeedClusters (min_pixels filter only).
 *  5. findStripCenterline   — pure PCA-projection algorithm (no aura-cast/RANSAC).
 *  6. Gating                — search-circle + auto-tuned min-length/collinearity
 *                             grid search selects which ridges are valid and
 *                             determines n_valid_lines.
 *  7. Classification        — n_valid_lines branching:
 *                               0    : no output.
 *                               1 + near-edge: reflected ATStrip produced via
 *                                 wall-contact ray cast; added to unified list.
 *                               2    : axis-angle (CW/ACW) test against the
 *                                 TGT ridge's perpendicular axes selects an
 *                                 optional CBC strip.
 *                               3    : axis-angle test selects TGT + CBC (the
 *                                 ridge closer to the CW/ACW axes wins as CBC).
 *                               >3   : all ridges discarded.
 *  8. AT strips             — buildATStripFromRidge for TGT/CBC ridges; the
 *                             n=1 near-edge case builds a reflected ATStrip
 *                             directly (wall-contact ray cast + reflection).
 *                             All AT strips (TGT, CBC, near-edge reflected)
 *                             are placed in a single unified list and
 *                             processed identically by downstream stages.
 *  9. Coordinate restore    — no-op pass-through (restore_origin is always (0,0)).
 * 10. Ghost correction      — clamps AT strip origins to pool-table interior.
 * 11. Shot building         — ray-cast with reflections and pocket detection.
 */

#include "PipelineEngine.h"
#include "TrajectoryPhysicsEngine.h"
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <sched.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <future>
#include <vector>
#include <functional>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#define LOG_TAG "QeightPipeline"
// LOGD/LOGE are gated by PipelineEngine::LOGS_ENABLED (declared in
// PipelineEngine.h), the same master switch used by logsActive().
// All call sites are inside PipelineEngine member functions, so the
// private static constexpr is accessible here. When disabled, calls
// compile away to nothing — no functional change, only logging output
// is suppressed.
#define LOGD(...) do { if (PipelineEngine::LOGS_ENABLED) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGE(...) do { if (PipelineEngine::LOGS_ENABLED) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } while (0)

static const float PI_F = static_cast<float>(CV_PI);
using HRC = std::chrono::high_resolution_clock;

static inline float elapsed_ms(HRC::time_point a, HRC::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

// Logging-gate helper: returns true only while LOGS_ENABLED is set and the
// engine has been alive for less than LOG_AUTO_OFF_MS milliseconds.
// Declared as a private member (logsActive() const) so it can read
// m_constructTime and the private constexpr constants directly.
bool PipelineEngine::logsActive() const {
    if (!LOGS_ENABLED) return false;
    return elapsed_ms(m_constructTime, HRC::now()) <
           static_cast<float>(LOG_AUTO_OFF_MS);
}

static inline cv::Point2f pt_scale(cv::Point2f p, float s) { return {p.x*s, p.y*s}; }
static inline cv::Point2f pt_add(cv::Point2f a, cv::Point2f b) { return {a.x+b.x, a.y+b.y}; }
static inline cv::Point2f pt_sub(cv::Point2f a, cv::Point2f b) { return {a.x-b.x, a.y-b.y}; }
static inline float pt_dot(cv::Point2f a, cv::Point2f b) { return a.x*b.x + a.y*b.y; }
static inline float pt_mag(cv::Point2f a) { return std::hypot(a.x, a.y); }
static inline cv::Point2f pt_norm(cv::Point2f a) {
    float m = pt_mag(a);
    return m > 1e-9f ? pt_scale(a, 1.f/m) : cv::Point2f(0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance-only parallel helper.
//
// This codebase pins its processing thread to 2 cores (6 and 7) via
// sched_setaffinity in the constructor. kWorkerThreads is kept consistent
// with that core count: splitting work across more threads than pinned
// cores would not add throughput and would only add scheduling overhead.
//
// No OpenMP pragmas are used here: there is no visible Android NDK build
// configuration in this file confirming -fopenmp is wired into the Android
// build chain (unlike the desktop build, which the original pipeline design
// doc says links it), so per the optimization-pass instructions a manual
// std::thread-based split is used instead, which is portable and requires
// no build-system changes.
//
// parallelForRows splits [0, rows) into kWorkerThreads contiguous,
// non-overlapping row ranges and runs `body(rowStart, rowEnd)` for each
// range on its own thread, then joins all threads before returning. Each
// invocation of `body` must only write to its own [rowStart, rowEnd) slice
// of any output buffer, so worker threads never write to overlapping memory.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int kWorkerThreads = 2; // matches sched_setaffinity cores {6,7}

static void parallelForRows(int rows, const std::function<void(int, int)>& body) {
    if (rows <= 0) return;
    // Small inputs: avoid thread-launch overhead, just run inline.
    if (rows < 64) { body(0, rows); return; }

    const int n_threads = std::min(kWorkerThreads, rows);
    const int chunk = (rows + n_threads - 1) / n_threads;

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    for (int t = 1; t < n_threads; ++t) {
        int r0 = std::min(rows, t * chunk);
        int r1 = std::min(rows, r0 + chunk);
        if (r0 >= r1) continue;
        workers.emplace_back(body, r0, r1);
    }
    // Run the first chunk on the calling thread instead of spawning an
    // extra worker for it.
    int r1_first = std::min(rows, chunk);
    body(0, r1_first);

    for (auto& w : workers) w.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

PipelineEngine::PipelineEngine(
        int captureScreenW, int captureScreenH,
        float captureScale,
        int roiX1, int roiY1, int roiX2, int roiY2)
        : m_screenW(captureScreenW), m_screenH(captureScreenH),
          m_roiX1(roiX1), m_roiY1(roiY1),
          m_roiX2(roiX2), m_roiY2(roiY2)
{
    // ── Constructor-time sanity check ─────────────────────────────────────────
    if (captureScale < 1.0f &&
        (captureScreenW > CAPTURE_DIM_SANITY_LIMIT ||
         captureScreenH > CAPTURE_DIM_SANITY_LIMIT))
    {
        LOGE("SANITY FAIL: captureScale=%.4f but captureScreenW=%d captureScreenH=%d "
             "exceed the %d-px plausibility limit. "
             "Did you pass native-resolution dimensions instead of "
             "capture-resolution dimensions? "
             "Every processFrame() call will fail byteCount validation until "
             "PipelineEngine is re-constructed with correct dimensions.",
             captureScale,
             captureScreenW,
             captureScreenH,
             CAPTURE_DIM_SANITY_LIMIT);
    }

    // ── Derive scaled constants ───────────────────────────────────────────────
    m_captureScale    = captureScale;
    m_caFbEdgeThresh  = 26.5f * captureScale;
    m_ghostRadius.store(std::max(4.f, 22.023f * captureScale));
    m_pocketR.store(    std::max(5, static_cast<int>(std::round(40.f  * captureScale))));
    m_pocketNS.store(   std::max(0, static_cast<int>(std::round(30.f  * captureScale))));

    // ── Scale new pixel-space constants once at construction time ────────────
    // These are stored as plain (non-atomic) members because they are set
    // only here and never mutated after construction.
    m_holePad            = static_cast<int>(std::round(static_cast<float>(HOLE_PAD)            * captureScale));
    m_holeCcMinArea      = static_cast<int>(std::round(static_cast<float>(HOLE_CC_MIN_AREA)    * captureScale * captureScale)); // area: scale^2
    m_holeDiagMin        = static_cast<int>(std::round(static_cast<float>(HOLE_DIAG_MIN)       * captureScale));
    m_holeDiagMax        = static_cast<int>(std::round(static_cast<float>(HOLE_DIAG_MAX)       * captureScale));
    m_psqCornerOffset    = static_cast<int>(std::round(static_cast<float>(PSQ_CORNER_OFFSET)   * captureScale));
    m_psqMidOffset       = static_cast<int>(std::round(static_cast<float>(PSQ_MID_OFFSET)      * captureScale));
    m_psqSize            = static_cast<int>(std::round(static_cast<float>(PSQ_SIZE)            * captureScale));
    m_cornerPocketShift  = static_cast<float>(CORNER_POCKET_SHIFT) * captureScale;

    // Record construction time for the logging auto-off gate.
    m_constructTime = HRC::now();

    LOGD("PipelineEngine: New scaled constants: holePad=%d holeCcMinArea=%d "
         "holeDiagMin=%d holeDiagMax=%d psqCornerOffset=%d psqMidOffset=%d "
         "psqSize=%d cornerPocketShift=%.2f",
         m_holePad, m_holeCcMinArea, m_holeDiagMin, m_holeDiagMax,
         m_psqCornerOffset, m_psqMidOffset, m_psqSize,
         m_cornerPocketShift);

    LOGD("PipelineEngine: Constructed screen=%dx%d scale=%.4f roi=(%d,%d,%d,%d)",
         captureScreenW, captureScreenH, captureScale, roiX1, roiY1, roiX2, roiY2);
    LOGD("PipelineEngine: Scaled constants: caFbEdgeThresh=%.3f "
         "ghostRadius=%.3f pocketR=%d pocketNS=%d",
         m_caFbEdgeThresh,
         m_ghostRadius.load(), m_pocketR.load(), m_pocketNS.load());

    m_trajectoryPhysics = std::make_unique<TrajectoryPhysicsEngine>();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    CPU_SET(7, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
        LOGD("PipelineEngine: sched_setaffinity failed — running on default policy");
    else
        LOGD("PipelineEngine: Thread pinned to cores 6+7");
}

PipelineEngine::~PipelineEngine() { LOGD("PipelineEngine: Destructor called"); }

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe parameter setters
// ─────────────────────────────────────────────────────────────────────────────

void PipelineEngine::setRoi(int x1, int y1, int x2, int y2) {
    m_roiX1=x1; m_roiY1=y1; m_roiX2=x2; m_roiY2=y2;
    LOGD("PipelineEngine::setRoi: (%d,%d,%d,%d)", x1, y1, x2, y2);
}
void PipelineEngine::setCbcReflections(int count)  { m_maxr_cbc   = std::clamp(count,  0,  8); }
void PipelineEngine::setTgtReflections(int count)  { m_maxr_tgt   = std::clamp(count,  0,  8); }
void PipelineEngine::setCushionShots(bool enabled) { m_cushionShots = enabled; }
void PipelineEngine::setPocketParams(int radius, int nsShift) {
    m_pocketR  = std::clamp(radius,  5, 200);
    m_pocketNS = std::clamp(nsShift, 0, 300);
}
void PipelineEngine::setTrajectoryPower(int pct) {
    m_trajectoryPowerPct = std::clamp(pct, 1, 100);
}
void PipelineEngine::setCueForce(int stat) {
    m_cueForceStat = std::clamp(stat, 0, 16);
}
void PipelineEngine::setCueSpin(int stat) {
    m_cueSpinStat = std::clamp(stat, 0, 16);
}
void PipelineEngine::setPoolTableBounds(int x1, int y1, int x2, int y2) {
    m_poolX1=x1; m_poolY1=y1; m_poolX2=x2; m_poolY2=y2;
    LOGD("PipelineEngine::setPoolTableBounds: (%d,%d,%d,%d)", x1, y1, x2, y2);
}

// ─────────────────────────────────────────────────────────────────────────────
// detectHollowBall
//
// New algorithm (replaces the old intensity-gate → CC hull-scoring →
// Hough-on-crop pipeline):
//   1. Intensity gate mask (unchanged helper).
//   2. connectedComponentsWithStats (8-conn) on the gate mask.
//   3. Morph-close the gate mask (15x15 ellipse) and XOR against the original
//      to build a "delta" mask highlighting pixels that get filled in by
//      closing — i.e. candidate hole regions.
//   4. For each label, count delta-mask pixels inside its bbox; the label
//      with the most delta pixels is the "winning" component (the one most
//      likely to have a hole).
//   5. Isolate the winning component into its own mask, pad it by 1px, and
//      flood-fill from each zero-valued corner to build an "exterior" mask
//      (the background reachable from outside the shape).
//   6. hole_pixels = NOT(exterior) AND NOT(winning component) — pixels that
//      are enclosed by the winning component's outline but not part of it.
//   7. Connected components on hole_pixels (area > m_holeCcMinArea) are
//      unioned into a cumulative bounding rect, subject to a component-count
//      guard and an aspect-ratio guard.
//   8. The padded (m_holePad) cumulative rect's center and half-diagonal
//      become the detected ball center/radius, gated by [m_holeDiagMin,
//      m_holeDiagMax].
//
// PLATFORM NOTE: this Android build already receives a pre-downscaled
// (~720p-equivalent) buffer as `roi`; no additional internal resize/downscale
// is applied here.
// ─────────────────────────────────────────────────────────────────────────────

HollowDetectResult PipelineEngine::detectHollowBall(const cv::Mat& roi)
{
    HollowDetectResult res;
    if (roi.empty()) return res;

    const int roiW = roi.cols, roiH = roi.rows;

    // ── Step 1: intensity gate mask ───────────────────────────────────────────
    cv::Mat gate_mask = computeIntensityGateMask(roi, SEED_INTENSITY, SEED_THRESHOLD);

    // ── Step 2: connected components with stats ───────────────────────────────
    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(gate_mask, labels, stats, centroids, 8, CV_32S);

    // ── Step 3: no labels found -> fail ───────────────────────────────────────
    if (n_labels <= 1) return res;

    // ── Step 4: morph-close delta mask ────────────────────────────────────────
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat closed_mask;
    cv::morphologyEx(gate_mask, closed_mask, cv::MORPH_CLOSE, close_kernel);
    cv::Mat delta_mask;
    cv::bitwise_xor(closed_mask, gate_mask, delta_mask);

    // ── Step 5: find the winning label by delta-pixel count within its bbox ──
    const cv::Rect img_rect(0, 0, roiW, roiH);
    int winning_label = -1;
    int best_delta_count = -1;
    for (int lbl = 1; lbl < n_labels; ++lbl) {
        cv::Rect bbox(stats.at<int>(lbl, cv::CC_STAT_LEFT),
                      stats.at<int>(lbl, cv::CC_STAT_TOP),
                      stats.at<int>(lbl, cv::CC_STAT_WIDTH),
                      stats.at<int>(lbl, cv::CC_STAT_HEIGHT));
        bbox &= img_rect;
        if (bbox.width <= 0 || bbox.height <= 0) continue;
        int delta_count = cv::countNonZero(delta_mask(bbox));
        if (delta_count > best_delta_count) {
            best_delta_count = delta_count;
            winning_label = lbl;
        }
    }
    if (winning_label < 0) return res;

    // ── Step 6: binary mask of the winning label only ─────────────────────────
    // Performance-only: each row of winning_mask depends only on the
    // corresponding row of labels, so this is split across kWorkerThreads
    // row-ranges via parallelForRows with no behavioral change.
    cv::Mat winning_mask = cv::Mat::zeros(roiH, roiW, CV_8U);
    parallelForRows(roiH, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const int* lrow = labels.ptr<int>(y);
            auto*      mrow = winning_mask.ptr<uchar>(y);
            for (int x = 0; x < roiW; ++x)
                if (lrow[x] == winning_label) mrow[x] = 255;
        }
    });

    // ── Step 7: pad by 1px on all sides ───────────────────────────────────────
    cv::Mat padded;
    cv::copyMakeBorder(winning_mask, padded, 1, 1, 1, 1,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    const int pw = padded.cols, ph = padded.rows;

    // ── Step 8: flood-fill from each zero-valued corner -> exterior mask ─────
    // Performance-only: each corner's flood-fill operates on its own
    // independent clone of `padded` and produces its own `filled` mask with
    // no shared mutable state, so the 4 flood-fills are dispatched in
    // parallel via std::async. The bitwise_or merge into exterior_padded is
    // still performed strictly sequentially, in the same fixed corner order
    // as before, so the final exterior mask is bit-for-bit identical.
    cv::Mat exterior_padded = cv::Mat::zeros(padded.size(), CV_8U);
    const cv::Point corners[4] = { {0, 0}, {pw - 1, 0}, {0, ph - 1}, {pw - 1, ph - 1} };

    std::vector<std::future<cv::Mat>> fill_futures;
    fill_futures.reserve(4);
    for (const auto& c : corners) {
        if (padded.at<uchar>(c.y, c.x) != 0) {
            fill_futures.emplace_back(); // empty future -> skip below
            continue;
        }
        fill_futures.push_back(std::async(std::launch::async, [&padded, c]() -> cv::Mat {
            cv::Mat work = padded.clone();
            cv::floodFill(work, c, cv::Scalar(200));
            cv::Mat filled;
            cv::compare(work, 200, filled, cv::CMP_EQ); // 0/255 mask of this fill
            return filled;
        }));
    }
    for (auto& fut : fill_futures) {
        if (!fut.valid()) continue;
        cv::Mat filled = fut.get();
        cv::bitwise_or(exterior_padded, filled, exterior_padded);
    }
    // Crop the 1px border back off.
    cv::Mat exterior = exterior_padded(cv::Rect(1, 1, roiW, roiH)).clone();

    // ── Step 9: hole_pixels = NOT(exterior) AND NOT(winning_label_mask) ──────
    cv::Mat not_exterior, not_winning, hole_pixels;
    cv::bitwise_not(exterior, not_exterior);
    cv::bitwise_not(winning_mask, not_winning);
    cv::bitwise_and(not_exterior, not_winning, hole_pixels);

    // ── Step 10: connected components on hole_pixels, area > 10 ──────────────
    cv::Mat hole_labels;
    int n_hole_labels = cv::connectedComponents(hole_pixels, hole_labels, 8, CV_32S);

    std::vector<cv::Rect> hole_rects;
    if (n_hole_labels > 1) {
        std::vector<int> hcount(n_hole_labels, 0);
        std::vector<int> hx0(n_hole_labels, INT_MAX), hy0(n_hole_labels, INT_MAX);
        std::vector<int> hx1(n_hole_labels, INT_MIN), hy1(n_hole_labels, INT_MIN);

        // Performance-only: row ranges are scanned in parallel into
        // thread-local partial accumulators (one set of hcount/hx0/hy0/hx1/hy1
        // per worker, sized n_hole_labels), which are then reduced into the
        // shared arrays above via min/max/sum after all workers join. Since
        // min/max/sum reduction is associative and commutative, the final
        // values are identical to the single-threaded accumulation regardless
        // of how rows were split or thread scheduling order.
        {
            const int n_threads = std::min(kWorkerThreads, std::max(1, roiH));
            const int chunk = (roiH + n_threads - 1) / std::max(1, n_threads);

            struct Partial {
                std::vector<int> count, x0, y0, x1, y1;
                explicit Partial(int n) : count(n, 0), x0(n, INT_MAX), y0(n, INT_MAX),
                                          x1(n, INT_MIN), y1(n, INT_MIN) {}
            };
            std::vector<Partial> partials;
            partials.reserve(std::max(1, n_threads));

            auto scanRows = [&](Partial& p, int y0r, int y1r) {
                for (int y = y0r; y < y1r; ++y) {
                    const int* lrow = hole_labels.ptr<int>(y);
                    for (int x = 0; x < roiW; ++x) {
                        int lbl = lrow[x];
                        if (lbl == 0) continue;
                        p.count[lbl]++;
                        if (x < p.x0[lbl]) p.x0[lbl] = x;
                        if (x > p.x1[lbl]) p.x1[lbl] = x;
                        if (y < p.y0[lbl]) p.y0[lbl] = y;
                        if (y > p.y1[lbl]) p.y1[lbl] = y;
                    }
                }
            };

            for (int t = 0; t < std::max(1, n_threads); ++t)
                partials.emplace_back(n_hole_labels);

            std::vector<std::thread> hole_workers;
            hole_workers.reserve(std::max(0, n_threads - 1));
            for (int t = 1; t < n_threads; ++t) {
                int yr0 = std::min(roiH, t * chunk);
                int yr1 = std::min(roiH, yr0 + chunk);
                if (yr0 >= yr1) continue;
                hole_workers.emplace_back(scanRows, std::ref(partials[t]), yr0, yr1);
            }
            int yr1_first = std::min(roiH, chunk);
            scanRows(partials[0], 0, yr1_first);
            for (auto& w : hole_workers) w.join();

            for (auto& p : partials) {
                for (int lbl = 1; lbl < n_hole_labels; ++lbl) {
                    hcount[lbl] += p.count[lbl];
                    hx0[lbl] = std::min(hx0[lbl], p.x0[lbl]);
                    hy0[lbl] = std::min(hy0[lbl], p.y0[lbl]);
                    hx1[lbl] = std::max(hx1[lbl], p.x1[lbl]);
                    hy1[lbl] = std::max(hy1[lbl], p.y1[lbl]);
                }
            }
        }

        for (int lbl = 1; lbl < n_hole_labels; ++lbl) {
            if (hcount[lbl] > m_holeCcMinArea && hx0[lbl] != INT_MAX) {
                hole_rects.emplace_back(hx0[lbl], hy0[lbl],
                                        hx1[lbl] - hx0[lbl] + 1,
                                        hy1[lbl] - hy0[lbl] + 1);
            }
        }
    }

    // ── Step 11: no qualifying hole component -> fail ─────────────────────────
    if (hole_rects.empty()) {
        if (logsActive()) LOGD("[BALL] rejected: no qualifying hole components (n_hole_labels=%d)", n_hole_labels);
        return res;
    }

    // ── Step 12: cumulative bounding rect (union of all hole-component rects) ─
    cv::Rect cumulative = hole_rects[0];
    for (size_t i = 1; i < hole_rects.size(); ++i) cumulative |= hole_rects[i];

    // ── Step 13: validity guard on component count ────────────────────────────
    double cumulative_area = static_cast<double>(cumulative.width) *
                             static_cast<double>(cumulative.height);
    double max_allowed_components = std::max(4.0, cumulative_area / 150.0);
    if (static_cast<double>(hole_rects.size()) > max_allowed_components) {
        if (logsActive()) LOGD("[BALL] rejected: fragment_count=%zu > max_allowed=%.1f", hole_rects.size(), max_allowed_components);
        return res;
    }

    // ── Step 14: pad cumulative rect by m_holePad, clamped to image bounds ──────
    int rx0 = std::max(0,     cumulative.x - m_holePad);
    int ry0 = std::max(0,     cumulative.y - m_holePad);
    int rx1 = std::min(roiW,  cumulative.x + cumulative.width  + m_holePad);
    int ry1 = std::min(roiH,  cumulative.y + cumulative.height + m_holePad);
    if (rx1 <= rx0 || ry1 <= ry0) return res;
    const int padded_w = rx1 - rx0, padded_h = ry1 - ry0;

    // ── Step 15: aspect ratio guard ────────────────────────────────────────────
    float aspect = static_cast<float>(std::max(padded_w, padded_h)) /
                   static_cast<float>(std::max(1, std::min(padded_w, padded_h)));
    if (aspect > 1.6f) {
        if (logsActive()) LOGD("[BALL] rejected: aspect=%.2f > 1.6 (padded_w=%d padded_h=%d)", aspect, padded_w, padded_h);
        return res;
    }

    // ── Step 16: center = rect center, radius = half the diagonal ────────────
    cv::Point2f center(static_cast<float>(rx0) + static_cast<float>(padded_w) * 0.5f,
                       static_cast<float>(ry0) + static_cast<float>(padded_h) * 0.5f);
    float radius = 0.5f * std::hypot(static_cast<float>(padded_w),
                                     static_cast<float>(padded_h));

    // Range gate: reject if the half-diagonal falls outside the plausible
    // ball-size window set by the scaled HOLE_DIAG_MIN/MAX constants.
    if (radius < static_cast<float>(m_holeDiagMin) ||
        radius > static_cast<float>(m_holeDiagMax)) {
        if (logsActive()) LOGD("[BALL] rejected: radius=%.1f outside [%d,%d]", radius, m_holeDiagMin, m_holeDiagMax);
        return res; // outside plausible ball-size range
    }

    // ── Step 17: return result ────────────────────────────────────────────────
    res.hollow_center_orig = center;
    res.hollow_radius_orig = radius;
    if (logsActive()) LOGD("[BALL] ACCEPTED center=(%.1f,%.1f) radius=%.1f", center.x, center.y, radius);
    res.valid = true;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildErasePadFull
// Erases only the hollow-ball circle from the full proc_roi. The erase
// radius is derived per-frame from the ball radius detectHollowBall actually
// measured this frame (erase_radius = max(2, round(detected_ball_radius))),
// rather than from a fixed constructor-time constant.
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat PipelineEngine::buildErasePadFull(
        const cv::Mat& src,
        cv::Point2f hollow_c,
        int erase_radius)
{
    cv::Mat out = src.clone();
    cv::circle(out,
               cv::Point(static_cast<int>(std::round(hollow_c.x)),
                         static_cast<int>(std::round(hollow_c.y))),
               erase_radius,
               cv::Scalar(0, 0, 0), -1);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeIntensityGateMask
//
// Performance-only change: the original single-threaded scalar double-loop is
// preserved byte-for-byte as computeIntensityGateMaskRowScalar (used both as
// the non-NEON fallback and for any row remainder NEON can't vector-process),
// and is now run across kWorkerThreads row-ranges in parallel via
// parallelForRows since each row's output is fully independent of every
// other row (disjoint mask rows, no shared state). On ARM targets, the
// per-pixel BGR-to-luminance weighted sum + inclusive-range compare is also
// vectorized with NEON float intrinsics rather than fixed-point, since the
// inclusive [lo,hi] compare is sensitive to rounding and SEED_THRESHOLD is
// only ±2. Note vmlaq_f32 may use a fused multiply-add on some ARM cores,
// which can differ from the scalar path's separate multiply-then-add by up
// to 1 ULP in principle; given pixel intensities are 8-bit-derived values far
// from the lo/hi boundary at float precision, this is functionally
// equivalent to the scalar path within standard floating-point tolerance,
// not necessarily bit-identical in every possible input.
// ─────────────────────────────────────────────────────────────────────────────

static inline void computeIntensityGateMaskRowScalar(
        const cv::Mat& bgr, cv::Mat& mask, int y0, int y1, float lo, float hi)
{
    const int cols = bgr.cols;
    for (int y = y0; y < y1; ++y) {
        const auto* row  = bgr.ptr<uchar>(y);
        auto*       mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            auto B = static_cast<float>(row[x * 3 + 0]);
            auto G = static_cast<float>(row[x * 3 + 1]);
            auto R = static_cast<float>(row[x * 3 + 2]);
            float intensity = 0.299f * R + 0.587f * G + 0.114f * B;
            mrow[x] = (intensity >= lo && intensity <= hi) ? 255u : 0u;
        }
    }
}

#if defined(__ARM_NEON) || defined(__aarch64__)
// Performance-only: NEON float-intrinsic version of the same per-pixel
// weighted-sum + inclusive-range compare as the scalar path above. Processes
// 8 pixels per iteration; any remaining (< 8) pixels at the end of a row
// fall back to the scalar helper so output is identical to the scalar path
// for every pixel, including row lengths not divisible by 8.
static inline void computeIntensityGateMaskRowNeon(
        const cv::Mat& bgr, cv::Mat& mask, int y0, int y1, float lo, float hi)
{    const int cols = bgr.cols;
    const int cols8 = cols - (cols % 8);
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    const float32x4_t wB  = vdupq_n_f32(0.114f);
    const float32x4_t wG  = vdupq_n_f32(0.587f);
    const float32x4_t wR  = vdupq_n_f32(0.299f);

    for (int y = y0; y < y1; ++y) {
        const auto* row  = bgr.ptr<uchar>(y);
        auto*       mrow = mask.ptr<uchar>(y);
        int x = 0;
        for (; x < cols8; x += 8) {
            // Deinterleave 8 BGR pixels (24 bytes) into separate B/G/R uint8 lanes.
            uint8x8x3_t bgr8 = vld3_u8(row + x * 3);

            // Lower 4 pixels.
            uint16x8_t Bw_lo = vmovl_u8(bgr8.val[0]);
            uint16x8_t Gw_lo = vmovl_u8(bgr8.val[1]);
            uint16x8_t Rw_lo = vmovl_u8(bgr8.val[2]);

            float32x4_t Bf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(Bw_lo)));
            float32x4_t Gf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(Gw_lo)));
            float32x4_t Rf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(Rw_lo)));
            float32x4_t Bf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(Bw_lo)));
            float32x4_t Gf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(Gw_lo)));
            float32x4_t Rf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(Rw_lo)));

            float32x4_t i0 = vmlaq_f32(vmlaq_f32(vmulq_f32(Rf0, wR), Gf0, wG), Bf0, wB);
            float32x4_t i1 = vmlaq_f32(vmlaq_f32(vmulq_f32(Rf1, wR), Gf1, wG), Bf1, wB);

            uint32x4_t ge0 = vcgeq_f32(i0, vlo);
            uint32x4_t le0 = vcleq_f32(i0, vhi);
            uint32x4_t pass0 = vandq_u32(ge0, le0);
            uint32x4_t ge1 = vcgeq_f32(i1, vlo);
            uint32x4_t le1 = vcleq_f32(i1, vhi);
            uint32x4_t pass1 = vandq_u32(ge1, le1);

            uint16x4_t p0_16 = vmovn_u32(pass0);
            uint16x4_t p1_16 = vmovn_u32(pass1);
            uint16x8_t p_16  = vcombine_u16(p0_16, p1_16);
            uint8x8_t  p_8   = vmovn_u16(p_16); // 0x00 or 0xFF per lane already

            vst1_u8(mrow + x, p_8);
        }
        // Remainder pixels for this row (row width not divisible by 8).
        if (x < cols) {
            for (; x < cols; ++x) {
                auto B = static_cast<float>(row[x * 3 + 0]);
                auto G = static_cast<float>(row[x * 3 + 1]);
                auto R = static_cast<float>(row[x * 3 + 2]);
                float intensity = 0.299f * R + 0.587f * G + 0.114f * B;
                mrow[x] = (intensity >= lo && intensity <= hi) ? 255u : 0u;
            }
        }
    }
}
#endif // __ARM_NEON || __aarch64__

cv::Mat PipelineEngine::computeIntensityGateMask(
        const cv::Mat& bgr, float seed_intensity, int threshold)
{
    cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_8U);
    float lo = seed_intensity - static_cast<float>(threshold);
    float hi = seed_intensity + static_cast<float>(threshold);

    // Performance-only: row ranges are fully independent (each thread writes
    // only its own disjoint slice of mask rows), so this is split across
    // kWorkerThreads via parallelForRows with no behavioral change.
    parallelForRows(bgr.rows, [&](int y0, int y1) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        computeIntensityGateMaskRowNeon(bgr, mask, y0, y1, lo, hi);
#else
        computeIntensityGateMaskRowScalar(bgr, mask, y0, y1, lo, hi);
#endif
    });

    return mask;
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// findSeedClusters  — min_pixels filter only; Bessel-corrected covariance.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<SeedCluster> PipelineEngine::findSeedClusters(
        const cv::Mat& candidate_mask,
        int min_pixels)
{
    cv::Mat labels;
    int num_labels = cv::connectedComponents(candidate_mask, labels, 8, CV_32S);
    if (num_labels <= 1) return {};

    std::vector<std::vector<cv::Point>> label_pixels(num_labels);
    for (int y = 0; y < labels.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int lbl = lrow[x];
            if (lbl > 0) label_pixels[lbl].emplace_back(x, y);
        }
    }

    std::vector<SeedCluster> result;
    result.reserve(num_labels - 1);

    for (int lbl = 1; lbl < num_labels; ++lbl) {
        auto& pix = label_pixels[lbl];
        if (static_cast<int>(pix.size()) < min_pixels) continue;

        SeedCluster sc;
        sc.pixels = pix;

        // Centroid
        double sx = 0, sy = 0;
        for (const auto& p : pix) { sx += p.x; sy += p.y; }
        auto n = static_cast<double>(pix.size());
        sc.centroid = cv::Point2f(static_cast<float>(sx / n),
                                  static_cast<float>(sy / n));

        // Bessel-corrected sample covariance
        double cxx = 0, cxy = 0, cyy = 0;
        float  cx  = sc.centroid.x, cy_f = sc.centroid.y;
        for (const auto& p : pix) {
            double dx = static_cast<double>(p.x) - cx;
            double dy = static_cast<double>(p.y) - cy_f;
            cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
        }
        if (n > 1.0) {
            cxx /= (n - 1.0);
            cxy /= (n - 1.0);
            cyy /= (n - 1.0);
        }

        // Eigen-decomposition of 2×2 symmetric matrix [[cxx,cxy],[cxy,cyy]]
        double trace = cxx + cyy;
        double disc  = std::sqrt(std::max(0.0, (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
        double lam1  = 0.5 * (trace + disc);

        sc.major_axis_length = 2.f * static_cast<float>(std::sqrt(lam1));
        sc.orientation_deg   = static_cast<float>(
                std::atan2(2.0 * cxy, cxx - cyy) * 0.5 * 180.0 / CV_PI);

        result.push_back(std::move(sc));
    }

    std::sort(result.begin(), result.end(),
              [](const SeedCluster& a, const SeedCluster& b){
                  return a.major_axis_length > b.major_axis_length;
              });
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// findStripCenterline  — pure PCA-projection, no aura/RANSAC.
// ─────────────────────────────────────────────────────────────────────────────

RidgeResult PipelineEngine::findStripCenterline(
        const SeedCluster& cluster,
        cv::Point2f hollow_c)
{
    RidgeResult result;

    if (cluster.pixels.size() < 2)       return result;
    if (cluster.major_axis_length < 4.f) return result;

    float angle_rad = cluster.orientation_deg * PI_F / 180.f;
    cv::Point2f evec = pt_norm(cv::Point2f(std::cos(angle_rad), std::sin(angle_rad)));
    cv::Point2f centroid = cluster.centroid;

    // Project every pixel onto evec to find the extremal endpoints
    float t_min = 1e9f, t_max = -1e9f;
    for (const auto& p : cluster.pixels) {
        float t = pt_dot(pt_sub(cv::Point2f(static_cast<float>(p.x),
                                            static_cast<float>(p.y)),
                                centroid), evec);
        t_min = std::min(t_min, t);
        t_max = std::max(t_max, t);
    }

    cv::Point2f pt1 = pt_add(centroid, pt_scale(evec, t_min));
    cv::Point2f pt2 = pt_add(centroid, pt_scale(evec, t_max));

    // Order medial_near / medial_far by distance to hollow_c
    float dist1 = pt_mag(pt_sub(pt1, hollow_c));
    float dist2 = pt_mag(pt_sub(pt2, hollow_c));

    cv::Point2f medial_near, medial_far;
    if (dist1 <= dist2) {
        medial_near = pt1;
        medial_far  = pt2;
    } else {
        medial_near = pt2;
        medial_far  = pt1;
        evec = pt_scale(evec, -1.f);
    }

    result.extend_dir         = evec;
    result.medial_near        = medial_near;
    result.medial_far         = medial_far;
    result.source_line.pt1    = pt1;
    result.source_line.pt2    = pt2;
    result.source_line.length = cluster.major_axis_length;
    result.source_line.valid  = true;
    result.valid = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildATStripFromRidge
// Origin is ALWAYS hollow_c_crop clamped to image bounds.
// evec from rr.extend_dir (or raw line dir), oriented away from hollow ball.
// ─────────────────────────────────────────────────────────────────────────────

ATStrip PipelineEngine::buildATStripFromRidge(
        const EDLine& /*ln*/, const RidgeResult& rr,
        cv::Point2f hollow_c_proc, int /*hollow_r*/,
        int iw, int ih, bool is_cbc)
{
    ATStrip at;
    at.is_cue_ball_cut = is_cbc;

    cv::Point2f origin = hollow_c_proc;
    origin.x = std::clamp(origin.x, 0.f, static_cast<float>(iw - 1));
    origin.y = std::clamp(origin.y, 0.f, static_cast<float>(ih - 1));
    at.origin = origin;

    cv::Point2f evec;
    if (rr.valid) {
        evec = rr.extend_dir;
        cv::Point2f away = pt_norm(pt_sub(rr.medial_far, hollow_c_proc));
        if (pt_dot(evec, away) < 0.f) evec = pt_scale(evec, -1.f);
    } else {
        cv::Point2f raw_dir = pt_norm(pt_sub(rr.source_line.pt2, rr.source_line.pt1));
        if (pt_mag(raw_dir) < 1e-9f) return at;
        evec = raw_dir;
        cv::Point2f away = pt_norm(pt_sub(rr.source_line.pt2, hollow_c_proc));
        if (pt_dot(evec, away) < 0.f) evec = pt_scale(evec, -1.f);
    }

    at.evec      = evec;
    at.angle_deg = normAng(std::atan2(evec.y, evec.x) * 180.f / PI_F);
    at.valid     = true;
    return at;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeGhostTangentOrigin
// ─────────────────────────────────────────────────────────────────────────────

GhostTangentResult PipelineEngine::computeGhostTangentOrigin(
        cv::Point2f axis_origin, cv::Point2f evec, float gr,
        float pool_x1, float pool_y1, float pool_x2, float pool_y2)
{
    GhostTangentResult gtr;
    auto fgr = gr;

    float wall_top    = pool_y1 + fgr;
    float wall_bottom = pool_y2 - 1.0f - fgr;
    float wall_left   = pool_x1 + fgr;
    float wall_right  = pool_x2 - 1.0f - fgr;

    bool inside = (axis_origin.x >= wall_left   && axis_origin.x <= wall_right &&
                   axis_origin.y >= wall_top     && axis_origin.y <= wall_bottom);
    if (inside) {
        gtr.corrected = false; gtr.center = axis_origin; gtr.wall = -1;
        return gtr;
    }

    cv::Point2f back = pt_scale(evec, -1.0f);
    float best_t = 1e18f; int best_wall = -1;

    auto tryWall = [&](float t, cv::Point2f candidate, int wall_idx) {
        if (t < -1e-3f) return;
        bool x_ok = (candidate.x >= wall_left  - 1.0f && candidate.x <= wall_right  + 1.0f);
        bool y_ok = (candidate.y >= wall_top    - 1.0f && candidate.y <= wall_bottom + 1.0f);
        if (x_ok && y_ok && t < best_t) { best_t = t; best_wall = wall_idx; }
    };

    if (std::abs(back.y) > 1e-9f) {
        float t = (wall_top    - axis_origin.y) / back.y;
        tryWall(t, pt_add(axis_origin, pt_scale(back, t)), 0);
        t = (wall_bottom - axis_origin.y) / back.y;
        tryWall(t, pt_add(axis_origin, pt_scale(back, t)), 1);
    }
    if (std::abs(back.x) > 1e-9f) {
        float t = (wall_left  - axis_origin.x) / back.x;
        tryWall(t, pt_add(axis_origin, pt_scale(back, t)), 2);
        t = (wall_right - axis_origin.x) / back.x;
        tryWall(t, pt_add(axis_origin, pt_scale(back, t)), 3);
    }

    if (best_wall >= 0) {
        cv::Point2f center = pt_add(axis_origin, pt_scale(back, best_t));
        center.x = std::clamp(center.x, wall_left, wall_right);
        center.y = std::clamp(center.y, wall_top,  wall_bottom);
        gtr.corrected = true; gtr.center = center; gtr.wall = best_wall;
        return gtr;
    }

    cv::Point2f clamped = axis_origin;
    clamped.x = std::clamp(clamped.x, wall_left, wall_right);
    clamped.y = std::clamp(clamped.y, wall_top,  wall_bottom);
    float dt = clamped.y - wall_top, db = wall_bottom - clamped.y;
    float dl = clamped.x - wall_left, dr = wall_right  - clamped.x;
    float mn = std::min({dt, db, dl, dr});
    int closest = 0;
    if      (mn == db) closest = 1;
    else if (mn == dl) closest = 2;
    else if (mn == dr) closest = 3;
    gtr.corrected = true; gtr.center = clamped; gtr.wall = closest;
    return gtr;
}

// ─────────────────────────────────────────────────────────────────────────────
// correctATGhostOrigin
// ─────────────────────────────────────────────────────────────────────────────

cv::Point2f PipelineEngine::correctATGhostOrigin(
        cv::Point2f axis_origin, cv::Point2f evec, float gr,
        float pool_x1, float pool_y1, float pool_x2, float pool_y2,
        int& tangent_wall_out)
{
    GhostTangentResult gtr = computeGhostTangentOrigin(
            axis_origin, evec, gr, pool_x1, pool_y1, pool_x2, pool_y2);
    tangent_wall_out = gtr.wall;
    return gtr.center;
}

// ─────────────────────────────────────────────────────────────────────────────
// restoreAT
// ─────────────────────────────────────────────────────────────────────────────

ATStrip PipelineEngine::restoreAT(ATStrip at, cv::Point2f o) {
    at.origin = pt_add(at.origin, o);
    return at;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

float PipelineEngine::normAng(float d) {
    while (d >  180.f) d -= 360.f;
    while (d < -180.f) d += 360.f;
    return d;
}

Bounds PipelineEngine::makeBounds(int x1, int y1, int x2, int y2, int inset) {
    return { static_cast<float>(y1 + inset),
             static_cast<float>(y2 - 1 - inset),
             static_cast<float>(x1 + inset),
             static_cast<float>(x2 - 1 - inset) };
}

cv::Point2f PipelineEngine::reflDir(cv::Point2f d, int w) {
    return (w == 0 || w == 1) ? cv::Point2f(d.x, -d.y) : cv::Point2f(-d.x, d.y);
}

cv::Point2f PipelineEngine::wallNorm(int w) {
    switch (w) {
        case 0:  return { 0,  1};
        case 1:  return { 0, -1};
        case 2:  return { 1,  0};
        default: return {-1,  0};
    }
}

int PipelineEngine::rayHit(
        cv::Point2f o, cv::Point2f d, const Bounds& b, int skip,
        float& tout, cv::Point2f& gout)
{
    if (!b.valid()) return -1;
    const float e = 1e-6f;
    float wt[4] = {1e18f, 1e18f, 1e18f, 1e18f};
    if (std::abs(d.y) > e) { wt[0] = (b.top    - o.y) / d.y; wt[1] = (b.bottom - o.y) / d.y; }
    if (std::abs(d.x) > e) { wt[2] = (b.left   - o.x) / d.x; wt[3] = (b.right  - o.x) / d.x; }
    int bw = -1; float bt = 1e18f;
    for (int w = 0; w < 4; ++w) {
        if (w == skip || wt[w] <= e) continue;
        float hx = o.x + d.x * wt[w], hy = o.y + d.y * wt[w];
        if (hx < b.left - 0.5f || hx > b.right  + 0.5f ||
            hy < b.top  - 0.5f || hy > b.bottom + 0.5f) continue;
        if (wt[w] < bt) { bt = wt[w]; bw = w; }
    }
    if (bw < 0) return -1;
    tout = bt;
    gout = {o.x + d.x * bt, o.y + d.y * bt};
    return bw;
}

bool PipelineEngine::inPocket(cv::Point2f p, const std::vector<Pocket>& pockets) {
    return std::any_of(pockets.begin(), pockets.end(),
                       [&p](const Pocket& pk){
                           return pt_mag(pt_sub(p, pk.center)) <= pk.radius;
                       });
}

float PipelineEngine::rayPocketT(
        cv::Point2f o, cv::Point2f d,
        const std::vector<Pocket>& pockets, float maxT)
{
    if (pockets.empty()) return -1.f;
    const float STEP = 1.0f;
    float t = STEP;
    while (t <= maxT + STEP) {
        float tc = std::min(t, maxT);
        if (inPocket(pt_add(o, pt_scale(d, tc)), pockets))
            return std::max(0.f, tc - STEP);
        t += STEP;
    }
    return -1.f;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildShots
// ─────────────────────────────────────────────────────────────────────────────

std::vector<ShotEx> PipelineEngine::buildShots(
        const std::vector<ATStrip>& ats,
        const Bounds& ptBounds,
        const std::vector<Pocket>& pockets,
        int iw, int ih,
        int maxr_cbc, int maxr_tgt)
{
    std::vector<ShotEx> result;
    if (!ptBounds.valid()) return result;

    for (const auto& strip : ats) {
        if (!strip.valid) continue;
        cv::Point2f o = strip.origin, d = pt_norm(strip.evec);
        if (pt_mag(d) < 1e-9f) continue;
        if (inPocket(o, pockets)) continue;
        if (o.x < ptBounds.left - 1.f || o.x > ptBounds.right  + 1.f ||
            o.y < ptBounds.top  - 1.f || o.y > ptBounds.bottom + 1.f) continue;

        int strip_max_refl = strip.is_cue_ball_cut ? maxr_cbc : maxr_tgt;
        int skipWall = -1;

        for (int refl = 0; refl <= strip_max_refl; ++refl) {
            float wallT = 0.f; cv::Point2f wallPt; int hitWall = -1;
            hitWall = rayHit(o, d, ptBounds, skipWall, wallT, wallPt);

            if (hitWall < 0) {
                const float ef = 1e-7f;
                float fb_t = 1e18f; int fb_w = -1; cv::Point2f fb_pt;
                auto tryFB = [&](float wc, float oc, float dc,
                                 float clo, float chi,
                                 float co, float cd, int ww) {
                    if (std::abs(dc) < ef) return;
                    float t = (wc - oc) / dc; if (t <= ef) return;
                    float hit = co + cd * t;
                    if (hit < clo - 0.5f || hit > chi + 0.5f) return;
                    if (t < fb_t) { fb_t = t; fb_w = ww; fb_pt = pt_add(o, pt_scale(d, t)); }
                };
                tryFB(ptBounds.top,    o.y, d.y, ptBounds.left,  ptBounds.right,  o.x, d.x, 0);
                tryFB(ptBounds.bottom, o.y, d.y, ptBounds.left,  ptBounds.right,  o.x, d.x, 1);
                tryFB(ptBounds.left,   o.x, d.x, ptBounds.top,   ptBounds.bottom, o.y, d.y, 2);
                tryFB(ptBounds.right,  o.x, d.x, ptBounds.top,   ptBounds.bottom, o.y, d.y, 3);
                if (fb_w < 0) break;
                wallT = fb_t; wallPt = fb_pt; hitWall = fb_w;
            }

            float pT = rayPocketT(o, d, pockets, wallT);
            if (pT >= 0.f) {
                ShotEx seg;
                seg.from = o; seg.to = pt_add(o, pt_scale(d, pT));
                seg.dir = d; seg.wall = -1; seg.pocket_stop = true;
                result.push_back(seg);
                break;
            }
            ShotEx seg;
            seg.from = o; seg.to = wallPt;
            seg.dir = d; seg.wall = hitWall; seg.pocket_stop = false;
            result.push_back(seg);
            if (refl == strip_max_refl) break;
            d = reflDir(d, hitWall);
            o = wallPt;
            skipWall = hitWall;
        }
    }
    return result;
}

std::vector<Pocket> PipelineEngine::makePockets(
        int ox, int oy, int w, int h, float pr, float ns_shift)
{
    float diag = m_cornerPocketShift * 0.70710678f;
    float mx = static_cast<float>(ox) + static_cast<float>(w - 1) * 0.5f;
    return {
            {{static_cast<float>(ox) - diag,               static_cast<float>(oy) - diag},          pr},
            {{static_cast<float>(ox + w - 1) + diag,       static_cast<float>(oy) - diag},           pr},
            {{static_cast<float>(ox) - diag,               static_cast<float>(oy + h - 1) + diag},   pr},
            {{static_cast<float>(ox + w - 1) + diag,       static_cast<float>(oy + h - 1) + diag},   pr},
            {{mx, static_cast<float>(oy)        - ns_shift},                                           pr},
            {{mx, static_cast<float>(oy + h-1)  + ns_shift},                                          pr}
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// countPassingPocketSquares
//
// Samples 6 small square crops — one at each of the 4 diagonal corners and
// one at each of the 2 mid-rail positions — and counts how many have a mean
// brightness in [psq_bright_min, psq_bright_max]. Used as a coarse scene
// check before running the full pipeline (Stage 0). The 6 sample evaluations
// are split across two threads (index split over [0,6), same chunking pattern
// as parallelForRows but indexed by sample rather than row), since each crop
// is independent and the sampling work is small but not negligible.
// ─────────────────────────────────────────────────────────────────────────────

int PipelineEngine::countPassingPocketSquares(
        const cv::Mat& roi,
        int ptOX, int ptOY, int ptW, int ptH,
        int psq_corner_offset, int psq_mid_offset, int psq_size,
        float psq_bright_min, float psq_bright_max)
{
    const float diag = static_cast<float>(psq_corner_offset) * 0.70710678f;
    const int half = psq_size / 2;
    const int iw = roi.cols, ih = roi.rows;

    cv::Point2f centers[6];
    centers[0] = {static_cast<float>(ptOX) - diag,                        static_cast<float>(ptOY) - diag};
    centers[1] = {static_cast<float>(ptOX + ptW) + diag,                  static_cast<float>(ptOY) - diag};
    centers[2] = {static_cast<float>(ptOX) - diag,                        static_cast<float>(ptOY + ptH) + diag};
    centers[3] = {static_cast<float>(ptOX + ptW) + diag,                  static_cast<float>(ptOY + ptH) + diag};
    float mid_x = static_cast<float>(ptOX) + static_cast<float>(ptW) * 0.5f;
    centers[4] = {mid_x, static_cast<float>(ptOY) - static_cast<float>(psq_mid_offset)};
    centers[5] = {mid_x, static_cast<float>(ptOY + ptH) + static_cast<float>(psq_mid_offset)};

    bool passed[6] = {};

    // Performance-only: the 6 crops are fully independent (disjoint read-only
    // roi access, disjoint passed[] writes), so they are split across 2 threads
    // with a contiguous index split over [0,6) — same chunking pattern as
    // parallelForRows. Each crop is tiny, so no NEON path is added here.
    auto sampleRange = [&](int s0, int s1) {
        for (int s = s0; s < s1; ++s) {
            int cx = static_cast<int>(std::round(centers[s].x));
            int cy = static_cast<int>(std::round(centers[s].y));
            int x0 = std::max(0, cx - half);
            int y0 = std::max(0, cy - half);
            int x1 = std::min(iw, cx + half + 1);
            int y1 = std::min(ih, cy + half + 1);
            if ((x1 - x0) * (y1 - y0) < 4) { passed[s] = false; continue; }
            double sum = 0.0;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                const auto* row = roi.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) {
                    double B = row[x * 3 + 0];
                    double G = row[x * 3 + 1];
                    double R = row[x * 3 + 2];
                    sum += 0.299 * R + 0.587 * G + 0.114 * B;
                    ++count;
                }
            }
            float mean = static_cast<float>(sum / count);
            passed[s] = (mean >= psq_bright_min && mean <= psq_bright_max);
        }
    };

    constexpr int kSamples = 6;
    constexpr int kSampleThreads = 2;
    constexpr int kChunk = (kSamples + kSampleThreads - 1) / kSampleThreads;
    std::thread worker(sampleRange, kChunk, kSamples);
    sampleRange(0, kChunk);
    worker.join();

    int total = 0;
    for (bool p : passed) if (p) ++total;
    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// processFrame
// ─────────────────────────────────────────────────────────────────────────────

void PipelineEngine::processFrame(void* pixelData, size_t byteCount) {
    auto t_total_start = HRC::now();
    m_frameCount++;

    HRC::time_point t_ds_start,    t_ds_end;
    HRC::time_point t_epc_start,   t_epc_end;
    HRC::time_point t_seed_start,  t_seed_end;
    HRC::time_point t_strip_start, t_strip_end;
    HRC::time_point t_cls_start,   t_cls_end;
    HRC::time_point t_at_start,    t_at_end;
    HRC::time_point t_ca_strip_start, t_ca_strip_end;
    HRC::time_point t_rest_start,  t_rest_end;
    auto now0 = HRC::now();
    t_ds_start = t_ds_end = t_epc_start = t_epc_end = now0;
    t_seed_start = t_seed_end = t_strip_start = t_strip_end = now0;
    t_cls_start  = t_cls_end  = t_at_start   = t_at_end    = now0;
    t_ca_strip_start = t_ca_strip_end = t_rest_start = t_rest_end  = now0;

    const int rx1 = m_roiX1.load(), ry1 = m_roiY1.load();
    const int rx2 = m_roiX2.load(), ry2 = m_roiY2.load();
    const float gr  = m_ghostRadius.load();
    const int pr  = m_pocketR.load(), pns = m_pocketNS.load();
    const bool cushion  = m_cushionShots.load();
    const int  maxr_cbc = cushion ? m_maxr_cbc.load() : 0;
    const int  maxr_tgt = cushion ? m_maxr_tgt.load() : 0;
    const int  px1 = m_poolX1.load(), py1 = m_poolY1.load();
    const int  px2 = m_poolX2.load(), py2 = m_poolY2.load();
    const bool poolValid = (px2 > px1 + 10 && py2 > py1 + 10);

    const int roiW = rx2 - rx1, roiH = ry2 - ry1;
    if (roiW <= 0 || roiH <= 0) {
        LOGE("processFrame: Invalid ROI (%d,%d,%d,%d)", rx1, ry1, rx2, ry2);
        return;
    }

    const size_t expectedBytes = static_cast<size_t>(m_screenW) * m_screenH * 4;
    if (byteCount < expectedBytes) {
        LOGE("processFrame: Buffer too small: %zu < %zu", byteCount, expectedBytes);
        return;
    }

    cv::Mat fullFrame(m_screenH, m_screenW, CV_8UC4, pixelData);
    cv::Rect roiRect(rx1, ry1, roiW, roiH);
    roiRect &= cv::Rect(0, 0, m_screenW, m_screenH);
    cv::Mat roiRgba = fullFrame(roiRect).clone();
    cv::Mat roi;
    cv::cvtColor(roiRgba, roi, cv::COLOR_RGBA2BGR);
    const int iw = roi.cols, ih = roi.rows;

    const int ptOX = poolValid ? (px1 - rx1) : 0;
    const int ptOY = poolValid ? (py1 - ry1) : 0;
    const int ptW  = poolValid ? (px2 - px1) : iw;
    const int ptH  = poolValid ? (py2 - py1) : ih;

    auto pockets = makePockets(ptOX, ptOY, ptW, ptH,
                               static_cast<float>(pr),
                               static_cast<float>(pns));

    float pool_local_x1 = poolValid ? static_cast<float>(px1 - rx1) : 0.f;
    float pool_local_y1 = poolValid ? static_cast<float>(py1 - ry1) : 0.f;
    float pool_local_x2 = poolValid ? static_cast<float>(px2 - rx1) : static_cast<float>(iw);
    float pool_local_y2 = poolValid ? static_cast<float>(py2 - ry1) : static_cast<float>(ih);

    // ── Stage 0: PSQ scene check (overlaps Stage 1 on a worker thread) ───────
    // Launched before detectHollowBall so the pocket-square brightness samples
    // run concurrently with the hollow-ball detection on the calling thread.
    auto psq_future = std::async(std::launch::async, [&]() {
        return countPassingPocketSquares(roi, ptOX, ptOY, ptW, ptH,
                                         m_psqCornerOffset, m_psqMidOffset, m_psqSize,
                                         PSQ_BRIGHT_MIN, PSQ_BRIGHT_MAX);
    });

    // ── Stage 1: hollow ball detection ───────────────────────────────────────
    t_ds_start = HRC::now();
    HollowDetectResult ds_res = detectHollowBall(roi);
    t_ds_end = HRC::now();

    // Collect the PSQ result now that Stage 1 is done.
    int psq_passing = psq_future.get();

    // ── Stage 0 gate: reject the frame if the scene lacks enough pocket
    //    squares — checked before the hollow-ball validity gate since this
    //    is a coarser, scene-level filter.
    if (psq_passing < PSQ_MIN_PASSING) {
        LOGD("[F#%llu] pool table scene not found (psq=%d/%d)",
             static_cast<unsigned long long>(m_frameCount), psq_passing, PSQ_MIN_PASSING);
        std::lock_guard<std::mutex> lock(m_resultMutex);
        m_result.ats = {}; m_result.shots = {};
        m_result.pockets = pockets;
        m_result.roiX1 = rx1; m_result.roiY1 = ry1;
        m_result.roiX2 = rx2; m_result.roiY2 = ry2;
        m_result.poolX1 = poolValid ? px1 : rx1;
        m_result.poolY1 = poolValid ? py1 : ry1;
        m_result.poolX2 = poolValid ? px2 : rx2;
        m_result.poolY2 = poolValid ? py2 : ry2;
        m_result.ghostRadius = gr;
        m_result.maxr_cbc    = maxr_cbc;
        m_result.maxr_tgt    = maxr_tgt;
        m_result.pocketR     = pr;
        m_result.pocketNS    = pns;
        return;
    }

    if (!ds_res.valid) {
        std::lock_guard<std::mutex> lock(m_resultMutex);
        m_result.ats      = {};
        m_result.pockets  = pockets;
        m_result.shots    = {};
        m_result.roiX1 = rx1; m_result.roiY1 = ry1;
        m_result.roiX2 = rx2; m_result.roiY2 = ry2;
        m_result.poolX1 = poolValid ? px1 : rx1;
        m_result.poolY1 = poolValid ? py1 : ry1;
        m_result.poolX2 = poolValid ? px2 : rx2;
        m_result.poolY2 = poolValid ? py2 : ry2;
        m_result.ghostRadius = gr;
        m_result.maxr_cbc    = maxr_cbc;
        m_result.maxr_tgt    = maxr_tgt;
        m_result.pocketR     = pr;
        m_result.pocketNS    = pns;
        return;
    }

    // ── Detection block ───────────────────────────────────────────────────────
    struct DetectionResult {
        std::vector<ATStrip>     ats;
        std::vector<RidgeResult> ridges;
        std::vector<SeedCluster> clusters;
        int                      n_valid_lines = 0;
    };

    auto runDetectionBlock = [&](
            const cv::Mat& proc_roi_img,
            cv::Point2f hollow_c_in_proc,
            float hollow_radius_in_proc,
            cv::Mat& erase_pad_erased_out)
            -> DetectionResult
    {
        DetectionResult dr;
        const int pw = proc_roi_img.cols, ph = proc_roi_img.rows;

        // Stage 3: erase hollow-ball circle from full proc_roi.
        // erase_radius is derived per-frame from the ball radius detectHollowBall
        // actually measured this frame, not from a fixed constructor-time constant.
        int erase_radius = std::max(2, static_cast<int>(std::round(hollow_radius_in_proc)));
        t_epc_start = HRC::now();
        erase_pad_erased_out = buildErasePadFull(
                proc_roi_img, hollow_c_in_proc, erase_radius);
        t_epc_end = HRC::now();

        // Stage 4: intensity gate + seed clusters
        t_seed_start = HRC::now();
        cv::Mat candidate_mask = computeIntensityGateMask(
                erase_pad_erased_out, SEED_INTENSITY, SEED_THRESHOLD);
        dr.clusters = findSeedClusters(candidate_mask, MIN_CLUSTER_PIXELS);
        t_seed_end = HRC::now();

        // Stage 5: findStripCenterline per cluster (pure PCA)
        t_strip_start = HRC::now();
        dr.ridges.reserve(dr.clusters.size());
        for (const auto& sc : dr.clusters) {
            RidgeResult rr = findStripCenterline(sc, hollow_c_in_proc);
            dr.ridges.push_back(rr);
        }
        t_strip_end = HRC::now();

        // ── Stage 6: search-circle + auto-tuned min-length/collinearity gate ────
        // Replaces the old single fixed "major_axis_length >= MIN_STRIP_LEN_DETECT"
        // check with a multi-step gate that determines validity for ALL clusters
        // simultaneously via a small grid search over (collinearity tolerance,
        // min length) candidate pairs, picking the combination whose result is
        // stable across consecutive min-length steps.
        std::vector<int> valid_indices;
        {
            const int n_clusters = static_cast<int>(dr.clusters.size());

            // Step 1: per-cluster gate inputs ------------------------------------
            // Search-circle radius: fixed 130.0f (already in detect-scale units
            // on this Android build — no native-to-detect conversion factor is
            // applied), clamped so the circle never extends past image edges.
            float search_radius = 130.0f;
            search_radius = std::min({search_radius,
                                      hollow_c_in_proc.x,
                                      hollow_c_in_proc.y,
                                      static_cast<float>(pw - 1) - hollow_c_in_proc.x,
                                      static_cast<float>(ph - 1) - hollow_c_in_proc.y});
            search_radius = std::max(search_radius, 1.0f);

            std::vector<bool>  within_circle(n_clusters, false);
            std::vector<float> major_axis_length(n_clusters, 0.f);
            std::vector<float> collinear_angle_deg(n_clusters, 0.f);

            for (int i = 0; i < n_clusters; ++i) {
                major_axis_length[i] = dr.clusters[i].major_axis_length;
                const RidgeResult& rr = dr.ridges[i];
                if (!rr.valid) continue;

                float dist_near = pt_mag(pt_sub(rr.medial_near, hollow_c_in_proc));
                within_circle[i] = (dist_near <= search_radius);

                cv::Point2f vecA = pt_norm(pt_sub(rr.medial_near, hollow_c_in_proc));
                cv::Point2f vecB = pt_norm(pt_sub(rr.medial_far,  hollow_c_in_proc));
                float dotv = std::clamp(pt_dot(vecA, vecB), -1.f, 1.f);
                collinear_angle_deg[i] = std::acos(dotv) * 180.f / PI_F;
            }

            // Step 2: grid search over candidate thresholds ----------------------
            static const float kMinLenCandidates[8] = {12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f};
            static const float kTolCandidates[11]    = {2.f, 3.f, 1.f, 4.f, 0.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
            constexpr int kNumMinLen = 8;
            constexpr int kNumTol    = 11;

            auto evalCell = [&](float tol, float min_len) -> std::vector<int> {
                std::vector<int> passing;
                for (int i = 0; i < n_clusters; ++i) {
                    if (within_circle[i] &&
                        major_axis_length[i] >= min_len &&
                        collinear_angle_deg[i] <= tol)
                        passing.push_back(i);
                }
                return passing;
            };

            std::vector<std::vector<std::vector<int>>> grid_indices(
                    kNumTol, std::vector<std::vector<int>>(kNumMinLen));

            // Performance-only: each of the kNumTol=11 tolerance rows is fully
            // independent (every cell only reads within_circle/major_axis_length/
            // collinear_angle_deg, and writes only to its own grid_indices[ti]
            // slot), so the 88-cell grid is filled across kWorkerThreads threads
            // split by tolerance row instead of sequentially. Each thread writes
            // to disjoint ti indices only, so there are no data races. Cell
            // values themselves are computed exactly as before (evalCell is
            // unchanged), so results are identical regardless of thread
            // scheduling. The cell-comparison/stability search below (Step 3)
            // is NOT parallelized and remains strictly sequential, since it
            // depends on iteration order — only the independent cell-fill above
            // is parallelized.
            if (kNumTol >= 2) {
                const int n_threads = std::min(kWorkerThreads, kNumTol);
                const int chunk = (kNumTol + n_threads - 1) / n_threads;
                std::vector<std::thread> grid_workers;
                grid_workers.reserve(n_threads - 1);
                auto fillRows = [&](int ti0, int ti1) {
                    for (int ti = ti0; ti < ti1; ++ti)
                        for (int mi = 0; mi < kNumMinLen; ++mi)
                            grid_indices[ti][mi] = evalCell(kTolCandidates[ti], kMinLenCandidates[mi]);
                };
                for (int t = 1; t < n_threads; ++t) {
                    int ti0 = std::min(kNumTol, t * chunk);
                    int ti1 = std::min(kNumTol, ti0 + chunk);
                    if (ti0 >= ti1) continue;
                    grid_workers.emplace_back(fillRows, ti0, ti1);
                }
                int ti1_first = std::min(kNumTol, chunk);
                fillRows(0, ti1_first);
                for (auto& w : grid_workers) w.join();
            } else {
                for (int ti = 0; ti < kNumTol; ++ti)
                    for (int mi = 0; mi < kNumMinLen; ++mi)
                        grid_indices[ti][mi] = evalCell(kTolCandidates[ti], kMinLenCandidates[mi]);
            }

            // Step 3: stability search --------------------------------------------
            // Iterate tolerances in the exact order given (not sorted). For each
            // tolerance, walk min_length from index 0 (value 12) down to index 7
            // (value 5). The moment two consecutive min_length steps produce an
            // IDENTICAL passing-index-list AND that count is 1, 2, or 3, that is
            // the winning combination (use the larger min_length of the two).
            std::vector<int> winning_indices;
            bool found_stable = false;
            for (int ti = 0; ti < kNumTol && !found_stable; ++ti) {
                for (int mi = 1; mi < kNumMinLen; ++mi) {
                    const auto& prev_list = grid_indices[ti][mi - 1];
                    const auto& curr_list = grid_indices[ti][mi];
                    int count = static_cast<int>(curr_list.size());
                    if (static_cast<int>(prev_list.size()) == count &&
                        prev_list == curr_list &&
                        (count == 1 || count == 2 || count == 3)) {
                        winning_indices = prev_list; // larger min_length (closer to 12)
                        found_stable = true;
                        break;
                    }
                }
            }
            if (!found_stable) {
                // Fall back to the very first cell evaluated: tolerance=2, min_length=12.
                winning_indices = grid_indices[0][0];
            }

            // Step 4: apply the result --------------------------------------------
            std::vector<bool> is_winning(n_clusters, false);
            for (int idx : winning_indices) is_winning[idx] = true;
            for (int i = 0; i < n_clusters; ++i)
                if (!is_winning[i]) dr.ridges[i].valid = false;

            valid_indices = winning_indices;
        }
        dr.n_valid_lines = static_cast<int>(valid_indices.size());

        auto nearDist = [&](int ridx) -> float {
            return pt_mag(pt_sub(dr.ridges[ridx].medial_near, hollow_c_in_proc));
        };

        auto orientedDirAwayFromHollow = [&](int ridx) -> cv::Point2f {
            cv::Point2f d = dr.ridges[ridx].extend_dir;
            cv::Point2f away = pt_norm(pt_sub(dr.ridges[ridx].medial_far, hollow_c_in_proc));
            if (pt_dot(d, away) < 0.f) d = pt_scale(d, -1.f);
            return d;
        };

        // ── Shared CW/ACW axis-angle helpers for n=2 / n=3 classification ───────
        // Given the TGT ridge's oriented unit direction, build two perpendicular
        // unit axes (cw_axis/acw_axis). These replace the old cue-ball color
        // template test entirely.
        auto buildCwAcwAxes = [](cv::Point2f tgt_dir,
                                 cv::Point2f& cw_axis, cv::Point2f& acw_axis) {
            cw_axis  = cv::Point2f(-tgt_dir.y,  tgt_dir.x);
            acw_axis = cv::Point2f( tgt_dir.y, -tgt_dir.x);
        };

        // angle_to_axes for a candidate ridge's oriented direction vector
        // (its extend_dir, used as-is — NOT re-oriented away from the ball).
        auto angleToAxes = [](cv::Point2f line_dir,
                              cv::Point2f cw_axis, cv::Point2f acw_axis) -> float {
            cv::Point2f ln = pt_norm(line_dir);
            float dot_cw  = std::clamp(std::abs(pt_dot(ln, cw_axis)),  0.f, 1.f);
            float dot_acw = std::clamp(std::abs(pt_dot(ln, acw_axis)), 0.f, 1.f);
            float angle_to_cw  = std::acos(dot_cw)  * 180.f / PI_F;
            float angle_to_acw = std::acos(dot_acw) * 180.f / PI_F;
            return std::min(angle_to_cw, angle_to_acw);
        };

        const int detected_hollow_r = erase_radius;

        t_cls_start = HRC::now();

        switch (dr.n_valid_lines) {

            case 0:
                break;

            case 1: {
                float hx_screen = ds_res.hollow_center_orig.x + static_cast<float>(rx1);
                float hy_screen = ds_res.hollow_center_orig.y + static_cast<float>(ry1);
                bool near_pool_edge;
                if (poolValid) {
                    float dist_left   = hx_screen - static_cast<float>(px1);
                    float dist_right  = static_cast<float>(px2) - hx_screen;
                    float dist_top    = hy_screen - static_cast<float>(py1);
                    float dist_bottom = static_cast<float>(py2) - hy_screen;
                    near_pool_edge = (dist_left   < m_caFbEdgeThresh ||
                                      dist_right  < m_caFbEdgeThresh ||
                                      dist_top    < m_caFbEdgeThresh ||
                                      dist_bottom < m_caFbEdgeThresh);
                } else {
                    near_pool_edge =
                            (ds_res.hollow_center_orig.x < m_caFbEdgeThresh ||
                             ds_res.hollow_center_orig.x > static_cast<float>(iw - 1) - m_caFbEdgeThresh ||
                             ds_res.hollow_center_orig.y < m_caFbEdgeThresh ||
                             ds_res.hollow_center_orig.y > static_cast<float>(ih - 1) - m_caFbEdgeThresh);
                }
                if (near_pool_edge) {
                    t_ca_strip_start = HRC::now();

                    const RidgeResult& ridge = dr.ridges[valid_indices[0]];
                    // Defensive check — if the ridge isn't actually valid,
                    // produce no strip this frame.
                    if (ridge.valid) {
                        // incoming = normalize(hollow_c - ridge.medial_far):
                        // the raw, unreflected direction from medial_far toward
                        // the hollow ball (i.e. toward the cushion the ball is
                        // approaching). The physics engine will handle the
                        // bounce itself using its own restitution/friction model,
                        // so we hand it the pre-bounce state directly.
                        cv::Point2f diff = pt_sub(hollow_c_in_proc, ridge.medial_far);
                        if (pt_mag(diff) >= 1e-9f) {
                            cv::Point2f incoming = pt_norm(diff);

                            // origin = ridge.medial_far clamped to image bounds,
                            // matching the clamp buildATStripFromRidge applies for
                            // TGT/CBC strips.
                            cv::Point2f origin = ridge.medial_far;
                            origin.x = std::clamp(origin.x, 0.f, static_cast<float>(iw - 1));
                            origin.y = std::clamp(origin.y, 0.f, static_cast<float>(ih - 1));

                            ATStrip at_prebounce;
                            at_prebounce.origin          = origin;
                            at_prebounce.evec            = incoming;
                            at_prebounce.is_cue_ball_cut = false;
                            at_prebounce.valid           = true;
                            at_prebounce.angle_deg = normAng(
                                    std::atan2(incoming.y, incoming.x) * 180.f / PI_F);
                            dr.ats.push_back(at_prebounce);
                        }
                    }

                    t_ca_strip_end = HRC::now();
                }
                break;
            }

            case 2: {
                int idx0 = valid_indices[0], idx1 = valid_indices[1];
                float nd0 = nearDist(idx0), nd1 = nearDist(idx1);
                int tgt_ridx   = (nd0 > nd1) ? idx0 : idx1;
                int other_ridx = (nd0 > nd1) ? idx1 : idx0;

                cv::Point2f tgt_dir = orientedDirAwayFromHollow(tgt_ridx);
                cv::Point2f cw_axis, acw_axis;
                buildCwAcwAxes(tgt_dir, cw_axis, acw_axis);

                float angle_to_axes = angleToAxes(
                        dr.ridges[other_ridx].extend_dir, cw_axis, acw_axis);

                t_at_start = HRC::now();
                ATStrip at_tgt = buildATStripFromRidge(
                        dr.ridges[tgt_ridx].source_line, dr.ridges[tgt_ridx],
                        hollow_c_in_proc, detected_hollow_r, pw, ph, false);
                if (at_tgt.valid) dr.ats.push_back(at_tgt);

                if (angle_to_axes < 45.0f) {
                    ATStrip at_cbc = buildATStripFromRidge(
                            dr.ridges[other_ridx].source_line, dr.ridges[other_ridx],
                            hollow_c_in_proc, detected_hollow_r, pw, ph, true);
                    if (at_cbc.valid) dr.ats.push_back(at_cbc);
                }
                // angle_to_axes >= 45.0f: the other ridge is discarded for AT-strip
                // purposes this frame — it is not classified as CBC and does not
                // trigger any fallback path. Only the TGT ATStrip is produced.
                t_at_end = HRC::now();
                break;
            }

            case 3: {
                int tgt_ridx = valid_indices[0];
                float tgt_nd = nearDist(tgt_ridx);
                for (int vi : valid_indices) {
                    float nd = nearDist(vi);
                    if (nd > tgt_nd) { tgt_nd = nd; tgt_ridx = vi; }
                }

                std::vector<int> others;
                for (int vi : valid_indices) if (vi != tgt_ridx) others.push_back(vi);

                int ridA = others[0], ridB = others[1];

                cv::Point2f tgt_dir = orientedDirAwayFromHollow(tgt_ridx);
                cv::Point2f cw_axis, acw_axis;
                buildCwAcwAxes(tgt_dir, cw_axis, acw_axis);

                float angleA = angleToAxes(dr.ridges[ridA].extend_dir, cw_axis, acw_axis);
                float angleB = angleToAxes(dr.ridges[ridB].extend_dir, cw_axis, acw_axis);

                // Whichever of the two has the SMALLER angle_to_axes is CBC;
                // the other is discarded (no ATStrip, no fallback).
                int cbc_ridx = (angleA <= angleB) ? ridA : ridB;

                t_at_start = HRC::now();
                ATStrip at_tgt = buildATStripFromRidge(
                        dr.ridges[tgt_ridx].source_line, dr.ridges[tgt_ridx],
                        hollow_c_in_proc, detected_hollow_r, pw, ph, false);
                if (at_tgt.valid) dr.ats.push_back(at_tgt);

                ATStrip at_cbc = buildATStripFromRidge(
                        dr.ridges[cbc_ridx].source_line, dr.ridges[cbc_ridx],
                        hollow_c_in_proc, detected_hollow_r, pw, ph, true);
                if (at_cbc.valid) dr.ats.push_back(at_cbc);
                t_at_end = HRC::now();
                break;
            }

            default:
                LOGD("processFrame: EXCESS mode (n_valid=%d) — discarding",
                     dr.n_valid_lines);
                break;
        }

        t_cls_end = HRC::now();
        return dr;
    };

    // ── Single-pass detection ─────────────────────────────────────────────────
    // The detection block runs exactly once per frame on the original,
    // unexpanded ROI. There is no two-pass / ROI-expansion mechanism.
    cv::Point2f hollow_c_in_proc = ds_res.hollow_center_orig;
    const cv::Mat& proc_roi = roi;

    cv::Mat erase_pad_erased;
    DetectionResult det = runDetectionBlock(
            proc_roi, hollow_c_in_proc, ds_res.hollow_radius_orig,
            erase_pad_erased);

    // ── Stage 8: coordinate restore ───────────────────────────────────────────
    // restore_origin is always (0,0) now that there is only a single detection
    // pass on the unexpanded ROI; kept as a no-op pass-through for structural
    // consistency with future changes.
    t_rest_start = HRC::now();
    cv::Point2f restore_origin(0.f, 0.f);

    std::vector<ATStrip> ats_roi;
    for (const auto& at : det.ats)
        ats_roi.push_back(restoreAT(at, restore_origin));

    for (auto& at : ats_roi) {
        if (!at.valid) continue;
        int tw_out = -1;
        cv::Point2f corrected = correctATGhostOrigin(
                at.origin, at.evec, gr,
                pool_local_x1, pool_local_y1, pool_local_x2, pool_local_y2, tw_out);
        at.origin = corrected;
        if (tw_out >= 0 && pt_dot(at.evec, wallNorm(tw_out)) < 0.f)
            at.evec = reflDir(at.evec, tw_out);
    }
    t_rest_end = HRC::now();

    // ── physics-based trajectory computation ────────────────────────────────────
    std::vector<ShotEx> shots;
    if (!ats_roi.empty()) {
        const int ptLeft   = ptOX, ptTop    = ptOY;
        const int ptRight  = ptOX + ptW, ptBottom = ptOY + ptH;

        shots = m_trajectoryPhysics->computeTrajectories(
                ats_roi,
                ptLeft, ptTop, ptRight, ptBottom,
                gr,
                m_trajectoryPowerPct.load(),
                m_cueForceStat.load(), m_cueSpinStat.load(),
                maxr_tgt, maxr_cbc);
    }

    // ── Publish result ────────────────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lock(m_resultMutex);
        m_result.ats      = ats_roi;
        m_result.pockets  = pockets;
        m_result.shots    = shots;
        m_result.roiX1 = rx1; m_result.roiY1 = ry1;
        m_result.roiX2 = rx2; m_result.roiY2 = ry2;
        m_result.poolX1 = poolValid ? px1 : rx1;
        m_result.poolY1 = poolValid ? py1 : ry1;
        m_result.poolX2 = poolValid ? px2 : rx2;
        m_result.poolY2 = poolValid ? py2 : ry2;
        m_result.ghostRadius = gr;
        m_result.maxr_cbc    = maxr_cbc;
        m_result.maxr_tgt    = maxr_tgt;
        m_result.pocketR     = pr;
        m_result.pocketNS    = pns;
    }

    LOGD("[F#%llu] hollow_detect=%.1fms erase_pad=%.1fms "
         "seed=%.1fms strip=%.1fms cls=%.1fms at=%.1fms "
         "ca_strip=%.1fms restore=%.1fms | TOTAL=%.1fms | "
         "hollow=(%.0f,%.0f) clusters=%d valid=%d ats=%d shots=%d "
         "[cbc=%d/tgt=%d]",
         static_cast<unsigned long long>(m_frameCount),
         elapsed_ms(t_ds_start,    t_ds_end),
         elapsed_ms(t_epc_start,   t_epc_end),
         elapsed_ms(t_seed_start,  t_seed_end),
         elapsed_ms(t_strip_start, t_strip_end),
         elapsed_ms(t_cls_start,   t_cls_end),
         elapsed_ms(t_at_start,    t_at_end),
         elapsed_ms(t_ca_strip_start,  t_ca_strip_end),
         elapsed_ms(t_rest_start,  t_rest_end),
         elapsed_ms(t_total_start, HRC::now()),
         ds_res.hollow_center_orig.x, ds_res.hollow_center_orig.y,
         static_cast<int>(det.clusters.size()),
         det.n_valid_lines,
         static_cast<int>(ats_roi.size()),
         static_cast<int>(shots.size()),
         maxr_cbc, maxr_tgt);
}