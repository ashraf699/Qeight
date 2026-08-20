// ============================================================
//  TrajectoryPhysicsEngine.cpp
//
//  Implementation of TrajectoryPhysicsEngine.
//
//  Design notes
//  ------------
//  •  All physics simulation happens in PoolPhysicsEngine's
//     centimetre coordinate system (table centre == origin).
//  •  px ↔ cm conversion is done per-axis so that a
//     non-square pixel rectangle maps correctly onto the fixed
//     real-world 2:1 table aspect ratio.
//  •  The conversion scale factors are recomputed on every
//     call because the calibrated table rectangle can change
//     frame-to-frame (e.g. camera zoom or re-calibration).
// ============================================================

#include "TrajectoryPhysicsEngine.h"
#include <cmath>        // std::sqrt, std::max
#include <algorithm>    // std::clamp


// -----------------------------------------------------------------------
//  Table dimension constants
//
//  PoolPhysicsEngine::TABLE_LENGTH_CM and TABLE_WIDTH_CM must be public
//  static constexpr members of PoolPhysicsEngine so that this file can
//  reference them directly (single source of truth).
//
//  If for any reason they cannot be exposed publicly, replace the two
//  usages below with the literal values
//      254.0f   (TABLE_LENGTH_CM – standard 9-foot table playing surface)
//      127.0f   (TABLE_WIDTH_CM  – exactly half the length)
//  and add a static_assert or comment warning that those values MUST stay
//  in sync with whatever PoolPhysicsEngine uses internally.
// -----------------------------------------------------------------------


// ============================================================
//  Constructor
// ============================================================

TrajectoryPhysicsEngine::TrajectoryPhysicsEngine()
        : m_physics()   // PoolPhysicsEngine default-ctor builds cushion
// polygon and pocket list; nothing else to do.
{}


// ============================================================
//  computeTrajectories
// ============================================================

std::vector<ShotEx> TrajectoryPhysicsEngine::computeTrajectories(
        const std::vector<ATStrip>& strips,
        int ptLeftPx,
        int ptTopPx,
        int ptRightPx,
        int ptBottomPx,
        int ballRadiusPx,
        int powerPct,
        int cueForceStat,
        int cueSpinStat,
        int maxReflectionsTgt,
        int maxReflectionsCbc) const
{
    std::vector<ShotEx> result;

    // ------------------------------------------------------------------
    //  0.  Validate the table rectangle.
    //      A degenerate rectangle means calibration has not run yet, or
    //      the frame is not usable; bail early with an empty result.
    // ------------------------------------------------------------------
    const float rectW = static_cast<float>(ptRightPx  - ptLeftPx);
    const float rectH = static_cast<float>(ptBottomPx - ptTopPx);
    if (rectW <= 1.f || rectH <= 1.f)
        return result;


    // ------------------------------------------------------------------
    //  1.  Derive px ↔ cm scale factors.
    //
    //  The table rectangle in px may have a different aspect ratio than
    //  the real table (perspective warp, lens distortion, etc.), so we
    //  keep separate X and Y factors.
    //
    //  pxPerCmAvg is the geometric mean of the two axis factors; it is
    //  used only for isotropic quantities (ball radius).
    // ------------------------------------------------------------------
    const float pxPerCmX   = rectW / PoolPhysicsEngine::TABLE_LENGTH_CM;
    const float pxPerCmY   = rectH / PoolPhysicsEngine::TABLE_WIDTH_CM;
    const float pxPerCmAvg = std::sqrt(pxPerCmX * pxPerCmY);

    // Pixel-space centre of the table rectangle (== cm-space origin).
    const float centerX = (static_cast<float>(ptLeftPx)  +
                           static_cast<float>(ptRightPx))  * 0.5f;
    const float centerY = (static_cast<float>(ptTopPx)   +
                           static_cast<float>(ptBottomPx)) * 0.5f;


    // ------------------------------------------------------------------
    //  2.  Coordinate-conversion lambdas.
    //
    //  cmToPx: cm-space point → ROI-local pixel point.
    //  pxToCm: ROI-local pixel point → cm-space point.
    //
    //  Both use the per-axis scale so anisotropy is handled correctly.
    // ------------------------------------------------------------------
    auto cmToPx = [&](cv::Point2f cm) -> cv::Point2f
    {
        return { centerX + cm.x * pxPerCmX,
                 centerY + cm.y * pxPerCmY };
    };

    auto pxToCm = [&](cv::Point2f px) -> cv::Point2f
    {
        return { (px.x - centerX) / pxPerCmX,
                 (px.y - centerY) / pxPerCmY };
    };


    // ------------------------------------------------------------------
    //  3.  Compute launch speed from cue stats and powerPct.
    //
    //  applyCueStats converts the raw 0-16 equipment attributes into a
    //  physical maximum power (cm/s or the engine's native speed unit)
    //  and a spin-scale ceiling.
    //
    //  The nonlinear power curve
    //      launchSpeed = (1 − √(1 − powerRatio)) × maxPower
    //  is a common game-feel ramp: it gives finer control at low power
    //  while still reaching full speed at powerPct = 100.
    // ------------------------------------------------------------------
    const auto  cueStats   = PoolPhysicsEngine::applyCueStats(cueForceStat,
                                                              cueSpinStat);
    const float powerRatio = std::clamp(powerPct, 1, 100) / 100.f;
    const float launchSpeed =
            (1.f - std::sqrt(std::max(0.f, 1.f - powerRatio))) * cueStats.maxPower;


    // ------------------------------------------------------------------
    //  4.  Derive physical ball radius.
    //
    //  Clamped to >= 0.5 cm so the physics engine never receives a
    //  zero-or-negative radius (which would violate its cushion-contact
    //  geometry assumptions).
    // ------------------------------------------------------------------
    const float ballRadiusCm =
            std::max(0.5f, static_cast<float>(ballRadiusPx) / pxPerCmAvg);


    // ------------------------------------------------------------------
    //  5.  Iterate over strips.
    // ------------------------------------------------------------------
    for (const auto& strip : strips)
    {
        // Skip strips that the vision pipeline marked as unreliable.
        if (!strip.valid)
            continue;

        // --------------------------------------------------------------
        //  5a.  Convert origin from px-space to cm-space.
        // --------------------------------------------------------------
        const cv::Point2f originCm = pxToCm(strip.origin);

        // --------------------------------------------------------------
        //  5b.  Convert the unit direction vector from px-space to
        //       cm-space.
        //
        //  strip.evec is a unit vector in (possibly anisotropic) px
        //  space.  To get the corresponding direction in cm-space we
        //  divide each component by that axis's px-per-cm factor (which
        //  is the same as multiplying by cm-per-px).  The result is
        //  then re-normalized to obtain a pure direction.
        //
        //  Why divide rather than multiply?
        //  A step of Δx px along X represents Δx/pxPerCmX cm.
        //  So the cm-space direction component is proportional to
        //  evec.x / pxPerCmX.
        // --------------------------------------------------------------
        const cv::Point2f dirCmRaw(strip.evec.x / pxPerCmX,
                                   strip.evec.y / pxPerCmY);

        const float dirLen = std::sqrt(dirCmRaw.x * dirCmRaw.x +
                                       dirCmRaw.y * dirCmRaw.y);

        // Skip strips whose direction is numerically degenerate.
        if (dirLen < 1e-6f)
            continue;

        const cv::Point2f dirCm(dirCmRaw.x / dirLen,
                                dirCmRaw.y / dirLen);

        // --------------------------------------------------------------
        //  5c.  Build the SimBall.
        //
        //  Spin components are all zero (see header NOTE on velocity
        //  direction).  PoolPhysicsEngine will accumulate spin naturally
        //  through its friction and cushion-contact models as the
        //  simulation advances.
        // --------------------------------------------------------------
        SimBall ball;
        ball.position = originCm;
        ball.velocity = { dirCm.x * launchSpeed,
                          dirCm.y * launchSpeed };
        ball.spinX    = 0.f;   // zero initial top/back spin – no english
        ball.spinY    = 0.f;   // input exists in this pipeline
        ball.spinZ    = 0.f;   // side-spin likewise zero at launch
        ball.radius   = ballRadiusCm;

        // --------------------------------------------------------------
        //  5d.  Choose the bounce cap for this strip type and simulate.
        //
        //  is_cue_ball_cut == true  → the strip models the cue ball's
        //                            path after contact, so use the
        //                            cue-ball-cut reflection cap.
        //  is_cue_ball_cut == false → target-ball path; use its cap.
        // --------------------------------------------------------------
        const int  cap  = strip.is_cue_ball_cut ? maxReflectionsCbc
                                                : maxReflectionsTgt;
        const auto path = m_physics.simulateSingleBall(ball, cap);

        // --------------------------------------------------------------
        //  5e.  Convert the returned path events into ShotEx segments.
        //
        //  path is a sequence of PathEvent structs.  Each consecutive
        //  pair [i, i+1] defines one straight-line segment (between a
        //  cushion bounce, pocket, or the start/stop of motion).
        //
        //  Both seg.wall and seg.pocket_stop are derived from the same
        //  path[i+1].type expression so they are always consistent:
        //
        //    wall == 0   → destination is a cushion bounce; the overlay
        //                  renderer's condition (wall >= 0 || pocket_stop)
        //                  is satisfied so a ghost circle is drawn there.
        //                  The actual cushion index is not mapped — 0 is
        //                  used purely as a non-negative sentinel.
        //    wall == -1  → destination is not a cushion event; no ghost
        //                  circle is drawn for wall alone (pocket_stop
        //                  may still trigger one independently).
        //    pocket_stop → true only when the ball enters a pocket,
        //                  always satisfies the renderer's condition.
        // --------------------------------------------------------------
        for (std::size_t i = 0; i + 1 < path.size(); ++i)
        {
            ShotEx seg;

            // Convert endpoints back to ROI-local px.
            seg.from = cmToPx(path[i    ].position);
            seg.to   = cmToPx(path[i + 1].position);

            // Normalised direction of this segment in px-space.
            const cv::Point2f d { seg.to.x - seg.from.x,
                                  seg.to.y - seg.from.y };
            const float dLen = std::sqrt(d.x * d.x + d.y * d.y);

            seg.dir = (dLen > 1e-6f)
                      ? cv::Point2f(d.x / dLen, d.y / dLen)
                      : cv::Point2f(0.f, 0.f);

            // Derive wall sentinel and pocket flag from destination type.
            seg.wall        = (path[i + 1].type == PathEventType::CUSHION)
                              ? 0 : -1;
            seg.pocket_stop = (path[i + 1].type == PathEventType::POCKET);

            result.push_back(seg);
        }

    }   // for each strip

    return result;
}