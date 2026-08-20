/**
 * @file IndirectShotSolver.cpp
 * @brief Implementation of every method declared in IndirectShotSolver.h.
 *
 * Design constraints honoured throughout this file:
 *  - Zero physics logic lives here. Every ball-motion result comes from
 *    m_indirectPhysics.simulateUntilCushionOrTarget(),
 *    IndirectPhysics::ballBallCollision(), or
 *    m_engine.simulateSingleBall(). No friction, reflection, collision, or
 *    pocket-capture formulae are duplicated.
 *  - Both engine references are const; neither's state is ever mutated.
 *  - Only <opencv2/core.hpp>, <vector>, <cmath>, "PoolPhysicsEngine.h",
 *    "IndirectPhysics.h", and "IndirectShotSolver.h" are included.
 *
 * Coordinate convention (inherited from PoolPhysicsEngine):
 *  - Table centred at origin.
 *  - +X along the long axis (right), +Y along the short axis (upward on a
 *    diagram with the shooter at the bottom).
 *  - Units: centimetres throughout; angles in radians, CCW from +X.
 */

#include "IndirectShotSolver.h"

#include <cmath>
#include <vector>

// =============================================================================
//  Internal-linkage helpers (anonymous namespace — not visible outside this TU)
// =============================================================================

namespace {

    /**
     * @brief Builds a launch SimBall for the given angle; spin is always zero.
     *
     * All positional and geometric fields are taken directly from @p request.
     * spinX, spinY, and spinZ are unconditionally set to 0 — this solver
     * operates purely on angle and lets the physics engine determine all
     * spin-dependent behaviour downstream.
     *
     * @param request    The originating indirect-shot request.
     * @param thetaRad   Launch direction in radians, CCW from +X.
     * @return           A fully initialised SimBall ready for simulation.
     */
    inline SimBall makeLaunchBall(const IndirectShotRequest& request,
                                  float                       thetaRad)
    {
        SimBall ball;
        ball.position = request.cueBallPos;
        ball.velocity = { std::cos(thetaRad) * request.launchSpeedCmS,
                          std::sin(thetaRad) * request.launchSpeedCmS };
        ball.spinX    = 0.f;
        ball.spinY    = 0.f;
        ball.spinZ    = 0.f;
        ball.radius   = request.ballRadiusCm;
        return ball;
    }

    /**
     * @brief Converts an IndirectPathPoint path into PathPoint-shaped data.
     *
     * TARGET events (which have no PathEventType equivalent) are mapped to
     * REST, since by the time this conversion happens the TARGET point is
     * being used purely as the join point of a stitched render path, not as
     * a "the ball stopped" signal.
     *
     * @param path   Path produced by simulateUntilCushionOrTarget().
     * @return       Equivalent path expressed in PathPoint / PathEventType terms.
     */
    inline std::vector<PathPoint> toRenderPath(
            const std::vector<IndirectPathPoint>& path)
    {
        std::vector<PathPoint> out;
        out.reserve(path.size());
        for (const auto& p : path)
        {
            PathEventType mapped;
            switch (p.type)
            {
                case IndirectPathEventType::START:   mapped = PathEventType::START;   break;
                case IndirectPathEventType::CUSHION: mapped = PathEventType::CUSHION; break;
                case IndirectPathEventType::REST:    mapped = PathEventType::REST;    break;
                case IndirectPathEventType::TARGET:  mapped = PathEventType::REST;    break;
                default:                              mapped = PathEventType::REST;    break;
            }
            out.push_back({ p.position, mapped, p.cushionSideIndex });
        }
        return out;
    }

} // anonymous namespace


// =============================================================================
//  Constructor
// =============================================================================

IndirectShotSolver::IndirectShotSolver(const PoolPhysicsEngine& engine,
                                       const IndirectPhysics&   indirectPhysics)
        : m_engine(engine)
        , m_indirectPhysics(indirectPhysics)
{
}


// =============================================================================
//  evaluateAngle — public entry point
// =============================================================================

/**
 * @brief Evaluates a single candidate launch angle against the indirect-shot
 *        request and returns a fully populated IndirectShotEvaluation.
 *
 * Evaluation proceeds in strict order:
 *  1. Build the launch SimBall (zero spin).
 *  2. Simulate the cue ball's path until it contacts the target ball or
 *     exhausts the allowed cushion budget.
 *  3. If the target was never reached, return early with touchesTarget = false.
 *  4. Otherwise resolve the ball-ball collision, roll out the target ball,
 *     build the stitched render path, and return the fully populated
 *     evaluation.
 *
 * No physics logic is duplicated here — all motion and collision results come
 * from m_indirectPhysics and m_engine exclusively.
 *
 * @param request    Fully populated shot request (positions, speed, geometry).
 * @param thetaRad   Candidate launch angle in radians, CCW from +X.
 * @return           IndirectShotEvaluation describing everything that happened.
 */
IndirectShotEvaluation IndirectShotSolver::evaluateAngle(
        const IndirectShotRequest& request,
        float                       thetaRad) const
{
    IndirectShotEvaluation eval;  // touchesTarget = false, pots = false by default

    // -------------------------------------------------------------------------
    // Step 1: Build the launch SimBall (spin always zero at launch).
    // -------------------------------------------------------------------------
    const SimBall launchBall = makeLaunchBall(request, thetaRad);

    // -------------------------------------------------------------------------
    // Step 2: Simulate the cue ball until it contacts the target ball or
    //         exceeds the cushion-bounce budget.
    // -------------------------------------------------------------------------
    const IndirectSimResult simResult =
            m_indirectPhysics.simulateUntilCushionOrTarget(
                    launchBall,
                    request.targetBallPos,
                    request.ballRadiusCm,
                    MAX_CUSHION_BOUNCES);

    // -------------------------------------------------------------------------
    // Step 3: If the path is empty or the terminal event is not TARGET the
    //         cue ball never reached the target ball — return early.
    // -------------------------------------------------------------------------
    if (simResult.path.empty() ||
        simResult.path.back().type != IndirectPathEventType::TARGET)
    {
        // touchesTarget already false; still populate stitchedPath with the
        // cue ball's cushion-only path so callers always have something to
        // render.
        eval.stitchedPath = toRenderPath(simResult.path);
        return eval;
    }

    // -------------------------------------------------------------------------
    // Step 4a: Record contact and reconstruct the cue ball's exact state at
    //          the moment of contact (pre-collision, as documented by
    //          IndirectPhysics.h).
    // -------------------------------------------------------------------------
    eval.touchesTarget = true;

    const cv::Point2f contactPos = simResult.path.back().position;

    SimBall cueAtContact;
    cueAtContact.position = contactPos;
    cueAtContact.velocity = simResult.cueVelocityAtContact;
    cueAtContact.spinX    = simResult.cueSpinXAtContact;
    cueAtContact.spinY    = simResult.cueSpinYAtContact;
    cueAtContact.spinZ    = simResult.cueSpinZAtContact;
    cueAtContact.radius   = request.ballRadiusCm;

    // -------------------------------------------------------------------------
    // Step 4b: Resolve the ball-ball collision to obtain the target ball's
    //          launch velocity.  ballBallCollision() updates cueAtContact's
    //          own velocity in-place (post-collision) and writes the target
    //          ball's resultant velocity into targetVelocity.
    // -------------------------------------------------------------------------
    cv::Point2f targetVelocity;
    IndirectPhysics::ballBallCollision(cueAtContact,
                                       request.targetBallPos,
                                       request.ballRadiusCm,
                                       targetVelocity);

    // -------------------------------------------------------------------------
    // Step 4c: Roll out the target ball from rest at its starting position
    //          using the resolved post-collision velocity.
    // -------------------------------------------------------------------------
    SimBall targetLaunch;
    targetLaunch.position = request.targetBallPos;
    targetLaunch.velocity = targetVelocity;
    targetLaunch.spinX    = 0.f;   // target ball was at rest — no pre-existing spin
    targetLaunch.spinY    = 0.f;
    targetLaunch.spinZ    = 0.f;
    targetLaunch.radius   = request.ballRadiusCm;

    const std::vector<PathPoint> targetPath =
            m_engine.simulateSingleBall(targetLaunch, MAX_CUSHION_BOUNCES);

    // Pots if and only if the target ball's terminal event is POCKET.
    if (!targetPath.empty() &&
        targetPath.back().type == PathEventType::POCKET)
    {
        eval.pots      = true;
        eval.pocketHit = targetPath.back().position;
    }

    // -------------------------------------------------------------------------
    // Step 4d: Build the stitched render path.
    //
    //  • Convert the cue leg (IndirectPathPoint) to PathPoint via toRenderPath.
    //  • The cue leg's last point IS the contact position; the target ball's
    //    path starts at that same position (its START point), so skip index 0
    //    of targetPath to avoid duplicating the join point.
    // -------------------------------------------------------------------------
    eval.stitchedPath = toRenderPath(simResult.path);

    for (std::size_t i = 1; i < targetPath.size(); ++i)
        eval.stitchedPath.push_back(targetPath[i]);

    return eval;
}