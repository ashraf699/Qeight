/**
 * @file IndirectShotSolver.h
 * @brief Single-angle evaluator for indirect (cushion-assisted ball-ball)
 *        pool shots, driven by a live user-rotated aiming control.
 *
 * IndirectShotSolver is a pure companion to PoolPhysicsEngine and
 * IndirectPhysics. It does NOT replicate any physics; it drives:
 *
 *   - IndirectPhysics::simulateUntilCushionOrTarget() to fly the cue ball
 *     from its start position, off zero or more cushions, until it makes
 *     contact with the target-ball circle,
 *   - IndirectPhysics::ballBallCollision() to resolve the resulting
 *     collision response,
 *   - PoolPhysicsEngine::simulateSingleBall() (completely unmodified) to
 *     roll out the target ball's subsequent leg toward whichever pocket it
 *     naturally reaches.
 *
 * ### Narrow purpose
 * This class is a THIN EVALUATOR, not a solver or searcher. Angle selection
 * is performed externally by the user rotating an aiming control; this class
 * receives a single angle, simulates exactly what that angle produces, and
 * reports the outcome as-is. There is no internal search, no refinement, no
 * candidate collection, no spin sweep, and no rejection of zero-cushion
 * (direct) hits — whatever the given angle produces is reported faithfully.
 *
 * Design constraints (do not relax without updating this header):
 *  - Zero physics logic lives here. All ball motion, friction, cushion
 *    normals, pocket detection, and coordinate conventions come exclusively
 *    from PoolPhysicsEngine's and IndirectPhysics's public APIs.
 *  - The evaluator is constructed with const references to caller-owned
 *    PoolPhysicsEngine and IndirectPhysics instances; it never creates its
 *    own instances of either.
 *  - The header is portable, standard C++17; no Android, JNI, or logging
 *    headers are included.
 *  - Only <opencv2/core.hpp>, <vector>, <cmath>, <limits>, <algorithm>,
 *    <optional>, "PoolPhysicsEngine.h", and "IndirectPhysics.h" are
 *    included.
 *  - The evaluator never searches over launch power: request.launchSpeedCmS
 *    is always used as-is (full power is a product requirement enforced by
 *    the caller, not by this class).
 *
 * ### Evaluation overview
 * Given a cue-ball position, a target-ball position, ball radius, launch
 * speed, and a caller-supplied angle:
 *
 * Step 1 — Cue ball flight:
 *   IndirectPhysics::simulateUntilCushionOrTarget() flies the cue ball from
 *   request.cueBallPos at the supplied angle and request.launchSpeedCmS,
 *   with zero spin, allowing up to MAX_CUSHION_BOUNCES reflections. The
 *   result is inspected to determine whether the cue ball made contact with
 *   the target ball at all (touchesTarget).
 *
 * Step 2 — Collision resolution (only if touchesTarget):
 *   IndirectPhysics::ballBallCollision() resolves the cue ball's
 *   post-contact velocity and the target ball's launch velocity from the
 *   contact state produced in Step 1.
 *
 * Step 3 — Target ball roll-out (only if touchesTarget):
 *   PoolPhysicsEngine::simulateSingleBall() rolls the target ball out from
 *   the contact position with the velocity from Step 2. If the terminal
 *   PathPoint is PathEventType::POCKET, pots is set to true and pocketHit
 *   is recorded. The pocket is whichever one the target ball naturally
 *   reaches — no specific pocket is chosen up front.
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <optional>
#include <cmath>
#include <limits>
#include <algorithm>
#include "PoolPhysicsEngine.h"
#include "IndirectPhysics.h"

// =============================================================================
//  Compile-time tuning constants
// =============================================================================

/**
 * @defgroup EvaluatorTuning Single-angle evaluator tuning constants
 * @{
 */

/**
 * @brief Maximum number of cushion reflections the cue ball is permitted
 *        before reaching the target ball, passed as maxReflections to
 *        IndirectPhysics::simulateUntilCushionOrTarget(), and also used as
 *        the maxReflections cap for the target ball's post-collision leg
 *        via PoolPhysicsEngine::simulateSingleBall().
 *
 * A generous ceiling is used since compute cost is not a hard constraint
 * for a single per-frame evaluation. Applies uniformly to both physics
 * rollouts inside evaluateAngle().
 */
static constexpr int MAX_CUSHION_BOUNCES = 6;

/** @} */


// =============================================================================
//  Public types
// =============================================================================

/**
 * @brief All inputs required by IndirectShotSolver::evaluateAngle().
 *
 * All positions are in centimetres in the same table-centred coordinate
 * system used by SimBall and PoolPhysicsEngine.
 */
struct IndirectShotRequest
{
    /// Cue-ball centre position in cm, table-centred.
    cv::Point2f cueBallPos;

    /// Target-ball centre position in cm, table-centred.
    cv::Point2f targetBallPos;

    /// Ball radius in cm. Defaults to a standard pool ball (~57 mm diameter).
    float ballRadiusCm = 3.0f;

    /**
     * @brief Cue-ball launch speed in cm/s.
     *
     * Always used at full power per product requirement: the evaluator does
     * NOT search over launch power. A value of 0 is invalid.
     *
     * Pre-compute via:
     * @code
     *   auto stats = PoolPhysicsEngine::applyCueStats(forceStat, spinStat);
     *   req.launchSpeedCmS = stats.maxPower;
     * @endcode
     */
    float launchSpeedCmS = 0.f;
};


/**
 * @brief The outcome of evaluating a single launch angle via
 *        IndirectShotSolver::evaluateAngle().
 *
 * All fields are populated unconditionally for the portions of the
 * simulation that ran; fields gated on touchesTarget or pots are
 * left at their documented defaults when the corresponding condition
 * was not reached.
 */
struct IndirectShotEvaluation
{
    /// Echoes the angle passed in to evaluateAngle(), radians, CCW from +X.
    float solvedLaunchAngleRad = 0.f;

    /**
     * @brief True if the cue ball's simulated path (cushions allowed) made
     *        contact with the target ball at all.
     *
     * When false, pots is always false, pocketHit is undefined, and
     * stitchedPath contains only the cue ball's cushion path with no
     * target leg appended.
     */
    bool touchesTarget = false;

    /**
     * @brief True only if, after resolving the ball-ball collision, the
     *        target ball's subsequent simulated leg terminates by entering
     *        ANY pocket — whichever one it naturally reaches, not a
     *        specific chosen one.
     *
     * Always false when touchesTarget is false.
     */
    bool pots = false;

    /**
     * @brief The pocket centre (cm-space) the target ball actually entered.
     *
     * Valid only when pots == true; undefined otherwise.
     */
    cv::Point2f pocketHit;

    /**
     * @brief Rendering-ready stitched path.
     *
     * When touchesTarget is true: the cue ball's leg from launch to the
     * contact point, concatenated with the target ball's leg from the
     * contact point onward. The contact position appears once at the join
     * (not duplicated). This is NOT a physically continuous single-ball
     * trajectory — the acting ball switches from cue to target at the join.
     *
     * When touchesTarget is false: contains only the cue ball's cushion
     * path; no target leg is appended.
     */
    std::vector<PathPoint> stitchedPath;
};


// =============================================================================
//  IndirectShotSolver
// =============================================================================

/**
 * @class IndirectShotSolver
 * @brief Thin single-angle evaluator for indirect pool shots, intended for
 *        use with a live, user-rotated aiming UI.
 *
 * Given a launch angle supplied by the caller, evaluateAngle() simulates
 * exactly what that angle produces — cue ball flight (with cushion bounces
 * allowed), optional ball-ball collision, and target ball roll-out to
 * whichever pocket it naturally reaches — and reports the outcome as-is.
 * No search, no refinement, no candidate collection, and no rejection of
 * any particular hit geometry are performed internally.
 *
 * ### Usage
 * @code
 *   PoolPhysicsEngine engine;
 *   IndirectPhysics   indirectPhysics(engine);
 *   IndirectShotSolver evaluator(engine, indirectPhysics);
 *
 *   IndirectShotRequest req;
 *   req.cueBallPos     = cv::Point2f(-10.f, -30.f);
 *   req.targetBallPos  = cv::Point2f( 20.f,  40.f);
 *   req.ballRadiusCm   = 3.0f;
 *   auto stats         = PoolPhysicsEngine::applyCueStats(3, 3);
 *   req.launchSpeedCmS = stats.maxPower;
 *
 *   float userAngleRad = /* from rotating UI control *\/;
 *   IndirectShotEvaluation eval = evaluator.evaluateAngle(req, userAngleRad);
 *   if (eval.pots) { /* highlight the pocket eval.pocketHit *\/ }
 * @endcode
 */
class IndirectShotSolver
{
public:

    // -------------------------------------------------------------------------
    //  Construction
    // -------------------------------------------------------------------------

    /**
     * @brief Constructs the evaluator.
     *
     * @param engine           Caller-owned PoolPhysicsEngine. Must outlive
     *                         this evaluator. Used for
     *                         simulateSingleBall() rollout of the target
     *                         ball's post-contact leg.
     * @param indirectPhysics  Caller-owned IndirectPhysics wrapping the same
     *                         engine. Must outlive this evaluator. Used for
     *                         simulateUntilCushionOrTarget() and
     *                         ballBallCollision().
     */
    IndirectShotSolver(const PoolPhysicsEngine& engine,
                       const IndirectPhysics&   indirectPhysics);

    ~IndirectShotSolver() = default;

    IndirectShotSolver(const IndirectShotSolver&)            = delete;
    IndirectShotSolver& operator=(const IndirectShotSolver&) = delete;
    IndirectShotSolver(IndirectShotSolver&&)                 = delete;
    IndirectShotSolver& operator=(IndirectShotSolver&&)      = delete;


    // -------------------------------------------------------------------------
    //  Public interface
    // -------------------------------------------------------------------------

    /**
     * @brief Evaluates a single launch angle and returns the simulated
     *        outcome with no internal search or refinement.
     *
     * Simulates the cue ball launched from request.cueBallPos at angle
     * thetaRad (radians, CCW from +X) and request.launchSpeedCmS, with
     * zero spin, using up to MAX_CUSHION_BOUNCES cushion reflections.
     * Checks for target-ball contact; if contact occurs, resolves the
     * ball-ball collision and rolls the target ball out to see if it
     * reaches any pocket (whichever it naturally reaches — no pocket is
     * chosen up front). No search, no refinement, no rejection of
     * zero-cushion (direct) hits — whatever the angle produces is
     * reported as-is.
     *
     * @param request   Fully populated IndirectShotRequest.
     *                  launchSpeedCmS must be > 0.
     * @param thetaRad  Launch angle in radians, CCW from the +X axis,
     *                  supplied directly by the caller's aiming UI.
     * @return          IndirectShotEvaluation describing what this specific
     *                  angle produces. Never throws; returns a
     *                  default-constructed (all-false, empty-path)
     *                  evaluation if launchSpeedCmS == 0.
     */
    IndirectShotEvaluation evaluateAngle(const IndirectShotRequest& request,
                                         float thetaRad) const;


private:

    // -------------------------------------------------------------------------
    //  Private data members
    // -------------------------------------------------------------------------

    /// Const reference to the caller-owned physics engine.
    const PoolPhysicsEngine& m_engine;

    /// Const reference to the caller-owned indirect-physics wrapper.
    const IndirectPhysics& m_indirectPhysics;
};