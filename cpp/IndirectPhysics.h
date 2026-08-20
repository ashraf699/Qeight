/**
 * @file IndirectPhysics.h
 * @brief Additive physics capabilities for indirect (bank-shot) solving.
 *
 * This header introduces exactly two new physics capabilities that do not
 * exist in PoolPhysicsEngine:
 *
 *   1. Ball-ball collision (time-of-impact + response) against a STATIONARY
 *      target ball — ported from the reference JS implementation's
 *      PoolMath.ballBallCollisionTime / ballBallCollision, specialised for
 *      a non-moving target.
 *
 *   2. A simulation primitive, simulateUntilCushionOrTarget(), which mirrors
 *      PoolPhysicsEngine::simulateSingleBall()'s cushion/friction/sub-stepping
 *      loop exactly, but also tests for collision against a fixed
 *      target-ball circle at every sub-step, stopping the instant that
 *      collision is the earliest event.
 *
 * IndirectPhysics does not reimplement any cushion or friction physics: it
 * wraps a const PoolPhysicsEngine& and reuses its public
 * getCushionPolygon(), ballLineCollisionTime(), ballPointCollisionTime(),
 * ballLineCollision(), tableBallFriction(), and calculateTheta() directly.
 *
 * PoolPhysicsEngine.h/.cpp are frozen and are not modified by this file in
 * any way. All conventions (CGS units, Doxygen style, ALMOST_ZERO-style
 * epsilon guards, anonymous-namespace internal helpers, const-correctness)
 * match PoolPhysicsEngine.h/.cpp exactly.
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>

#include "PoolPhysicsEngine.h"

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/**
 * @brief Categorises the terminal event of simulateUntilCushionOrTarget().
 *
 * This is a separate, local categorisation — PoolPhysicsEngine.h's
 * PathEventType enum is frozen and is not extended with a new enumerator
 * for target-ball contact.
 */
enum class IndirectPathEventType {
    START,    ///< Initial position supplied by the caller
    CUSHION,  ///< Ball made contact with a cushion edge or vertex
    TARGET,   ///< Ball made contact with the fixed target-ball circle
    REST      ///< Ball came to a complete stop without reaching the target
};

/**
 * @brief A single waypoint in a simulateUntilCushionOrTarget() path.
 *
 * Mirrors PoolPhysicsEngine's PathPoint conventions exactly, substituting
 * IndirectPathEventType for PathEventType since the latter is frozen and
 * has no TARGET category.
 */
struct IndirectPathPoint {
    cv::Point2f position;             ///< cm-space position of the event
    IndirectPathEventType type;       ///< Nature of the event at this waypoint
    int cushionSideIndex = -1;        ///< For CUSHION events, index into the
    ///< engine's cushion polygon of the edge
    ///< struck (or nearest edge, for vertex
    ///< collisions). -1 for START/TARGET/REST.
};

/**
 * @brief Full result of simulateUntilCushionOrTarget().
 *
 * When the simulation terminates because of a TARGET event (the earliest
 * event at some sub-step was contact with the fixed target-ball circle),
 * cueVelocityAtContact and cueSpinAtContact hold the simulated ball's exact
 * velocity and spin at the moment of contact, BEFORE any ball-ball collision
 * response has been applied — the caller is expected to feed these into
 * IndirectPhysics::ballBallCollision() (capability #1) separately to resolve
 * the collision response itself.
 *
 * For CUSHION or REST termination, cueVelocityAtContact/cueSpinAtContact
 * simply hold the ball's velocity/spin at the final recorded path point
 * (i.e. the same state the ball would carry if simulation continued);
 * callers terminating for those reasons do not need these fields.
 */
struct IndirectSimResult {
    std::vector<IndirectPathPoint> path;  ///< Ordered waypoints, START-first.

    cv::Point2f cueVelocityAtContact;      ///< Ball velocity (cm/s) at path.back().
    float       cueSpinXAtContact = 0.f;   ///< Ball spinX at path.back().
    float       cueSpinYAtContact = 0.f;   ///< Ball spinY at path.back().
    float       cueSpinZAtContact = 0.f;   ///< Ball spinZ at path.back().
};

// ---------------------------------------------------------------------------
// IndirectPhysics
// ---------------------------------------------------------------------------

/**
 * @brief Additive physics capabilities for indirect (bank-shot) solving.
 *
 * Wraps a const PoolPhysicsEngine& (same const-reference ownership pattern
 * as IndirectShotSolver): stores it, never mutates it, never constructs a
 * second PoolPhysicsEngine. All cushion/friction physics is delegated to
 * the wrapped engine's existing public static/const methods.
 *
 * Thread-safety: const methods are safe to call concurrently from multiple
 * threads, provided the wrapped PoolPhysicsEngine is itself safe to use
 * concurrently (it is, per its own contract).
 */
class IndirectPhysics
{
public:

    /**
     * @brief Constructs the wrapper around an existing engine instance.
     *
     * @param engine  Engine to delegate cushion/friction physics to. Stored
     *                as a const reference; the referenced object must
     *                outlive this IndirectPhysics instance.
     */
    explicit IndirectPhysics(const PoolPhysicsEngine& engine);

    // -----------------------------------------------------------------------
    // Capability #1 — ball-ball collision against a STATIONARY target ball
    // -----------------------------------------------------------------------

    /**
     * @brief Computes the earliest time t ∈ (0, maxT] at which @p ball will
     *        contact a STATIONARY target ball.
     *
     * Specialised from the reference implementation's general two-moving-
     * balls ballBallCollisionTime formula for the case where the target's
     * velocity is always zero (its velocity terms therefore drop out of the
     * general formula; the general moving-target case is intentionally not
     * ported).
     *
     * @param ball          Moving ball's current state (position, velocity,
     *                      radius). Spin is irrelevant to this calculation
     *                      and is ignored.
     * @param targetPos     Stationary target ball's centre (cm).
     * @param targetRadius  Stationary target ball's radius (cm).
     * @param maxT          Upper bound of the time window to test (seconds).
     * @return              Time of first contact in seconds, or a value
     *                      greater than maxT if no collision occurs within
     *                      the interval.
     */
    static float ballBallCollisionTime(const SimBall& ball,
                                       cv::Point2f     targetPos,
                                       float           targetRadius,
                                       float           maxT);

    /**
     * @brief Applies a ball-ball collision response against a STATIONARY
     *        target ball.
     *
     * Pure velocity decomposition (normal/tangential split), specialised
     * from the reference implementation's general ballBallCollision for a
     * zero-velocity target. Confirmed from the reference: this response
     * reads and writes ONLY velocity.x/velocity.y on both balls — spinX,
     * spinY, and spinZ on @p ball and @p targetVelocityOut are left
     * completely untouched, matching the reference exactly.
     *
     * @param ball               Moving ball; velocity is updated in-place
     *                           to its post-collision value. Position,
     *                           radius, and all spin components are left
     *                           unmodified.
     * @param targetPos          Stationary target ball's centre (cm) at the
     *                           moment of contact.
     * @param targetRadius       Stationary target ball's radius (cm). Used,
     *                           together with @p ball's radius, to
     *                           re-derive a clean contact position exactly
     *                           on the sum-of-radii circle around
     *                           @p targetPos before computing the collision
     *                           normal — @p ball's position may be slightly
     *                           inside that circle due to upstream
     *                           near-zero time-of-contact snapping.
     * @param targetVelocityOut  Receives the target ball's post-collision
     *                           velocity (cm/s). The target's own spin is
     *                           untouched by this call — the caller is
     *                           responsible for combining this velocity
     *                           with the target ball's pre-existing spin
     *                           state, if any.
     */
    static void ballBallCollision(SimBall&     ball,
                                  cv::Point2f   targetPos,
                                  float         targetRadius,
                                  cv::Point2f&  targetVelocityOut);

    // -----------------------------------------------------------------------
    // Capability #2 — simulate until cushion or target-ball contact
    // -----------------------------------------------------------------------

    /**
     * @brief Simulates a single ball under cushion/friction physics exactly
     *        like PoolPhysicsEngine::simulateSingleBall(), but stops the
     *        instant the ball contacts a fixed target-ball circle.
     *
     * Reuses the wrapped engine's exact cushion/friction/sub-stepping
     * structure and PathPoint/PathEventType conventions (translated to
     * IndirectPathPoint/IndirectPathEventType, since PathEventType is
     * frozen and has no TARGET category). At every sub-step, in addition to
     * the existing cushion edge/vertex collision-time tests, this also
     * tests for collision against the fixed target-ball circle using
     * ballBallCollisionTime() (capability #1), and resolves whichever
     * collision — cushion or target — has the earliest time.
     *
     * When a target-ball collision is the earliest event at some sub-step:
     *   - the ball is advanced to the contact time (but the ball-ball
     *     collision RESPONSE is NOT applied — that is left to the caller,
     *     via ballBallCollision(), since the caller needs the pre-collision
     *     state to resolve the collision separately),
     *   - a final IndirectPathPoint of type TARGET is appended, with
     *     position set to the ball's centre at the moment of contact,
     *   - the returned IndirectSimResult's cueVelocityAtContact/
     *     cueSpinXAtContact/cueSpinYAtContact/cueSpinZAtContact fields hold
     *     the ball's exact velocity and spin at that same moment,
     *   - simulation returns immediately; no further integration occurs.
     *
     * All other behaviour (cushion collision resolution, friction
     * application once per FIXED_DT_SEC frame, REST detection via
     * PoolPhysicsEngine::isMovingOrSpinning(), the maxReflections cap on
     * cushion bounces, and the MAX_SIM_STEPS safety ceiling) is identical
     * to simulateSingleBall(), using the wrapped engine's own constants
     * (PoolPhysicsEngine::FIXED_DT_SEC, ::MAX_SIM_STEPS, etc.) throughout.
     *
     * @param ball            Initial ball state (position, velocity, spin).
     * @param targetPos       Stationary target ball's centre (cm).
     * @param targetRadius    Stationary target ball's radius (cm).
     * @param maxReflections  Maximum number of cushion bounces to simulate
     *                        after the first straight leg (≥ 0), with the
     *                        same semantics as simulateSingleBall(): the
     *                        path may contain at most (maxReflections + 1)
     *                        CUSHION-type points before simulation is
     *                        forced to stop (as a REST point) even if the
     *                        target was never reached.
     * @return  IndirectSimResult containing the ordered path (START-first,
     *          terminating in TARGET or REST — this call never produces a
     *          POCKET point, since pocket capture is not tested) and the
     *          ball's velocity/spin at the final path point.
     */
    IndirectSimResult simulateUntilCushionOrTarget(SimBall     ball,
                                                   cv::Point2f targetPos,
                                                   float       targetRadius,
                                                   int         maxReflections) const;

    // -----------------------------------------------------------------------
    // Cheap analytic pre-filtering helper (NOT a collision authority)
    // -----------------------------------------------------------------------

    /**
     * @brief Returns the signed perpendicular distance from a target-ball
     *        centre to a ray, given an already-known bounce point and
     *        post-bounce direction.
     *
     * Pure geometry — no simulation is performed. This is intended purely
     * as a cheap analytic pre-filter during search (e.g. rejecting
     * candidate bounce directions that pass nowhere near the target ball
     * before paying for a full simulateUntilCushionOrTarget() call).
     *
     * IMPORTANT: this helper is valid ONLY for a single straight-line leg
     * (i.e. it assumes the ball travels in a straight line from
     * @p bouncePoint along @p direction with no cushion bounces and no
     * curvature from spin). It is an optimisation aid, NOT an authority on
     * contact — a small perpendicular distance does not guarantee an actual
     * collision (the closest approach may occur before the ray reaches the
     * target, or beyond where a cushion bounce would actually occur), and a
     * large one does not by itself rule out a collision on a curved or
     * multi-leg path. Callers must still use
     * simulateUntilCushionOrTarget() for authoritative collision detection.
     *
     * @param bouncePoint   Origin of the ray (cm) — e.g. a known cushion
     *                      contact point.
     * @param direction     Post-bounce direction (need not be normalised;
     *                      only its direction is used).
     * @param targetPos     Target ball centre (cm).
     * @return  Signed perpendicular distance (cm) from @p targetPos to the
     *          infinite line through @p bouncePoint along @p direction.
     *          The sign follows the same left/right convention as a 2-D
     *          cross product of @p direction with (targetPos - bouncePoint):
     *          positive when the target lies to the left of the direction
     *          of travel, negative when to the right. Magnitude only
     *          (unsigned) is typically what search pre-filtering wants;
     *          callers needing only a distance can std::fabs() the result.
     */
    static float signedPerpendicularDistanceToRay(cv::Point2f bouncePoint,
                                                  cv::Point2f direction,
                                                  cv::Point2f targetPos);

private:

    /// Wrapped engine — never mutated, never duplicated. Referenced object
    /// must outlive this IndirectPhysics instance.
    const PoolPhysicsEngine& m_engine;
};