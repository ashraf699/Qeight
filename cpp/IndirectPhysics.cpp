/**
 * @file IndirectPhysics.cpp
 * @brief Implementation of every method declared in IndirectPhysics.h.
 *
 * Pure, portable C++ — no Android/JNI/logging dependencies.
 * All physics quantities are in CGS units (cm, g, s) unless noted otherwise.
 */

#include "IndirectPhysics.h"

#include <cmath>
#include <algorithm>
#include <limits>

// ---------------------------------------------------------------------------
// Internal convenience
// ---------------------------------------------------------------------------

namespace {

/// Returns Euclidean distance between two 2-D points.
    inline float pointDist(cv::Point2f a, cv::Point2f b)
    {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }

} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

IndirectPhysics::IndirectPhysics(const PoolPhysicsEngine& engine)
        : m_engine(engine)
{}

// ============================================================================
// Capability #1a — ballBallCollisionTime (stationary target special case)
// ============================================================================

// Reference (PoolMath.ballBallCollisionTime, general two-moving-balls case):
//
//   const sumR = a.radius + b.radius;
//   const rpX = b.position.x - a.position.x, rpY = b.position.y - a.position.y;
//   const rvX = b.velocity.x - a.velocity.x, rvY = b.velocity.y - a.velocity.y;
//   const A = rvX*rvX + rvY*rvY;
//   const B = 2*(rpX*rvX + rpY*rvY);
//   const C = rpX*rpX + rpY*rpY - sumR*sumR;
//   let t = (-B - Math.sqrt(B*B - 4*A*C)) / (2*A);
//   if(t < 0 && t + 0.05 > 0) t = M.ALMOST_ZERO;
//   if(t < 0 || t - M.ALMOST_ZERO > maxT || B >= 0) return Infinity;
//   return t;
//
// Specialised here for a STATIONARY target (b.velocity == (0,0)): the
// target's velocity terms drop out, so rvX/rvY reduce to -ball.velocity.x/y
// and rpX/rpY reduce to (targetPos - ball.position). Only this static-target
// special case is ported; the general two-moving-balls case is not.
float IndirectPhysics::ballBallCollisionTime(const SimBall& ball,
                                             cv::Point2f     targetPos,
                                             float           targetRadius,
                                             float           maxT)
{
    const float INF = std::numeric_limits<float>::infinity();

    const float sumR = ball.radius + targetRadius;

    // rp = targetPos - ball.position (b.position - a.position with b static)
    const float rpX = targetPos.x - ball.position.x;
    const float rpY = targetPos.y - ball.position.y;

    // rv = (0,0) - ball.velocity = -ball.velocity (b.velocity - a.velocity
    // with b static)
    const float rvX = -ball.velocity.x;
    const float rvY = -ball.velocity.y;

    const float A = rvX * rvX + rvY * rvY;
    const float B = 2.0f * (rpX * rvX + rpY * rvY);
    const float C = rpX * rpX + rpY * rpY - sumR * sumR;

    if (A < PoolPhysicsEngine::ALMOST_ZERO)
        return INF;   // ball is not moving — no collision can occur

    const float disc = B * B - 4.0f * A * C;
    if (disc < 0.0f)
        return INF;

    float t = (-B - std::sqrt(disc)) / (2.0f * A);

    // Reference epsilon-snap: a slightly-negative root within 0.05s of zero
    // (i.e. the ball already touching/overlapping the target at t=0, most
    // likely due to floating point error) is snapped forward to
    // ALMOST_ZERO rather than rejected outright.
    if (t < 0.0f && t + 0.05f > 0.0f)
        t = PoolPhysicsEngine::ALMOST_ZERO;

    // B >= 0 means the ball is not closing on the target (moving away or
    // exactly tangential) — reject exactly as the reference does.
    if (t < 0.0f || t - PoolPhysicsEngine::ALMOST_ZERO > maxT || B >= 0.0f)
        return INF;

    return t;
}

// ============================================================================
// Capability #1b — ballBallCollision (stationary target special case)
// ============================================================================

// Reference (PoolMath.ballBallCollision, general two-moving-balls case):
//
//   const dX = a.position.x - b.position.x, dY = a.position.y - b.position.y;
//   const dLen = Math.sqrt(dX*dX + dY*dY);
//   const nX = dX/dLen, nY = dY/dLen;
//   const aDotN = a.velocity.x*-nX + a.velocity.y*-nY;
//   const nAX = aDotN*-nX, nAY = aDotN*-nY;
//   const tAX = nAX - a.velocity.x, tAY = nAY - a.velocity.y;
//   const bDotN = b.velocity.x*nX + b.velocity.y*nY;
//   const nBX = bDotN*nX, nBY = bDotN*nY;
//   const tBX = nBX - b.velocity.x, tBY = nBY - b.velocity.y;
//   a.velocity.x = -tAX + nBX;
//   a.velocity.y = -tAY + nBY;
//   b.velocity.x = -tBX + nAX;
//   b.velocity.y = -tBY + nAY;
//
// Confirmed from the reference: this is pure velocity decomposition
// (normal/tangential split) — only velocity.x/y are read or written on
// either ball; spinX/Y/Z are never touched. Specialised here for a
// STATIONARY target (b.velocity == (0,0) going in): bDotN, nBX, nBY are
// therefore all zero, and tBX/tBY reduce to -b.velocity.x/y = (0,0). The
// full expression structure is kept (rather than algebraically collapsed)
// so this stays a faithful, checkable transcription of the reference for
// the static-target special case.
//
// The collision normal is derived directly from ball.position and
// targetPos, with no re-derived/corrected contact point — matching the
// reference and the pattern used elsewhere in this codebase (e.g. the
// vertex-collision normal in PoolPhysicsEngine::simulateSingleBall, which
// likewise trusts ball.position, advanced by velocity * t, directly for
// every collision type).
void IndirectPhysics::ballBallCollision(SimBall&    ball,
                                        cv::Point2f  targetPos,
                                        float        /*targetRadius*/,
                                        cv::Point2f& targetVelocityOut)
{
    // dX/dY/nX/nY: unit normal pointing from target (b) toward ball (a),
    // exactly as in the reference (a.position - b.position, normalised),
    // computed directly from ball.position — matching the pattern used
    // elsewhere in this codebase (e.g. the vertex-collision normal in
    // PoolPhysicsEngine::simulateSingleBall, which likewise trusts
    // ball.position directly with no re-derived contact point).
    const float dX = ball.position.x - targetPos.x;
    const float dY = ball.position.y - targetPos.y;
    const float dLen = std::sqrt(dX * dX + dY * dY);

    // dLen should never be ~0 at a genuine contact (ball and target centres
    // would have to coincide), but guard defensively against divide-by-zero
    // exactly as the rest of this codebase guards against degenerate inputs.
    if (dLen < PoolPhysicsEngine::ALMOST_ZERO) {
        targetVelocityOut = cv::Point2f(0.0f, 0.0f);
        return;
    }

    const float nX = dX / dLen;
    const float nY = dY / dLen;

    // Ball (a)'s normal/tangential decomposition.
    const float aDotN = ball.velocity.x * -nX + ball.velocity.y * -nY;
    const float nAX   = aDotN * -nX;
    const float nAY   = aDotN * -nY;
    const float tAX   = nAX - ball.velocity.x;
    const float tAY   = nAY - ball.velocity.y;

    // Target (b)'s normal/tangential decomposition — b.velocity is (0,0),
    // so bDotN, nBX, nBY are all zero; kept explicit for fidelity to the
    // reference structure.
    const float bVelX = 0.0f;
    const float bVelY = 0.0f;
    const float bDotN = bVelX * nX + bVelY * nY;
    const float nBX   = bDotN * nX;
    const float nBY   = bDotN * nY;
    const float tBX   = nBX - bVelX;
    const float tBY   = nBY - bVelY;

    ball.velocity.x = -tAX + nBX;
    ball.velocity.y = -tAY + nBY;

    targetVelocityOut.x = -tBX + nAX;
    targetVelocityOut.y = -tBY + nAY;

    // spinX/spinY/spinZ on `ball` are intentionally left untouched — the
    // reference's ballBallCollision never reads or writes spin, and the
    // target's spin is likewise never referenced (it isn't even a
    // parameter here; targetVelocityOut carries only the resulting
    // velocity, matching the reference's ball-ball response exactly).
}

// ============================================================================
// Cheap analytic pre-filtering helper
// ============================================================================

float IndirectPhysics::signedPerpendicularDistanceToRay(cv::Point2f bouncePoint,
                                                        cv::Point2f direction,
                                                        cv::Point2f targetPos)
{
    const float dirLen = std::sqrt(direction.x * direction.x
                                   + direction.y * direction.y);
    if (dirLen < PoolPhysicsEngine::ALMOST_ZERO)
        return 0.0f;   // degenerate direction — no meaningful line to test against

    const float ux = direction.x / dirLen;
    const float uy = direction.y / dirLen;

    const float toTargetX = targetPos.x - bouncePoint.x;
    const float toTargetY = targetPos.y - bouncePoint.y;

    // 2-D cross product of the (unit) direction with the vector to the
    // target — this is exactly the signed perpendicular distance from the
    // target to the infinite line through bouncePoint along direction.
    return ux * toTargetY - uy * toTargetX;
}

// ============================================================================
// Capability #2 — simulateUntilCushionOrTarget
// ============================================================================

IndirectSimResult IndirectPhysics::simulateUntilCushionOrTarget(
        SimBall     ball,
        cv::Point2f targetPos,
        float       targetRadius,
        int         maxReflections) const
{
    IndirectSimResult out;
    out.path.reserve(static_cast<size_t>(maxReflections) + 8u);
    out.path.push_back({ ball.position, IndirectPathEventType::START });

    const std::vector<cv::Point2f>& cushionPolygon = m_engine.getCushionPolygon();

    int      cushionCount = 0;
    uint32_t steps        = 0u;

    // Populates out.velocity/spin-at-contact fields from the ball's current
    // (live) state — used for every return path so the struct always
    // reflects the state at out.path.back().
    auto captureCurrentState = [&]() {
        out.cueVelocityAtContact = ball.velocity;
        out.cueSpinXAtContact    = ball.spinX;
        out.cueSpinYAtContact    = ball.spinY;
        out.cueSpinZAtContact    = ball.spinZ;
    };

    // ---- Main simulation loop — one outer iteration = one 5 ms frame,
    //      exactly mirroring PoolPhysicsEngine::simulateSingleBall(). ----
    while (PoolPhysicsEngine::isMovingOrSpinning(ball)
           && steps < PoolPhysicsEngine::MAX_SIM_STEPS) {
        ++steps;

        float timeLeft = PoolPhysicsEngine::FIXED_DT_SEC;
        bool  stopped  = false;

        // ---- Inner sub-step loop: advance through collisions within this
        //      frame, now also testing the fixed target-ball circle
        //      alongside the existing cushion edge/vertex tests. ----
        while (timeLeft > PoolPhysicsEngine::ALMOST_ZERO) {

            float       bestT      = timeLeft;
            int         bestKind   = 0;   // 0 = none, 1 = edge, 2 = vertex, 3 = target
            cv::Point2f edgeA, edgeB, hitVertex;
            size_t      bestEdgeIndex   = 0;
            size_t      bestVertexIndex = 0;

            const size_t nPts = cushionPolygon.size();

            // Search every polygon edge (wrapping last → first) — reusing
            // the engine's own ballLineCollisionTime() exactly as
            // simulateSingleBall() does.
            for (size_t i = 0; i < nPts; ++i) {
                const cv::Point2f pA = cushionPolygon[i];
                const cv::Point2f pB = cushionPolygon[(i + 1u) % nPts];

                const float t = PoolPhysicsEngine::ballLineCollisionTime(ball, pA, pB, bestT);
                if (t < bestT) {
                    bestT    = t;
                    bestKind = 1;
                    edgeA    = pA;
                    edgeB    = pB;
                    bestEdgeIndex = i;
                }
            }

            // Search every polygon vertex — reusing the engine's own
            // ballPointCollisionTime() exactly as simulateSingleBall() does.
            for (size_t i = 0; i < nPts; ++i) {
                const cv::Point2f v = cushionPolygon[i];
                const float t = PoolPhysicsEngine::ballPointCollisionTime(ball, v, bestT);
                if (t < bestT) {
                    bestT     = t;
                    bestKind  = 2;
                    hitVertex = v;
                    bestVertexIndex = i;
                }
            }

            // Test the fixed target-ball circle using capability #1's
            // collision-time formula, alongside the cushion tests above.
            {
                const float t = ballBallCollisionTime(ball, targetPos, targetRadius, bestT);
                if (t < bestT) {
                    bestT    = t;
                    bestKind = 3;
                }
            }

            // Advance the ball to the earliest collision (or to the end of
            // the remaining time if no collision was found) — identical to
            // simulateSingleBall().
            ball.position.x += ball.velocity.x * bestT;
            ball.position.y += ball.velocity.y * bestT;
            timeLeft -= bestT;

            if (bestKind == 3) {
                // ---- Target-ball collision is the earliest event: record
                //      a TARGET point at the contact position, capture the
                //      ball's exact velocity/spin at contact, and return
                //      immediately WITHOUT applying the ball-ball collision
                //      response (that is left to the caller). ----
                out.path.push_back({ ball.position, IndirectPathEventType::TARGET });
                captureCurrentState();
                return out;
            }

            // ---- Resolve cushion collision (edge or vertex) — identical
            //      logic to simulateSingleBall(), reusing the engine's own
            //      calculateTheta() and ballLineCollision(). ----
            int resolvedEdgeIndex = -1;
            if (bestKind == 1) {
                const float theta = -PoolPhysicsEngine::calculateTheta(edgeB.x - edgeA.x,
                                                                       edgeB.y - edgeA.y);
                PoolPhysicsEngine::ballLineCollision(ball, theta);
                resolvedEdgeIndex = static_cast<int>(bestEdgeIndex);

            } else if (bestKind == 2) {
                const float nx    = hitVertex.x - ball.position.x;
                const float ny    = hitVertex.y - ball.position.y;
                const float theta = -PoolPhysicsEngine::calculateTheta(ny, -nx);
                PoolPhysicsEngine::ballLineCollision(ball, theta);

                const size_t prevEdgeIndex = (bestVertexIndex + nPts - 1u) % nPts;
                const size_t nextEdgeIndex = bestVertexIndex;
                const cv::Point2f prevA = cushionPolygon[prevEdgeIndex];
                const cv::Point2f prevB = cushionPolygon[(prevEdgeIndex + 1u) % nPts];
                const cv::Point2f nextA = cushionPolygon[nextEdgeIndex];
                const cv::Point2f nextB = cushionPolygon[(nextEdgeIndex + 1u) % nPts];

                const cv::Point2f prevMid((prevA.x + prevB.x) * 0.5f, (prevA.y + prevB.y) * 0.5f);
                const cv::Point2f nextMid((nextA.x + nextB.x) * 0.5f, (nextA.y + nextB.y) * 0.5f);

                const float distToPrev = pointDist(hitVertex, prevMid);
                const float distToNext = pointDist(hitVertex, nextMid);

                resolvedEdgeIndex = static_cast<int>(
                        (distToPrev <= distToNext) ? prevEdgeIndex : nextEdgeIndex);
            }

            // ---- Record cushion event and check the maxReflections stop
            //      condition — identical to simulateSingleBall(), except
            //      there is no pocket-capture check (pockets are not
            //      relevant to indirect target-ball solving here; the
            //      target-ball circle test above is this primitive's
            //      analogous "did we reach the goal" check). ----
            if (bestKind != 0) {
                ++cushionCount;
                out.path.push_back({ ball.position, IndirectPathEventType::CUSHION, resolvedEdgeIndex });

                if (cushionCount >= maxReflections + 1) {
                    stopped = true;
                    break;
                }
            }
            // If bestKind == 0 the entire remaining time was consumed
            // without any collision; the inner while condition
            // (timeLeft > ALMOST_ZERO) will now be false and we exit
            // naturally, exactly as in simulateSingleBall().
        } // inner sub-step loop

        if (stopped) {
            captureCurrentState();
            return out;
        }

        // ---- Apply cloth friction for this 5 ms frame — identical to
        //      simulateSingleBall(), reusing the engine's own
        //      tableBallFriction(). ----
        PoolPhysicsEngine::tableBallFriction(ball, PoolPhysicsEngine::FIXED_DT_SEC);

        // Only record a new point when the ball has truly come to rest.
        if (!PoolPhysicsEngine::isMovingOrSpinning(ball)) {
            out.path.push_back({ ball.position, IndirectPathEventType::REST });
            captureCurrentState();
            return out;
        }

        // NOTE: unlike simulateSingleBall(), there is no post-friction
        // pocket-capture check here — pockets are not part of this
        // primitive's contract (see the class-level Doxygen comment: this
        // call never produces a POCKET-equivalent point).

        // Otherwise: no event this frame — continue to the next iteration,
        // exactly as in simulateSingleBall().

    } // outer fixed-dt loop

    // ---- Safety: MAX_SIM_STEPS reached or ball already at rest on entry —
    //      identical to simulateSingleBall(). ----
    if (out.path.back().type != IndirectPathEventType::REST
        && out.path.back().type != IndirectPathEventType::TARGET)
    {
        out.path.push_back({ ball.position, IndirectPathEventType::REST });
    }
    captureCurrentState();

    return out;
}