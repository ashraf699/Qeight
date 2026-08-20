/**
 * @file PoolPhysicsEngine.cpp
 * @brief Implementation of every method declared in PoolPhysicsEngine.h.
 *
 * Pure, portable C++ — no Android/JNI/logging dependencies.
 * All physics quantities are in CGS units (cm, g, s) unless noted otherwise.
 */

#include "PoolPhysicsEngine.h"

#include <cmath>
#include <algorithm>
#include <limits>

// ---------------------------------------------------------------------------
// Internal convenience
// ---------------------------------------------------------------------------

namespace {

/// π, sourced from OpenCV's CV_PI so the value is consistent with the rest
/// of any OpenCV-using project, falling back to M_PI if needed.
    static constexpr double PI_D = CV_PI;          // double precision for atan2 etc.
    static constexpr float  PI_F = static_cast<float>(CV_PI);

/// Returns Euclidean length of a 2-D vector.
    inline float vec2Length(cv::Point2f v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y);
    }

/// Returns Euclidean distance between two 2-D points.
    inline float pointDist(cv::Point2f a, cv::Point2f b)
    {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }

} // anonymous namespace

// ============================================================================
// Constructor — builds and caches geometry
// ============================================================================

PoolPhysicsEngine::PoolPhysicsEngine()
        : m_cushionPolygon(buildCushionPolygon())
        , m_pocketCenters (buildPocketCenters())
{}

const std::vector<cv::Point2f>& PoolPhysicsEngine::getCushionPolygon() const
{
    return m_cushionPolygon;
}

const std::vector<cv::Point2f>& PoolPhysicsEngine::getPocketCenters() const
{
    return m_pocketCenters;
}

// ============================================================================
// Private static geometry builders
// ============================================================================

std::vector<cv::Point2f> PoolPhysicsEngine::buildCushionPolygon()
{
    // -----------------------------------------------------------------------
    // Fixed local profiles
    // -----------------------------------------------------------------------

    // Corner pocket-mouth cutout profile (7 points).
    // Each point is (x, y) in a local coordinate system; the assembly rules
    // below translate / mirror them into table-centred cm-space.
    const std::vector<cv::Point2f> cornerProfile = {
            {  0.0f,  10.6f },
            { -9.9f,  -0.6f },
            {-11.2f,  -5.7f },
            { -9.7f,  -9.7f },
            { -5.7f, -11.2f },
            { -0.6f,  -9.9f },
            { 10.6f,   0.0f }
    };

    // Side (middle-rail) pocket-mouth cutout profile (9 points).
    const std::vector<cv::Point2f> sideProfile = {
            { -7.9f,   0.0f },
            { -6.2f,  -5.1f },
            { -5.8f,  -9.2f },
            { -3.8f, -11.9f },
            {  0.0f, -13.2f },
            {  3.8f, -11.9f },
            {  5.8f,  -9.2f },
            {  6.2f,  -5.1f },
            {  7.9f,   0.0f }
    };

    std::vector<cv::Point2f> poly;
    // Maximum vertices: 7*4 + 9*2 = 46
    poly.reserve(46);

    // -----------------------------------------------------------------------
    // 1. Top-left corner (cornerProfile, forward order)
    //    transform: (c.x - 127,  c.y - 63.5)
    // -----------------------------------------------------------------------
    for (const auto& c : cornerProfile)
        poly.emplace_back(c.x - 127.0f, c.y - 63.5f);

    // -----------------------------------------------------------------------
    // 2. Top-middle side pocket (sideProfile, forward order)
    //    transform: (s.x,  s.y - 63.5)
    // -----------------------------------------------------------------------
    for (const auto& s : sideProfile)
        poly.emplace_back(s.x, s.y - 63.5f);

    // -----------------------------------------------------------------------
    // 3. Top-right corner (cornerProfile, REVERSE order)
    //    transform: (-c.x + 127,  c.y - 63.5)
    // -----------------------------------------------------------------------
    for (auto it = cornerProfile.rbegin(); it != cornerProfile.rend(); ++it)
        poly.emplace_back(-it->x + 127.0f, it->y - 63.5f);

    // -----------------------------------------------------------------------
    // 4. Bottom-right corner (cornerProfile, forward order)
    //    transform: (-c.x + 127,  -c.y + 63.5)
    // -----------------------------------------------------------------------
    for (const auto& c : cornerProfile)
        poly.emplace_back(-c.x + 127.0f, -c.y + 63.5f);

    // -----------------------------------------------------------------------
    // 5. Bottom-middle side pocket (sideProfile, REVERSE order)
    //    transform: (s.x,  -s.y + 63.5)
    // -----------------------------------------------------------------------
    for (auto it = sideProfile.rbegin(); it != sideProfile.rend(); ++it)
        poly.emplace_back(it->x, -it->y + 63.5f);

    // -----------------------------------------------------------------------
    // 6. Bottom-left corner (cornerProfile, REVERSE order)
    //    transform: (c.x - 127,  -c.y + 63.5)
    // -----------------------------------------------------------------------
    for (auto it = cornerProfile.rbegin(); it != cornerProfile.rend(); ++it)
        poly.emplace_back(it->x - 127.0f, -it->y + 63.5f);

    return poly;
}

std::vector<cv::Point2f> PoolPhysicsEngine::buildPocketCenters()
{
    // Exact order as specified in the header contract.
    return {
            cv::Point2f(-130.8f, -67.3f),  // bottom-left  corner
            cv::Point2f(   0.0f, -71.0f),  // bottom-middle side
            cv::Point2f( 130.8f, -67.3f),  // bottom-right corner
            cv::Point2f( 130.8f,  67.3f),  // top-right    corner
            cv::Point2f(   0.0f,  71.0f),  // top-middle   side
            cv::Point2f(-130.8f,  67.3f)   // top-left     corner
    };
}

// ============================================================================
// Static helper — calculateTheta
// ============================================================================

float PoolPhysicsEngine::calculateTheta(float x, float y)
{
    if (x == 0.0f)
        return (y >= 0.0f) ? (PI_F / 2.0f) : (3.0f * PI_F / 2.0f);

    float a = std::atan(y / x);
    return (x < 0.0f) ? (a + PI_F) : a;
}

// ============================================================================
// Static helper — ballLineCollisionTime
// ============================================================================

float PoolPhysicsEngine::ballLineCollisionTime(const SimBall&  ball,
                                               cv::Point2f     lineStart,
                                               cv::Point2f     lineEnd,
                                               float           maxT)
{
    const float INF = std::numeric_limits<float>::infinity();

    // Ball must be moving.
    if (ball.velocity.x == 0.0f && ball.velocity.y == 0.0f)
        return INF;

    // Edge direction vector and its length.
    const cv::Point2f rl    = lineEnd - lineStart;
    const float       lineLen = vec2Length(rl);
    if (lineLen < ALMOST_ZERO)
        return INF;

    // Inward-facing edge normal (rotated 90° CCW from rl, then normalised).
    const cv::Point2f nl(-rl.y / lineLen, rl.x / lineLen);

    // Ball position relative to lineStart, offset by one radius along normal.
    const cv::Point2f rb = ball.position - lineStart - nl * ball.radius;

    // Negated velocity (used as the "ray" direction in the 2-D intersection).
    const float nvx = -ball.velocity.x;
    const float nvy = -ball.velocity.y;

    // Cross product of rl and (-velocity): scalar determinant.
    const float det = rl.x * nvy - rl.y * nvx;
    if (det == 0.0f)
        return INF;   // parallel — no intersection

    // Parameter along the segment [0,1].
    const float d1      = nvy * rb.x - nvx * rb.y;
    const float d1DivD  = d1 / det;
    if (d1DivD <= 0.0f || d1DivD >= 1.0f)
        return INF;   // contact point falls outside the segment

    // Time of contact.
    const float d2 = rl.x * rb.y - rl.y * rb.x;
    const float t  = d2 / det;
    if (t <= 0.0f || t - ALMOST_ZERO > maxT)
        return INF;

    // Ball must be moving toward this edge (dot of inward normal and velocity ≤ 0).
    const float normalDot = nl.x * ball.velocity.x + nl.y * ball.velocity.y;
    if (normalDot > 0.0f)
        return INF;   // moving away from this edge

    return t;
}

// ============================================================================
// Static helper — ballPointCollisionTime
// ============================================================================

float PoolPhysicsEngine::ballPointCollisionTime(const SimBall& ball,
                                                cv::Point2f    point,
                                                float          maxT)
{
    const float INF = std::numeric_limits<float>::infinity();

    const float vx = ball.velocity.x;
    const float vy = ball.velocity.y;

    // Coefficient A of the quadratic (speed squared).
    const float A = vx * vx + vy * vy;
    if (A < ALMOST_ZERO)
        return INF;

    const float dx = point.x - ball.position.x;
    const float dy = point.y - ball.position.y;

    // B = -2·(velocity · displacement)
    const float B = -2.0f * vx * dx - 2.0f * vy * dy;

    // C = |displacement|² (distance² from ball centre to vertex)
    const float C = dx * dx + dy * dy;

    // Quick rejection: closest approach is still outside ball radius.
    // Closest approach distance² = C - B²/(4A).
    const float r2 = ball.radius * ball.radius;
    if (-B * B / (4.0f * A) + C >= r2)
        return INF;

    const float disc = B * B - 4.0f * A * (C - r2);
    if (disc < 0.0f)
        return INF;

    const float t = (-B - std::sqrt(disc)) / (2.0f * A);

    // t must be positive, within the time window, and the ball must be
    // approaching the vertex (B > 0 means the velocity points away).
    if (t < 0.0f || t - ALMOST_ZERO > maxT || B > 0.0f)
        return INF;

    return t;
}

// ============================================================================
// Static helper — ballLineCollision
// ============================================================================

void PoolPhysicsEngine::ballLineCollision(SimBall& ball, float theta)
{
    const float cosT = std::cos(theta);
    const float sinT = std::sin(theta);

    // ---- Rotate velocity and spin into the edge-local frame ----
    float vX =  cosT * ball.velocity.x - sinT * ball.velocity.y;
    float vY =  sinT * ball.velocity.x + cosT * ball.velocity.y;

    float sX =  cosT * ball.spinX - sinT * ball.spinY;
    float sY =  sinT * ball.spinX + cosT * ball.spinY;

    // Transfer tangential cushion-contact velocity to sX (grip effect).
    sX -= vY * CUSHION_SPIN_RATIO / ball.radius;

    // ---- Friction impulse along the cushion face (X direction) ----
    // Relative contact-point velocity in X (ball surface vs cushion).
    const float bcvX   = vX - ball.spinZ * ball.radius;
    const float bcvLen = std::fabs(bcvX);
    const float bcvDir = (bcvX > 0.0f) ? 1.0f : -1.0f;

    // Maximum Δv that can be applied before the sliding stops.
    const float maxDv    = bcvLen / FIVE_DIV_TWO;
    // Friction impulse magnitude proportional to the normal force (vY).
    const float impactDv = 2.0f * COEFFICIENT_OF_SLIDING_FRICTION * std::fabs(vY);
    const float dvX      = -bcvDir * std::min(maxDv, impactDv);

    vX      += dvX / FIVE_DIV_TWO;
    ball.spinZ -= FIVE_DIV_TWO * dvX / ball.radius;

    // ---- Reflect normal component with restitution ----
    vY = -vY * COEFFICIENT_OF_RESTITUTION;

    // ---- Rotate back to world frame ----
    ball.velocity.x =  cosT * vX + sinT * vY;
    ball.velocity.y = -sinT * vX + cosT * vY;

    ball.spinX =  cosT * sX + sinT * sY;
    ball.spinY = -sinT * sX + cosT * sY;
    // spinZ was already updated in-place above.
}

// ============================================================================
// Static helper — tableBallFriction
// ============================================================================

void PoolPhysicsEngine::tableBallFriction(SimBall& ball, float dt)
{
    if (!isMovingOrSpinning(ball))
        return;

    // ---- Sliding phase ----
    // Relative velocity of ball surface w.r.t. cloth.
    const float tbvX = -ball.velocity.x - ball.spinY * ball.radius;
    const float tbvY = -ball.velocity.y + ball.spinX * ball.radius;
    const float tbvLen = std::sqrt(tbvX * tbvX + tbvY * tbvY);

    // Time until sliding stops (ball reaches the rolling condition).
    const float tEq = TWO_DIV_SEVEN * tbvLen
                      / (COEFFICIENT_OF_SLIDING_FRICTION * GRAVITATIONAL_FORCE);

    if (tEq > ALMOST_ZERO) {
        const float slideT    = std::min(tEq, dt);
        const float reduction = COEFFICIENT_OF_SLIDING_FRICTION
                                * GRAVITATIONAL_FORCE
                                * slideT;

        if (tbvLen > ALMOST_ZERO) {
            // Unit vector along sliding direction, scaled by impulse magnitude.
            const float ax = tbvX * reduction / tbvLen;
            const float ay = tbvY * reduction / tbvLen;

            ball.velocity.x += ax;
            ball.velocity.y += ay;

            // Spin change from sliding friction torque.
            ball.spinX -= FIVE_DIV_TWO * ay / ball.radius;
            ball.spinY += FIVE_DIV_TWO * ax / ball.radius;
        }
    }

    // ---- Rolling phase (begins after tEq, if tEq < dt) ----
    if (tEq < dt) {
        const float rollT     = dt - tEq;
        const float reduction = COEFFICIENT_OF_ROLLING_FRICTION
                                * GRAVITATIONAL_FORCE
                                * rollT;

        const float speed = std::sqrt(ball.velocity.x * ball.velocity.x
                                      + ball.velocity.y * ball.velocity.y);
        if (speed > ALMOST_ZERO) {
            const float scale = std::max(0.0f, 1.0f - reduction / speed);
            ball.velocity.x *= scale;
            ball.velocity.y *= scale;
        }

        // Enforce rolling constraint: spin tracks velocity.
        ball.spinX =  ball.velocity.y / ball.radius;
        ball.spinY = -ball.velocity.x / ball.radius;
    }

    // ---- Spinning (pivot) phase — acts throughout dt ----
    const float dSpin = COEFFICIENT_OF_SPINNING_FRICTION / FIVE_DIV_TWO
                        * GRAVITATIONAL_FORCE * dt;
    if (ball.spinZ > 0.0f)
        ball.spinZ = std::max(0.0f, ball.spinZ - dSpin);
    else
        ball.spinZ = std::min(0.0f, ball.spinZ + dSpin);
}

// ============================================================================
// Static helper — applyCueStats
// ============================================================================

CueStatResult PoolPhysicsEngine::applyCueStats(int forceStat, int spinStat)
{
    // Power: increase or decrease relative to default, depending on stat delta.
    const float powerRatio = (forceStat < CUE_DEFAULT_STAT)
                             ? CUE_FORCE_DECREASE_PCT * static_cast<float>(forceStat - CUE_DEFAULT_STAT)
                             : CUE_FORCE_INCREASE_PCT * static_cast<float>(forceStat - CUE_DEFAULT_STAT);

    const float maxPower = (1.0f + powerRatio) * CUE_DEFAULT_PROPERTIES_MAX_POWER;

    // Spin scale: same pattern.
    const float spinRatio = (spinStat < CUE_DEFAULT_STAT)
                            ? CUE_SPIN_DECREASE_PCT * static_cast<float>(spinStat - CUE_DEFAULT_STAT)
                            : CUE_SPIN_INCREASE_PCT * static_cast<float>(spinStat - CUE_DEFAULT_STAT);

    const float spinScale = (1.0f + spinRatio) * CUE_DEFAULT_PROPERTIES_SPIN;

    return { maxPower, spinScale };
}

// ============================================================================
// Static helper — isMovingOrSpinning
// ============================================================================

bool PoolPhysicsEngine::isMovingOrSpinning(const SimBall& ball)
{
    return ball.velocity.x != 0.0f
           || ball.velocity.y != 0.0f
           || ball.spinX      != 0.0f
           || ball.spinY      != 0.0f
           || ball.spinZ      != 0.0f;
}

// ============================================================================
// Primary simulation — simulateSingleBall
// ============================================================================

std::vector<PathPoint> PoolPhysicsEngine::simulateSingleBall(SimBall ball,
                                                             int     maxReflections) const
{
    // ---- Initialise result with the starting position ----
    std::vector<PathPoint> result;
    result.reserve(maxReflections + 8);
    result.push_back({ ball.position, PathEventType::START });

    int      cushionCount = 0;
    uint32_t steps        = 0u;

    // ---- Pocket-capture lambda (captures result, ball, m_pocketCenters by ref) ----
    auto checkPocketCapture = [&]() -> bool {
        for (const auto& pk : m_pocketCenters) {
            if (pointDist(ball.position, pk) < POCKET_RADIUS_CM) {
                result.push_back({ pk, PathEventType::POCKET });
                return true;
            }
        }
        return false;
    };

    // ---- Main simulation loop — one outer iteration = one 5 ms frame ----
    while (isMovingOrSpinning(ball) && steps < MAX_SIM_STEPS) {
        ++steps;

        float timeLeft = FIXED_DT_SEC;
        bool  stopped  = false;

        // ---- Inner sub-step loop: advance through collisions within this frame ----
        while (timeLeft > ALMOST_ZERO) {

            float       bestT      = timeLeft;
            int         bestKind   = 0;          // 0 = none, 1 = edge, 2 = vertex
            cv::Point2f edgeA, edgeB, hitVertex;
            size_t      bestEdgeIndex   = 0;
            size_t      bestVertexIndex = 0;

            const size_t nPts = m_cushionPolygon.size();

            // Search every polygon edge (wrapping last → first).
            for (size_t i = 0; i < nPts; ++i) {
                const cv::Point2f pA = m_cushionPolygon[i];
                const cv::Point2f pB = m_cushionPolygon[(i + 1u) % nPts];

                const float t = ballLineCollisionTime(ball, pA, pB, bestT);
                if (t < bestT) {
                    bestT    = t;
                    bestKind = 1;
                    edgeA    = pA;
                    edgeB    = pB;
                    bestEdgeIndex = i;
                }
            }

            // Search every polygon vertex.
            for (size_t i = 0; i < nPts; ++i) {
                const cv::Point2f v = m_cushionPolygon[i];
                const float t = ballPointCollisionTime(ball, v, bestT);
                if (t < bestT) {
                    bestT     = t;
                    bestKind  = 2;
                    hitVertex = v;
                    bestVertexIndex = i;
                }
            }

            // Advance the ball to the earliest collision (or to the end of
            // the remaining time if no collision was found).
            ball.position.x += ball.velocity.x * bestT;
            ball.position.y += ball.velocity.y * bestT;
            timeLeft -= bestT;

            // ---- Resolve collision ----
            int resolvedEdgeIndex = -1;
            if (bestKind == 1) {
                // Edge collision: compute inward-normal angle, then reflect.
                const float theta = -calculateTheta(edgeB.x - edgeA.x,
                                                    edgeB.y - edgeA.y);
                ballLineCollision(ball, theta);
                resolvedEdgeIndex = static_cast<int>(bestEdgeIndex);

            } else if (bestKind == 2) {
                // Vertex collision: normal points from vertex toward ball centre.
                const float nx    = hitVertex.x - ball.position.x;
                const float ny    = hitVertex.y - ball.position.y;
                const float theta = -calculateTheta(ny, -nx);
                ballLineCollision(ball, theta);

                // Determine nearest of the two edges adjacent to this vertex.
                const size_t prevEdgeIndex = (bestVertexIndex + nPts - 1u) % nPts;
                const size_t nextEdgeIndex = bestVertexIndex;
                const cv::Point2f prevA = m_cushionPolygon[prevEdgeIndex];
                const cv::Point2f prevB = m_cushionPolygon[(prevEdgeIndex + 1u) % nPts];
                const cv::Point2f nextA = m_cushionPolygon[nextEdgeIndex];
                const cv::Point2f nextB = m_cushionPolygon[(nextEdgeIndex + 1u) % nPts];

                const cv::Point2f prevMid((prevA.x + prevB.x) * 0.5f, (prevA.y + prevB.y) * 0.5f);
                const cv::Point2f nextMid((nextA.x + nextB.x) * 0.5f, (nextA.y + nextB.y) * 0.5f);

                const float distToPrev = pointDist(hitVertex, prevMid);
                const float distToNext = pointDist(hitVertex, nextMid);

                resolvedEdgeIndex = static_cast<int>(
                        (distToPrev <= distToNext) ? prevEdgeIndex : nextEdgeIndex);
            }

            // ---- Record cushion event and check stop conditions ----
            if (bestKind != 0) {
                ++cushionCount;
                result.push_back({ ball.position, PathEventType::CUSHION, resolvedEdgeIndex });

                if (checkPocketCapture()) {
                    stopped = true;
                    break;
                }

                // maxReflections: the path may contain at most
                // (maxReflections + 1) CUSHION points total.
                if (cushionCount >= maxReflections + 1) {
                    stopped = true;
                    break;
                }
            }
            // If bestKind == 0 the entire remaining time was consumed without
            // a collision; the inner while condition (timeLeft > ALMOST_ZERO)
            // will now be false and we exit naturally.
        } // inner sub-step loop

        if (stopped)
            return result;

        // ---- Apply cloth friction for this 5 ms frame ----
        tableBallFriction(ball, FIXED_DT_SEC);

        // Only record a new point when the ball has truly come to rest.
        if (!isMovingOrSpinning(ball)) {
            result.push_back({ ball.position, PathEventType::REST });
            return result;
        }

        // Pocket check after friction (ball may have rolled into a pocket
        // without hitting a cushion this frame).
        if (checkPocketCapture())
            return result;

        // Otherwise: no event this frame — continue to the next iteration.
        // (We intentionally do NOT push a per-frame intermediate point here.)

    } // outer fixed-dt loop

    // ---- Safety: MAX_SIM_STEPS reached or ball already at rest on entry ----
    if (result.back().type != PathEventType::REST
        && result.back().type != PathEventType::POCKET)
    {
        result.push_back({ ball.position, PathEventType::REST });
    }

    return result;
}