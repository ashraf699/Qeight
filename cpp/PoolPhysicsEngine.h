/**
 * @file PoolPhysicsEngine.h
 * @brief Self-contained single-ball pool/billiards physics engine.
 *
 * Pure geometry and physics — zero dependency on rendering, JNI, or any
 * Android-specific API.  The only external dependency is cv::Point2f from
 * opencv2/core.hpp; everything else is standard C++.
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/**
 * @brief Complete kinematic state of a single pool ball.
 *
 * All linear quantities are in centimetres / seconds.
 * Spin components (spinX, spinY, spinZ) are expressed in rad/s-equivalent
 * units consistent with the friction/collision formulae used internally.
 */
struct SimBall {
    cv::Point2f position;          ///< cm-space, table-centred coordinate system
    cv::Point2f velocity;          ///< cm/s
    float spinX  = 0.f;            ///< rad/s-equivalent side-spin (x-axis)
    float spinY  = 0.f;            ///< rad/s-equivalent side-spin (y-axis)
    float spinZ  = 0.f;            ///< rad/s-equivalent top/back-spin (z-axis)
    float radius = 3.0f;           ///< Ball radius in cm
};

/**
 * @brief Categorises a recorded waypoint on the simulated ball path.
 */
enum class PathEventType {
    START,    ///< Initial position supplied by the caller
    CUSHION,  ///< Ball made contact with a cushion edge or vertex
    POCKET,   ///< Ball entered a pocket (simulation terminates)
    REST      ///< Ball came to a complete stop (simulation terminates)
};

/**
 * @brief A single waypoint in the simulated ball path.
 */
struct PathPoint {
    cv::Point2f position;  ///< cm-space position of the event
    PathEventType type;    ///< Nature of the event at this waypoint
    int cushionSideIndex = -1;  ///< For CUSHION events, index (0..45) into
    ///< m_cushionPolygon of the edge struck (or
    ///< nearest edge, for vertex collisions).
    ///< -1 for START/POCKET/REST events.
};

/**
 * @brief Output of applyCueStats(): derived cue-power and spin-scale values.
 */
struct CueStatResult {
    float maxPower;   ///< Maximum achievable launch speed in cm/s
    float spinScale;  ///< Multiplier applied to the raw spin input
};

// ---------------------------------------------------------------------------
// PoolPhysicsEngine
// ---------------------------------------------------------------------------

/**
 * @brief Self-contained, reusable physics engine for single-ball pool
 *        simulation on a standard 8-foot table.
 *
 * Construct once; the constructor precomputes and caches the cushion polygon
 * and pocket-centre geometry.  All simulation methods are then available as
 * const (or static) calls with no further setup required.
 *
 * Thread-safety: const methods are safe to call concurrently from multiple
 * threads after construction.
 */
class PoolPhysicsEngine
{
public:

    // -----------------------------------------------------------------------
    // Physical / geometric constants
    // -----------------------------------------------------------------------

    /// Threshold below which a floating-point quantity is treated as zero.
    static constexpr float ALMOST_ZERO                     = 1e-11f;

    /// Gravitational acceleration used in friction calculations (cm/s²).
    static constexpr float GRAVITATIONAL_FORCE             = 980.0f;

    /// Kinetic (sliding) friction coefficient between ball and cloth.
    static constexpr float COEFFICIENT_OF_SLIDING_FRICTION = 0.2f;

    /// Rolling friction coefficient between ball and cloth.
    static constexpr float COEFFICIENT_OF_ROLLING_FRICTION = 0.0111f;

    /// Spinning (pivot) friction coefficient between ball and cloth.
    static constexpr float COEFFICIENT_OF_SPINNING_FRICTION = 0.025f;

    /// Coefficient of restitution for ball–cushion collisions.
    static constexpr float COEFFICIENT_OF_RESTITUTION      = 0.804f;

    /// Fraction of pre-collision tangential speed transferred as spin during
    /// a cushion collision.
    static constexpr float CUSHION_SPIN_RATIO              = 0.54f;

    /// Convenience constant: 5/2, used in spin-to-velocity coupling.
    static constexpr float FIVE_DIV_TWO                    = 2.5f;

    /// Convenience constant: 2/7, used in rolling-friction spin transfer.
    static constexpr float TWO_DIV_SEVEN                   = (2.0f / 7.0f);

    // -----------------------------------------------------------------------
    // Table geometry constants (cm)
    // -----------------------------------------------------------------------

    /// Full playfield length along the long axis (cm).
    static constexpr float TABLE_LENGTH_CM   = 254.0f;

    /// Full playfield width along the short axis (cm).
    static constexpr float TABLE_WIDTH_CM    = 127.0f;

    /// Capture radius of each pocket centre; ball is pocketed when closer
    /// than this distance (cm).
    static constexpr float POCKET_RADIUS_CM  = 8.0f;

    // -----------------------------------------------------------------------
    // Cue-stat constants
    // -----------------------------------------------------------------------

    /// Default maximum launch speed for a cue at base stats (cm/s).
    static constexpr float CUE_DEFAULT_PROPERTIES_MAX_POWER = 685.0f;

    /// Default spin scale for a cue at base stats.
    static constexpr float CUE_DEFAULT_PROPERTIES_SPIN      = (20.0f / 29.0f);

    /// Baseline integer stat level for a default cue.
    static constexpr int   CUE_DEFAULT_STAT                  = 3;

    /// Fractional power increase per stat point above default.
    static constexpr float CUE_FORCE_INCREASE_PCT            = 0.03f;

    /// Fractional power decrease per stat point below default.
    static constexpr float CUE_FORCE_DECREASE_PCT            = 0.08f;

    /// Fractional spin increase per stat point above default.
    static constexpr float CUE_SPIN_INCREASE_PCT             = 0.04f;

    /// Fractional spin decrease per stat point below default.
    static constexpr float CUE_SPIN_DECREASE_PCT             = 0.15f;

    // -----------------------------------------------------------------------
    // Simulation time-stepping constants
    // -----------------------------------------------------------------------

    /// Fixed integration time-step used throughout the simulation (seconds).
    static constexpr float    FIXED_DT_SEC    = 0.005f;

    /// Hard upper bound on the number of fixed-dt frames per simulation call;
    /// prevents runaway loops on degenerate inputs.
    static constexpr uint32_t MAX_SIM_STEPS   = 20000u;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Constructs the engine and precomputes the cushion polygon and
     *        pocket-centre list.
     *
     * The table is centred at the origin: x ∈ [−127, 127] cm,
     * y ∈ [−63.5, 63.5] cm.  Corner- and side-pocket-mouth notches are
     * embedded in the polygon using fixed local profiles (see implementation).
     * These two geometry sets are cached and reused for every subsequent
     * simulation call.
     */
    PoolPhysicsEngine();

    // -----------------------------------------------------------------------
    // Primary simulation interface
    // -----------------------------------------------------------------------

    /**
     * @brief Simulates a single ball rolling, spinning, and bouncing under
     *        friction until it pockets, comes to rest, or a stop condition
     *        is reached.
     *
     * Contract (enforced in the .cpp implementation):
     *
     * - Uses a fixed time-step of FIXED_DT_SEC seconds per frame, with a hard
     *   ceiling of MAX_SIM_STEPS frames to prevent infinite loops.
     *
     * - On each frame, tableBallFriction() is applied to update the ball's
     *   velocity and spin, then a pocket-proximity check is performed.  If
     *   the ball centre lies within POCKET_RADIUS_CM of any pocket centre, a
     *   final PathPoint of type POCKET is appended (position = pocket centre)
     *   and the simulation terminates immediately.
     *
     * - Before each frame's friction step, the edges and vertices of
     *   m_cushionPolygon are tested for collision within the current timestep
     *   using ballLineCollisionTime() and ballPointCollisionTime().  When a
     *   collision is found, the ball is advanced to the collision time,
     *   ballLineCollision() (or the vertex-reflection equivalent) is applied,
     *   and a PathPoint of type CUSHION is recorded.
     *
     * - @p maxReflections controls how many CUSHION events may be recorded
     *   after the initial straight leg.  The path may therefore contain at
     *   most (maxReflections + 1) CUSHION-type points in total.  The moment
     *   the (maxReflections + 1)-th CUSHION event is resolved and recorded,
     *   simulation stops immediately without further integration.
     *
     * - When the ball's velocity and all spin components reach exactly zero
     *   (as determined by isMovingOrSpinning()), a final PathPoint of type
     *   REST is appended and the simulation terminates.
     *
     * - If MAX_SIM_STEPS is exhausted without another stop condition, a REST
     *   point at the ball's current position is appended and the simulation
     *   terminates.
     *
     * - The first element of the returned vector is always
     *   {ball.position, PathEventType::START} — the unmodified input position.
     *
     * @param ball            Initial ball state (position, velocity, spin).
     * @param maxReflections  Maximum number of cushion bounces to simulate
     *                        after the first straight leg (≥ 0).
     * @return Ordered list of path waypoints from START through to the
     *         terminal POCKET or REST event.
     */
    std::vector<PathPoint> simulateSingleBall(SimBall ball,
                                              int     maxReflections) const;

    /**
     * @brief Returns the cached cushion polygon in table-centred cm-space.
     * @return Const reference to m_cushionPolygon.
     */
    const std::vector<cv::Point2f>& getCushionPolygon() const;

    /**
     * @brief Returns the cached pocket-centre list in table-centred cm-space.
     *
     * Order: bottom-left corner, bottom-middle side, bottom-right corner,
     *        top-right corner, top-middle side, top-left corner — exactly
     *        as documented on m_pocketCenters and built by
     *        buildPocketCenters().
     *
     * This is the SAME list simulateSingleBall()'s checkPocketCapture()
     * lambda tests against internally. Any caller that wants to target a
     * specific pocket for simulation purposes (e.g. IndirectShotSolver via
     * QeightJNI) must use a point from THIS list, not an independently
     * derived or pixel-converted coordinate — only these exact points are
     * within POCKET_RADIUS_CM capture range of a pocket-bound simulated ball.
     *
     * @return Const reference to m_pocketCenters (always size 6).
     */
    const std::vector<cv::Point2f>& getPocketCenters() const;

    // -----------------------------------------------------------------------
    // Static helper methods
    // -----------------------------------------------------------------------

    /**
     * @brief Computes the angle (in radians) of the vector (x, y) from the
     *        positive x-axis, following the convention used by the friction
     *        and collision routines in this engine.
     *
     * @param x  Horizontal component.
     * @param y  Vertical component.
     * @return   Angle in radians in the range [0, 2π).
     */
    static float calculateTheta(float x, float y);

    /**
     * @brief Computes the earliest time t ∈ (0, maxT] at which @p ball's
     *        leading edge will contact the finite line segment
     *        [@p lineStart, @p lineEnd].
     *
     * Returns a value > maxT (or a sentinel such as maxT + 1) when no
     * collision occurs within the interval.
     *
     * @param ball       Current ball state (position, velocity, radius).
     * @param lineStart  First endpoint of the cushion segment (cm).
     * @param lineEnd    Second endpoint of the cushion segment (cm).
     * @param maxT       Upper bound of the time window to test (seconds).
     * @return           Time of first contact in seconds, or > maxT if none.
     */
    static float ballLineCollisionTime(const SimBall&  ball,
                                       cv::Point2f     lineStart,
                                       cv::Point2f     lineEnd,
                                       float           maxT);

    /**
     * @brief Computes the earliest time t ∈ (0, maxT] at which @p ball's
     *        leading edge will contact a single point (cushion vertex).
     *
     * Used for pocket-mouth corners and polygon vertices where two cushion
     * segments meet at an angle that requires vertex-collision handling.
     *
     * @param ball   Current ball state (position, velocity, radius).
     * @param point  Vertex position in cm.
     * @param maxT   Upper bound of the time window to test (seconds).
     * @return       Time of first contact in seconds, or > maxT if none.
     */
    static float ballPointCollisionTime(const SimBall& ball,
                                        cv::Point2f    point,
                                        float          maxT);

    /**
     * @brief Applies a cushion-line reflection to @p ball's velocity and spin.
     *
     * @p theta is the angle of the cushion's inward normal, as returned by
     * calculateTheta() for the relevant edge.  The method modifies @p ball
     * in-place, updating velocity components using COEFFICIENT_OF_RESTITUTION
     * and updating spin components using CUSHION_SPIN_RATIO.
     *
     * @param ball   Ball state to modify (velocity and spin updated in-place).
     * @param theta  Angle of the cushion's inward normal in radians.
     */
    static void ballLineCollision(SimBall& ball, float theta);

    /**
     * @brief Advances the ball's velocity and spin by one fixed time-step
     *        under table-cloth friction (sliding, rolling, and spinning).
     *
     * Friction transitions (sliding → rolling → rest) are handled within
     * this call.  Components that fall below ALMOST_ZERO are clamped to zero
     * so that isMovingOrSpinning() can detect true rest in finite time.
     *
     * @param ball  Ball state to modify in-place.
     * @param dt    Time-step duration in seconds (normally FIXED_DT_SEC).
     */
    static void tableBallFriction(SimBall& ball, float dt);

    /**
     * @brief Derives the effective maximum power and spin scale from integer
     *        force and spin stat values relative to CUE_DEFAULT_STAT.
     *
     * Stat values above CUE_DEFAULT_STAT apply the *_INCREASE_PCT rates;
     * values below apply the *_DECREASE_PCT rates.
     *
     * @param forceStat  Integer force/power stat of the cue.
     * @param spinStat   Integer spin stat of the cue.
     * @return           Computed CueStatResult containing maxPower and
     *                   spinScale.
     */
    static CueStatResult applyCueStats(int forceStat, int spinStat);

    /**
     * @brief Returns true if the ball has any residual translational velocity
     *        or any residual spin, false when it has fully come to rest.
     *
     * Uses ALMOST_ZERO as the threshold for all components so that
     * floating-point rounding does not cause spurious non-zero readings.
     *
     * @param ball  Ball state to test.
     * @return      true  — ball is still moving or spinning.
     *              false — ball is at complete rest.
     */
    static bool isMovingOrSpinning(const SimBall& ball);

private:

    // -----------------------------------------------------------------------
    // Cached geometry (built once in the constructor)
    // -----------------------------------------------------------------------

    /**
     * @brief Closed cushion polygon in table-centred cm-space.
     *
     * The last point implicitly connects back to the first.  The polygon
     * traces the inner (ball-contact) face of the cushion rubber clockwise
     * around the table, with a pocket-mouth notch cut into each of the four
     * corners and both mid-rail positions.
     *
     * Segment count: 7 points × 4 corners + 9 points × 2 side pockets = 46
     * vertices total (some shared between adjacent notch profiles).
     */
    std::vector<cv::Point2f> m_cushionPolygon;

    /**
     * @brief Centres of the six pockets in table-centred cm-space.
     *
     * Order: bottom-left corner, bottom-middle side, bottom-right corner,
     *        top-right corner, top-middle side, top-left corner.
     * (Exact coordinates are defined by the constructor.)
     */
    std::vector<cv::Point2f> m_pocketCenters;

    // -----------------------------------------------------------------------
    // Private geometry helpers (used only during construction)
    // -----------------------------------------------------------------------

    /**
     * @brief Builds and returns the cushion polygon from fixed local profiles.
     * @return Fully assembled cushion polygon (closed loop).
     */
    static std::vector<cv::Point2f> buildCushionPolygon();

    /**
     * @brief Builds and returns the list of six pocket-centre coordinates.
     * @return Vector of exactly 6 pocket-centre points in cm-space.
     */
    static std::vector<cv::Point2f> buildPocketCenters();
};