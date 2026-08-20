#pragma once

// ============================================================
//  TrajectoryPhysicsEngine.h
//
//  Bridges the pixel-space pool-shot-analysis pipeline
//  (ATStrip / ShotEx, declared in PipelineEngine.h) to the
//  separate physics simulation module (PoolPhysicsEngine).
//
//  Coordinate systems
//  ------------------
//  •  "px-space"  – ROI-local pixel coordinates used by the
//                   vision pipeline (ATStrip::origin, ::evec,
//                   and every ShotEx::from/::to point).
//  •  "cm-space"  – centimetre coordinates used internally by
//                   PoolPhysicsEngine, with the table centre
//                   as origin and +X along the long axis.
//
//  This class is the only place that knows about both spaces
//  and performs the conversion in both directions.
// ============================================================

#include "PipelineEngine.h"      // ATStrip, ShotEx
#include "PoolPhysicsEngine.h"   // PoolPhysicsEngine, SimBall, PathEventType …
#include <vector>


class TrajectoryPhysicsEngine
{
public:
    // ------------------------------------------------------------------
    //  Constructor
    //  PoolPhysicsEngine's default constructor already builds its internal
    //  fixed cushion polygon and pocket list, so nothing extra is needed
    //  here beyond default-constructing m_physics.
    // ------------------------------------------------------------------
    TrajectoryPhysicsEngine();

    // ------------------------------------------------------------------
    //  computeTrajectories
    //
    //  Converts each *valid* ATStrip into a physically-simulated
    //  trajectory and returns the combined result as a flat list of
    //  straight-line segments, in the same pixel coordinate space as the
    //  input strips (ROI-local).
    //
    //  Parameters
    //  ----------
    //  strips            – Output of the vision pipeline; only entries
    //                      with strip.valid == true are processed.
    //
    //  ptLeftPx          – Left edge   of the calibrated playable surface,
    //  ptTopPx           – Top edge                                  (px).
    //  ptRightPx         – Right edge                                (px).
    //  ptBottomPx        – Bottom edge                               (px).
    //
    //                      This rectangle defines the plain playing
    //                      surface only (no pocket geometry baked in).
    //                      Pocket and cushion shapes are derived entirely
    //                      by this class from PoolPhysicsEngine's fixed
    //                      real-world table proportions, scaled to fit
    //                      this rectangle.
    //
    //  ballRadiusPx      – On-screen ball radius (px), used to derive the
    //                      simulated ball's physical radius in cm.
    //
    //  powerPct          – Shot power as a percentage [1 … 100] of the
    //                      current maximum cue power.
    //
    //  cueForceStat      – Raw cue-force  attribute [0 … 16], passed to
    //  cueSpinStat       – Raw cue-spin   attribute [0 … 16], passed to
    //                      PoolPhysicsEngine::applyCueStats to compute the
    //                      actual maximum power and spin-scale ceiling for
    //                      this frame.
    //
    //  maxReflectionsTgt – Maximum cushion bounces allowed when simulating
    //                      a strip whose is_cue_ball_cut == false
    //                      (target-ball trajectory).
    //  maxReflectionsCbc – Maximum cushion bounces allowed when simulating
    //                      a strip whose is_cue_ball_cut == true
    //                      (cue-ball-cut trajectory).
    //
    //  NOTE – velocity direction
    //  -------------------------
    //  The launch direction is derived purely from each strip's evec field.
    //  No english / spin-direction input exists in this system, so the
    //  launched ball always starts with zero spin (spinX = spinY = spinZ =
    //  0).  Spin only develops during the simulated rollout itself, through
    //  friction between the ball and cloth and through cushion contact
    //  impulses applied by PoolPhysicsEngine.
    // ------------------------------------------------------------------
    std::vector<ShotEx> computeTrajectories(
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
            int maxReflectionsCbc) const;

    // ── Additive read-only getter (used by QeightJNI indirect-shot path) ────

    /**
     * Returns a const reference to the PoolPhysicsEngine instance owned by
     * this object. Allows QeightJNI to pass the same physics instance to
     * IndirectShotSolver, ensuring identical cushion geometry, pocket list,
     * and friction constants between the normal trajectory path and the
     * indirect-shot solver — without constructing a second, separate
     * PoolPhysicsEngine with potentially different calibration.
     * The reference is valid for the lifetime of this TrajectoryPhysicsEngine.
     */
    const PoolPhysicsEngine& getPhysicsEngine() const { return m_physics; }

private:
    // Owns the physics world: cushion polygon, pocket list, friction
    // coefficients, and the ODE integrator.  Constructed once and reused
    // across every call to computeTrajectories.
    PoolPhysicsEngine m_physics;
};