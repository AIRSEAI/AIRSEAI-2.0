"""VR-only position-priority differential IK, with feedback target bounds.

This module has no simulator imports. Errors are metres, radians, and radians
per joint. Input Cartesian errors must already have the existing speed caps.
The returned target always obeys both soft limits and previous-target slew.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear, minimize


@dataclass
class PoseStep:
    target: np.ndarray
    diagnostics: dict


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def solve_pose_step(jacobian, position_error, rotation_error, actual,
                    previous, limits, max_joint_step, *, feedback=True):
    """Solve translation first, then orientation without changing that result.

    In addition to historical command slew, feedback=True caps target error
    relative to measured joints to one original joint step. If a historical
    target is too far ahead for these intervals to overlap, that joint takes
    the closest legal target to measured position. It therefore returns toward
    feedback at the original slew, and reports ``recovering_lag``. It never
    replaces the old command with measured position in one unbounded jump.

    A failed orientation stage returns the verified position-stage result.
    A failed position stage returns the closest allowed feedback target. Both
    cases are explicit in diagnostics; bad/nonfinite inputs raise ValueError.
    """
    actual = np.asarray(actual, dtype=float)
    if actual.ndim != 1 or not len(actual):
        raise ValueError("actual must be a nonempty joint vector")
    n = len(actual)
    q = _array(actual, (n,), "actual")
    prev = _array(previous, (n,), "previous")
    lim = _array(limits, (n, 2), "limits")
    J = _array(jacobian, (6, n), "jacobian")
    pe = _array(position_error, (3,), "position_error")
    re = _array(rotation_error, (3,), "rotation_error")
    step = float(max_joint_step)
    if not np.isfinite(step) or step <= 0 or np.any(lim[:, 0] > lim[:, 1]):
        raise ValueError("invalid step or joint limits")
    lo_target = np.maximum(lim[:, 0], prev - step)
    hi_target = np.minimum(lim[:, 1], prev + step)
    if np.any(lo_target > hi_target + 1e-12):
        raise ValueError("previous target outside limits beyond one allowed slew")
    # Resolve floating-point-only inversions without widening either bound.
    hi_target = np.maximum(hi_target, lo_target)
    lag_recovery = []
    if feedback:
        safe_lo = np.maximum(lo_target, q - step)
        safe_hi = np.minimum(hi_target, q + step)
        incompatible = safe_lo > safe_hi
        lag_recovery = np.flatnonzero(incompatible).tolist()
        closest = np.clip(q, lo_target, hi_target)
        safe_lo[incompatible] = closest[incompatible]
        safe_hi[incompatible] = closest[incompatible]
        lo_target, hi_target = safe_lo, safe_hi
    lo, hi = lo_target - q, hi_target - q
    fixed = hi - lo <= 1e-12
    free = ~fixed
    dq = np.clip(np.zeros(n), lo, hi)
    dq[fixed] = lo[fixed]
    diagnostics = {
        "solver": "position_priority_bvls_slsqp",
        "feedback": bool(feedback),
        "mode": "recovering_lag" if lag_recovery else "tracking",
        "lag_recovery_joints": lag_recovery,
        "fixed_joints": np.flatnonzero(fixed).tolist(),
        "prior_lag_deg": np.degrees(prev - q).tolist(),
        "max_joint_step_deg": float(np.degrees(step)),
        "position_stage_success": False,
        "orientation_stage_success": False,
        "fallback": None,
        "lower": lo.tolist(), "upper": hi.tolist(),
    }
    Jp, Jr = J[:3], J[3:]
    primary = dq.copy()
    if np.any(free):
        Jpf = Jp[:, free]
        primary_rhs = pe - Jp[:, fixed] @ dq[fixed]
        try:
            fit = lsq_linear(Jpf, primary_rhs,
                             bounds=(lo[free], hi[free]), method="bvls",
                             tol=1e-12, max_iter=100)
            valid = (fit.success and np.isfinite(fit.x).all()
                     and np.all(fit.x >= lo[free] - 1e-10)
                     and np.all(fit.x <= hi[free] + 1e-10))
            diagnostics["position_stage_message"] = str(fit.message)
            diagnostics["position_stage_iterations"] = int(fit.nit)
            if valid:
                primary[free] = np.clip(fit.x, lo[free], hi[free])
                diagnostics["position_stage_success"] = True
            else:
                diagnostics["fallback"] = "bounded_feedback"
        except (ValueError, np.linalg.LinAlgError) as error:
            diagnostics["position_stage_message"] = repr(error)
            diagnostics["fallback"] = "bounded_feedback"
        dq = primary.copy()
        if diagnostics["position_stage_success"]:
            # Independent row constraints avoid SLSQP singular-equality errors
            # at singular postures, sparse Jacobians, or fixed boundary joints.
            _, singular_values, Vt = np.linalg.svd(Jpf, full_matrices=False)
            rank = int(np.sum(singular_values > max(1e-10,
                (singular_values[0] if len(singular_values) else 0.) * 1e-9)))
            E = Vt[:rank]
            equality_target = E @ primary[free]
            # Original 0.01 damping is used only after position is fixed.
            # The tie-breaker penalizes command changes, not a nominal posture.
            reg = .01
            A = np.vstack((Jr[:, free], reg * np.eye(np.count_nonzero(free))))
            b = np.r_[re - Jr[:, fixed] @ dq[fixed], reg * (prev - q)[free]]
            H, g = A.T @ A, A.T @ b
            constraints = [] if rank == 0 else [{
                "type": "eq", "fun": lambda x: E @ x - equality_target,
                "jac": lambda x: E,
            }]
            diagnostics["position_constraint_rank"] = rank
            try:
                secondary = minimize(lambda x: .5 * float(x @ H @ x) - float(g @ x),
                    primary[free], jac=lambda x: H @ x - g,
                    method="SLSQP", bounds=list(zip(lo[free], hi[free])),
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 100, "disp": False})
                valid = (secondary.success and np.isfinite(secondary.x).all()
                         and np.all(secondary.x >= lo[free] - 1e-9)
                         and np.all(secondary.x <= hi[free] + 1e-9)
                         and np.linalg.norm(Jpf @ (secondary.x - primary[free])) <= 1e-8)
                diagnostics["orientation_stage_message"] = str(secondary.message)
                diagnostics["orientation_stage_iterations"] = int(secondary.nit)
                if valid:
                    dq[free] = np.clip(secondary.x, lo[free], hi[free])
                    diagnostics["orientation_stage_success"] = True
                else:
                    diagnostics["fallback"] = "position_stage"
            except (ValueError, np.linalg.LinAlgError) as error:
                diagnostics["orientation_stage_message"] = repr(error)
                diagnostics["fallback"] = "position_stage"
    else:
        diagnostics["position_stage_success"] = True
        diagnostics["orientation_stage_success"] = True
        diagnostics["position_constraint_rank"] = 0
    target = q + dq
    if (not np.isfinite(target).all()
            or np.any(target < lim[:, 0] - 1e-9)
            or np.any(target > lim[:, 1] + 1e-9)
            or np.any(np.abs(target - prev) > step + 1e-9)):
        raise ValueError("solver candidate failed final safety verification")
    diagnostics.update(
        dq_rad=dq.tolist(), Jdq=(J @ dq).tolist(),
        position_request=pe.tolist(), rotation_request=re.tolist(),
        position_residual_m=float(np.linalg.norm(Jp @ dq - pe)),
        orientation_residual_rad=float(np.linalg.norm(Jr @ dq - re)),
        position_priority_change_m=float(np.linalg.norm(Jp @ (dq - primary))),
        target_lag_deg=np.degrees(target - q).tolist(),
        command_step_deg=np.degrees(target - prev).tolist())
    return PoseStep(target, diagnostics)


def make_harness_compute(harness, max_joint_speed_deg_s, *, feedback=True,
                         diagnostic_sink=None):
    """Return Isaac Lab IK compute adapter, installed only during VR apply.

    The normal Harness already clips Cartesian errors and re-verifies the
    returned joint target. Neither Harness nor its expert/scripted callers need
    modification. Imports remain lazy for standalone CPU tests.
    """
    import torch
    from isaaclab.utils.math import compute_pose_error

    def compute(ee_pos, ee_quat, jacobian, joint_pos):
        pe, re = compute_pose_error(ee_pos, ee_quat,
            harness.ik_pose.ee_pos_des, harness.ik_pose.ee_quat_des,
            rot_error_type="axis_angle")
        robot = harness.robot
        ids = harness.ch["arm"].joint_ids
        previous = harness._hold if harness._hold is not None else robot.data.joint_pos_target
        as_numpy = lambda x: x.detach().cpu().double().numpy()
        solved = solve_pose_step(as_numpy(jacobian[0]), as_numpy(pe[0]),
            as_numpy(re[0]), as_numpy(joint_pos[0]), as_numpy(previous[0, ids]),
            as_numpy(robot.data.soft_joint_pos_limits[0, ids]),
            np.radians(float(max_joint_speed_deg_s)) * harness.control_dt,
            feedback=feedback)
        solved.diagnostics["side"] = harness.ch["side"]
        if diagnostic_sink is not None:
            diagnostic_sink(solved.diagnostics)
        return torch.as_tensor(solved.target, dtype=joint_pos.dtype,
                               device=joint_pos.device).unsqueeze(0)
    return compute
