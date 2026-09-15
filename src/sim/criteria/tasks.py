"""Per-task success criteria, one function per benchmark task.

Each returns a `Verdict` and takes its thresholds as keyword arguments;
`tests/test_criteria.py` pins the defaults. Where several detectors fire, the
precedence is chosen per task.

No criterion uses a tactile force threshold (TacSL magnitudes are
uncalibrated); `hold_and_screw`'s torque window is a PhysX joint quantity.
"""

from __future__ import annotations

import numpy as np

from sim.criteria.detectors import (
    detect_grasp_failure,
    detect_mistiming,
    detect_slip,
)
from sim.criteria.trace import EpisodeTrace, FailureMode, Verdict

# Geometric thresholds come from the task's dimensions module.
from sim.dims.handover_cup_transfer import MIN_SEAT
from sim.dims.sleeve_on_shaft import MIN_SEAT as SLEEVE_MIN_SEAT
from sim.dims.stack_cube import CUBE_SIZE

__all__ = ["bimanual_lift", "blind_peg_insertion",
           "handover", "handover_cup_transfer", "hold_and_screw",
           "sleeve_on_shaft", "slip_detect_regrasp", "stack_cube",
           "texture_sort", "unfixed_socket_insertion"]

RIGHT_PADS = ("right_left_finger_tactile", "right_right_finger_tactile")
LEFT_PADS = ("left_left_finger_tactile", "left_right_finger_tactile")


def _placed(trace: EpisodeTrace, obj: str, target_xy, tol_m: float) -> tuple[bool, float]:
    pos = np.asarray(trace.object_pos[obj], dtype=float)[-1]
    err = float(np.linalg.norm(pos[:2] - np.asarray(target_xy, dtype=float)))
    return err < tol_m, err


# -- Tier 1: single arm ----------------------------------------------------

def sleeve_on_shaft(trace: EpisodeTrace, *, min_seat_m: float = SLEEVE_MIN_SEAT,
                    pads: tuple[str, ...] = RIGHT_PADS) -> Verdict:
    """A split sleeve, carried in the gripper, ends up seated on the shaft.

    `extra["seat_depth_m"]` is the sleeve's lower rim below the shaft top,
    zero unless the sleeve is around the shaft and near upright
    (`sim.criteria.probes.seated_on_shaft`). Judged on the FINAL value: a
    sleeve that went on and was pulled back off has not been seated.

    `extra["bore_margin_m"]` is how much wider the bore is than the shaft
    along the squeeze axis: split gap plus twice the clearance, minus how far
    the grip has closed the shells. Negative means the grip shut the bore
    below the shaft diameter.

    In simulation the shaft's chamfer can wedge a compliant grip back open, so
    any sample with the sleeve on the shaft (seat depth above zero) and a
    negative bore margin is `OVERFORCE`, whatever the final seat.

    Precedence: never held `GRASP_FAILURE` -> forced on through a shut bore
    `OVERFORCE` -> seated, success -> not seated with the bore shut while the
    pads held the sleeve `OVERFORCE` -> otherwise `UNFINISHED`.
    """
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    notes: list[str] = []
    seat = trace.extra.get("seat_depth_m")
    margin = trace.extra.get("bore_margin_m")
    metrics = {**m_grasp, "min_seat_m": min_seat_m}
    if seat is None:
        notes.append("no seat_depth_m recorded; cannot judge a seat")
        seat_arr = np.zeros(len(trace.t))
    else:
        seat_arr = np.asarray(seat, dtype=float)
    seat_final = float(seat_arr[-1])
    metrics["seat_depth_m"] = seat_final
    forced_on = shut_while_held = False
    if margin is not None:
        margin_arr = np.asarray(margin, dtype=float)
        metrics["min_bore_margin_m"] = float(np.min(margin_arr))
        on_shaft = seat_arr > 0.0
        if on_shaft.any():
            metrics["min_bore_margin_on_shaft_m"] = float(np.min(margin_arr[on_shaft]))
            forced_on = metrics["min_bore_margin_on_shaft_m"] < 0.0
        held = np.zeros(len(trace.t), dtype=bool)
        for p in pads:
            held |= trace.in_contact(p, 5.0)
        if held.any():
            metrics["min_bore_margin_held_m"] = float(np.min(margin_arr[held]))
            shut_while_held = metrics["min_bore_margin_held_m"] < 0.0
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics, tuple(notes))
    if forced_on:
        return Verdict(False, FailureMode.OVERFORCE, metrics, tuple(notes))
    if seat_final >= min_seat_m:
        return Verdict(True, metrics=metrics, notes=tuple(notes))
    if shut_while_held:
        return Verdict(False, FailureMode.OVERFORCE, metrics, tuple(notes))
    return Verdict(False, FailureMode.UNFINISHED, metrics, tuple(notes))


def slip_detect_regrasp(trace: EpisodeTrace, *, max_slip_m: float = 0.005,
                        hold_s: float = 5.0,
                        pads: tuple[str, ...] = RIGHT_PADS) -> Verdict:
    """Hold against a growing load; react to slip before losing it.

    There is no over-force check: it would be a threshold in uncalibrated
    units, so "squeeze as hard as possible" is not penalised here.
    """
    slipped, m_slip = detect_slip(trace, "manipuland", pads, max_slip_m)
    dropped, m_grasp = detect_grasp_failure(trace, pads, min_hold_s=hold_s)
    metrics = {**m_slip, **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if slipped:
        return Verdict(False, FailureMode.SLIP, metrics)
    return Verdict(True, metrics=metrics)


def blind_peg_insertion(trace: EpisodeTrace, *, min_depth_m: float = 0.025,
                        max_tilt_deg: float = 20.0,
                        pads: tuple[str, ...] = RIGHT_PADS) -> Verdict:
    """Insert into an occluded hole under injected pose error.

    Depth comes from `extra["insertion_depth_m"]`, zero whenever the peg is
    not in the hole (`sim.criteria.probes.insertion_depth`). Judged on the
    final depth, not the deepest moment.

    The jam test is on tilt, not lateral force: tilt separates a peg still
    upright in the grasp from one that has rotated out of it.
    """
    depth = np.asarray(trace.extra.get("insertion_depth_m", [0.0]), dtype=float)
    tilt = np.asarray(trace.extra.get("peg_tilt_deg", [0.0]), dtype=float)
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    metrics = {"depth_m": float(depth[-1]), "peak_depth_m": float(depth.max()),
               "min_depth_m": min_depth_m,
               "peak_tilt_deg": float(tilt.max()), "max_tilt_deg": max_tilt_deg,
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if float(tilt.max()) > max_tilt_deg:
        return Verdict(False, FailureMode.SLIP, metrics)
    if float(depth[-1]) < min_depth_m:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)


def texture_sort(trace: EpisodeTrace, *, chosen_bin: int, correct_bin: int,
                 pads: tuple[str, ...] = RIGHT_PADS) -> Verdict:
    """Classify by surface, place in the matching bin.

    A wrong bin reports UNFINISHED; the failure taxonomy is closed.
    """
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    metrics = {"chosen_bin": float(chosen_bin), "correct_bin": float(correct_bin),
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if chosen_bin != correct_bin:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)


# -- Tier 2: loosely coupled bimanual --------------------------------------

def bimanual_lift(trace: EpisodeTrace, target_xy, *, place_tol_m: float = 0.02,
                  drop_height_m: float = 0.05) -> Verdict:
    """Two-point carry of a load one arm cannot take.

    There is no internal-force metric or crush limit: both would be
    denominated in uncalibrated units.
    """
    z = np.asarray(trace.object_pos["manipuland"], dtype=float)[:, 2]
    fell = bool(z.min() < z[0] - drop_height_m)
    ok_place, err = _placed(trace, "manipuland", target_xy, place_tol_m)
    metrics = {"place_err_m": err, "place_tol_m": place_tol_m}
    if fell:
        return Verdict(False, FailureMode.SLIP, metrics)
    if not ok_place:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)


def handover(trace: EpisodeTrace, *, max_gap_s: float = 0.5,
             max_tilt_deg: float = 15.0, drop_height_m: float = 0.10) -> Verdict:
    """The left arm hands the bar to the right arm.

    Mistiming outranks tilt: a swinging bar is usually the consequence of a
    late release.

    Tilt is judged up to the giver's last contact, taken from the contact
    signals rather than phase labels; after that the bar hangs from one arm.
    `peak_tilt_after_deg` is still reported in the metrics.
    """
    # Left arm gives (holds first), right arm receives: the receiver must gain
    # load before the giver drops it.
    giver, receiver = "left_left_finger_tactile", "right_left_finger_tactile"
    mistimed, m_time = detect_mistiming(trace, receiver, giver, max_gap_s)
    tilt = np.asarray(trace.extra.get("tilt_deg", [0.0]), dtype=float)
    lose = trace.in_contact(giver) if giver in trace.pad_area else np.zeros(0, bool)
    if lose.any():
        last = len(lose) - 1 - int(np.argmax(lose[::-1]))
        during, after = tilt[:last + 1], tilt[last + 1:]
    else:
        during, after = tilt, tilt[:0]
    z = np.asarray(trace.object_pos["manipuland"], dtype=float)[:, 2]
    fell = bool(z.min() < z[0] - drop_height_m)
    metrics = {**m_time,
               "peak_tilt_deg": float(during.max()) if during.size else 0.0,
               "peak_tilt_after_deg": float(after.max()) if after.size else 0.0,
               "max_tilt_deg": max_tilt_deg}
    tilt = during if during.size else tilt
    if fell:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    if mistimed:
        return Verdict(False, FailureMode.MISTIMING, metrics)
    if float(tilt.max()) > max_tilt_deg:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    return Verdict(True, metrics=metrics)


def handover_cup_transfer(trace: EpisodeTrace, *, min_seat_m: float = MIN_SEAT,
                          drop_z_m: float = 0.15,
                          pads: tuple[str, ...] = RIGHT_PADS) -> Verdict:
    """A bar, taken from one hand by the other, ends up in the second cup.

    `extra["seat_depth_m"]` is the bar's foot below the destination rim,
    inside the bore, near upright (`sim.criteria.probes.seated_in_cup`).
    Judged on the FINAL value.

    COORDINATION   the bar fell more than `drop_z_m` below its starting
                   height. Ranked first.
    GRASP_FAILURE  the receiving hand never held it (`detect_grasp_failure`
                   on the right pads).
    """
    seat = np.asarray(trace.extra.get("seat_depth_m", [0.0]), dtype=float)
    z = np.asarray(trace.object_pos["bar"], dtype=float)[:, 2]
    fell = bool(z.min() < z[0] - drop_z_m)
    dropped, m_grasp = detect_grasp_failure(trace, pads)
    metrics = {"seat_depth_m": float(seat[-1]), "peak_seat_depth_m": float(seat.max()),
               "min_seat_m": min_seat_m, "bar_min_z_m": float(z.min()),
               **m_grasp}
    if fell:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if float(seat[-1]) < min_seat_m:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)


def hold_and_screw(trace: EpisodeTrace, *, torque_target_nm: float = 0.30,
                   torque_tol_nm: float = 0.10, max_bottle_tilt_deg: float = 10.0,
                   hold_pads: tuple[str, ...] = LEFT_PADS) -> Verdict:
    """One arm holds an unfixed bottle, the other screws to torque.

    Cap torque is a PhysX joint quantity in newton-metres. There is no crush
    check on the holding pads; a bottle squeezed so hard it rotates fails on
    tilt.
    """
    torque = np.asarray(trace.extra.get("cap_torque_nm", [0.0]), dtype=float)
    final = float(torque[-1])
    tilt = np.asarray(trace.extra.get("bottle_tilt_deg", [0.0]), dtype=float)
    metrics = {"final_torque_nm": final, "torque_target_nm": torque_target_nm,
               "torque_tol_nm": torque_tol_nm, "peak_torque_nm": float(torque.max()),
               "peak_bottle_tilt_deg": float(tilt.max())}
    if float(tilt.max()) > max_bottle_tilt_deg:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    if abs(final - torque_target_nm) > torque_tol_nm:
        mode = (FailureMode.OVERFORCE if final > torque_target_nm
                else FailureMode.UNFINISHED)
        return Verdict(False, mode, metrics)
    return Verdict(True, metrics=metrics)


# -- Tier 3: tightly coupled ------------------------------------------------

def unfixed_socket_insertion(trace: EpisodeTrace, *, min_depth_m: float = 0.012,
                             max_socket_shift_m: float = 0.003,
                             hold_pads: tuple[str, ...] = LEFT_PADS) -> Verdict:
    """One arm inserts into a socket the left arm holds in mid-air.

    Socket shift outranks depth: an insertion completed by shoving the socket
    away does not count.
    """
    depth = np.asarray(trace.extra.get("insertion_depth_m", [0.0]), dtype=float)
    socket = np.asarray(trace.object_pos["socket"], dtype=float)
    shift = float(np.linalg.norm(socket - socket[0], axis=1).max())
    dropped, m_grasp = detect_grasp_failure(trace, hold_pads)
    metrics = {"depth_m": float(depth.max()), "min_depth_m": min_depth_m,
               "socket_shift_m": shift, "max_socket_shift_m": max_socket_shift_m,
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if shift > max_socket_shift_m:
        return Verdict(False, FailureMode.COORDINATION, metrics)
    if float(depth.max()) < min_depth_m:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)


# -- Template task ----------------------------------------------------------

def stack_cube(trace: EpisodeTrace, *, cube_size: float = CUBE_SIZE,
               min_area_mm2: float = 5.0, pads: tuple[str, ...] = RIGHT_PADS
               ) -> Verdict:
    """Cube A stacked on cube B.

    A must end within half a cube of B in xy and one cube up, and the pads
    must have held a load-bearing contact during the carry.
    """
    a = np.asarray(trace.object_pos["cube_a"], dtype=float)[-1]
    b = np.asarray(trace.object_pos["cube_b"], dtype=float)[-1]
    xy_err = float(np.linalg.norm(a[:2] - b[:2]))
    z_err = float(abs(a[2] - (b[2] + cube_size)))
    stacked = xy_err < cube_size / 2 and z_err < cube_size / 2
    dropped, m_grasp = detect_grasp_failure(trace, pads, area_mm2=min_area_mm2)
    metrics = {"xy_err_m": xy_err, "z_err_m": z_err, "cube_size_m": cube_size,
               **m_grasp}
    if dropped:
        return Verdict(False, FailureMode.GRASP_FAILURE, metrics)
    if not stacked:
        return Verdict(False, FailureMode.UNFINISHED, metrics)
    return Verdict(True, metrics=metrics)
