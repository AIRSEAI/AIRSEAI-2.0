"""blind_peg_insertion expert.

Phases::

    home -> above_peg -> orient -> descend -> close -> lift -> measure
        -> level -> above_hole -> probe -> search -> insert -> release
        -> retreat
"""

from __future__ import annotations

import math as _math

import numpy as _np
import torch as _torch

from sim.criteria.blind_peg_insertion import insertion_depth as _depth
from sim.criteria.blind_peg_insertion import verdict as _verdict
from sim.criteria.probes import tilt_deg as _tilt
from sim.criteria.probes import tip_position as _tip
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.blind_peg_insertion import (
    BOARD_THICKNESS,
    DEFAULT_CLEARANCE,
    PEG_LENGTH,
)
from sim.envs.bimanual_tactile import TABLE_TOP_Z, gel_gap
from sim.envs.blind_peg_insertion import (
    BOARD_TOP_Z,
    DEFAULT_LIGHT_SCALE,
    HOLE_JITTER,
    HOLE_XY,
    PEG_SPAWN_XY,
    SPAWN_JITTER,
    BlindPegInsertionSceneCfg,
)
from sim.tactile import apply_ccd, apply_sdf_collision, apply_triangle_mesh_collision
from sim.tactile.quantities import quat_apply as _quat_apply
from sim.tasks.scene_utils import oblique_grasp_quat
from sim.tasks.spec import Knob, RetryReset, Task

# Controller parameters.
# Grip point along the peg, metres from the tip. Keeps the sensor housing
# clear of the board at full insertion and the gel window on the shaft.
GRASP_HEIGHT = 0.065


__all__ = ["BlindPegInsertionTask", "TASK"]


# --- insertion ------------------------------------------------------------
TARGET_DEPTH = 0.027          # m, commanded depth: 2 mm past the success bar
INSERT_STALL_S = 0.75         # s of no progress that counts as a jam
INSERT_STALL_PROGRESS = 0.0005  # m; less than this over INSERT_STALL_S is a stall
INSERT_CAPTURE_DEPTH = 0.010  # m; past one peg diameter the hole guides it home

# --- search lattice (both are knobs) --------------------------------------
# Pitch must be <= clearance * sqrt(2) for some waypoint to put the tip over
# the hole; the radius only has to cover the injected pose error.
SEARCH_SPACING = 0.0015       # m between lattice points
SEARCH_R_MAX = 0.006          # m, radius of the search disc

# --- press at one waypoint -------------------------------------------------
# Each press descends from the hop height and ends on felt contact, not on a
# commanded height. Reaching the floor with nothing felt means the hole.
HOP_CLEAR = 0.003             # m the tip clears the board between presses
SEARCH_LEVEL_TOL = 2.0        # deg of tilt that triggers a re-level during a hop
PRESS_RATE = 0.020            # m/s descent speed
PRESS_FLOOR = 0.003           # m below the board top the descent may reach
PRESS_DWELL = 0.30            # s held after contact before lifting
FOLLOW_MM = 0.002             # m the TCP follows through after contact
FOLLOW_RATE = 0.008           # m/s of that follow-through
WP_TIMEOUT = 0.6              # s after the hop; advance regardless
UNREACH_WPS = 6               # blocked waypoints in a row before abandoning
GRASP_CREEP_MAX = 0.012       # m the peg may walk up the gel before abandoning

# --- contact detection -----------------------------------------------------
# Two tactile signatures, either one decides: a shear SPIKE (impact) or an
# axial UNLOAD (the peg's weight leaving the gel). A proprioceptive fallback
# catches what neither does.
PRESS_SHEAR_RATIO = 1.5       # contact: mean shear >= this x the hop median
PRESS_SHEAR_MIN_RISE = 15000.0  # ...and at least this much, x1e3 units summed
PRESS_MEAN_N = 3              # frames the rise must hold
PRESS_AXIAL_FRAC = 0.4        # unload: vertical support <= this fraction of the hop's
PRESS_AXIAL_N = 4             # frames the unload must hold
PRESS_AXIAL_MIN = 2000.0      # x1e3; below this the hop carried nothing measurable
PRESS_BLOCKED_ERR = 0.0035    # m of TCP lag that means something is holding the peg
PRESS_BLOCKED_N = 4           # frames the lag must hold

# --- grasp watchdog --------------------------------------------------------
LOST_N = 6                    # frames the pads must stay empty to call it lost
LOST_AREA_FRAC = 0.25         # empty: contact area below this fraction of its median

# --- entry and jaws --------------------------------------------------------
ENTRY_DEPTH = 0.0015          # m below the mouth that counts as entered
OPEN_WIDTH = 0.016            # m per jaw, approaching
CLOSE_WIDTH = 0.0030          # m per jaw, gripping the peg

# --- pick approach ----------------------------------------------------------
APPROACH_BACKOFF = 0.060      # m back along the blade from the grasp to the pre-grasp point
HOME_ABOVE_PREGRASP = 0.030   # m above the pre-grasp point where `home` ends
APPROACH_SPEED = 0.10         # m/s, the command point's speed into the grasp
ORIENT_TOL_DEG = 3.0          # wrist within this of its pick orientation
# Measured gel gap, m, before the pick orientation is computed: the jaw axis
# has no reliable sign while the gels overlap.
LEVEL_MIN_GEL_GAP = 0.010


def _search_offsets(spacing: float, r_max: float) -> list[tuple[float, float]]:
    """Lattice points inside the search disc, ordered by distance from centre."""
    n = int(_math.ceil(r_max / spacing))
    pts = [(i * spacing, j * spacing)
           for i in range(-n, n + 1) for j in range(-n, n + 1)
           if _math.hypot(i * spacing, j * spacing) <= r_max + 1e-9]
    return sorted(pts, key=lambda p: _math.hypot(*p))


class BlindPegInsertionTask(Task):
    name = "blind_peg_insertion"
    instruction = "insert the peg into the hole"
    report_name = "blind_peg_insertion_scripted.json"
    knobs = {
        "clearance": Knob(DEFAULT_CLEARANCE,
                          "radial clearance; must be one of the authored boards"),
        "pose_err": Knob(0.009,
                         "upper bound on the injected visual pose error, metres"),
        "light_scale": Knob(DEFAULT_LIGHT_SCALE,
                            "multiplier on the dome and key lights"),
        "hole_side_offset": Knob(0.06,
                                 "hole offset toward the active arm's side, metres; "
                                 "0 is the midline"),
        "pose_err_min": Knob(0.006,
                             "lower bound on the injected error's magnitude, metres"),
        "hole_jitter": Knob(HOLE_JITTER,
                            "per-episode board jitter on x and y, metres"),
        "search_spacing": Knob(None,
                               "tactile search pitch, metres; None derives it from the clearance",
                               type=lambda v: None if v in (None, "", "None") else float(v)),
        "search_r_max": Knob(None,
                             "search disc radius, metres; None is pose_err + one pitch",
                             type=lambda v: None if v in (None, "", "None") else float(v)),
        "max_tilt_deg": Knob(20.0, "jam threshold handed to the criterion"),
        "search_hop_s": Knob(0.30,
                             "seconds lifted clear of the board between search waypoints"),
        "hole_y": Knob(None, "probe: override the hole's y (metres)"),
        "table_raise": Knob(0.0, "probe: raise the whole scene rigidly (metres)"),
        # Shallower than the top-grasp tasks: a steeper blade puts the fingertips
        # on the board before the peg is fully in.
        "grasp_pitch_deg": Knob(30.0, "blade pitch below horizontal at the pick, deg"),
        "grasp_yaw_inward_deg": Knob(30.0, "blade heading at the pick, deg toward the midline"),
    }
    phases = [
        # The episode starts at HOME; the first move from there is long.
        ("home", 3.0), ("above_peg", 2.5), ("orient", 2.5), ("descend", 3.0), ("close", 2.5),
        ("lift", 2.0), ("measure", 0.4), ("level", 2.5), ("above_hole", 3.0),
        # search is long: the lattice grows as the clearance shrinks.
        ("probe", 2.0), ("search", 300.0), ("insert", 6.0), ("release", 0.6),
        ("retreat", 1.5),
    ]
    # Record from t = 0, HOME included. The pick uses the peg's true pose.
    handover_phase = None
    horizon_s = 30.0
    scene_cam = ((0.68, -0.42, 0.26 + 0.10), (0.26, 0.0, TABLE_TOP_Z + 0.06))

    # -- search lattice, derived from the task's own knobs -------------------
    @staticmethod
    def _lattice(knobs: dict) -> tuple[float, float]:
        """(pitch, radius) for this configuration.

        Pitch follows the clearance, radius follows the error bound. Both are
        overridable; the defaults keep a sweep over `clearance` or `pose_err`
        measuring the task rather than a lattice tuned for one point.
        """
        c = float(knobs["clearance"])
        pitch = knobs.get("search_spacing")
        pitch = min(SEARCH_SPACING, c * 1.4) if pitch is None else float(pitch)
        r = knobs.get("search_r_max")
        r = float(knobs["pose_err"]) + pitch if r is None else float(r)
        return pitch, r

    # -- geometry, honouring the probe knobs ---------------------------------
    def _geom(self, knobs: dict) -> dict:
        raise_ = float(knobs.get("table_raise") or 0.0)
        hole = HOLE_XY if knobs.get("hole_y") is None else (HOLE_XY[0], float(knobs["hole_y"]))
        table = TABLE_TOP_Z + raise_
        board = BOARD_TOP_Z + raise_
        return {"hole": hole, "raise": raise_, "table_top": table, "board_top": board,
                "grasp_tcp": table + GRASP_HEIGHT,
                "touch_tcp": board + GRASP_HEIGHT,
                "hover_tcp": board + GRASP_HEIGHT + 0.030}

    def make_scene_cfg(self, knobs: dict):
        g = self._geom(knobs)
        cfg = BlindPegInsertionSceneCfg(num_envs=1, env_spacing=3.0)
        bp = cfg.board.init_state.pos
        cfg.board.init_state.pos = (g["hole"][0], g["hole"][1], bp[2] + g["raise"])
        pp = cfg.peg.init_state.pos
        cfg.peg.init_state.pos = (pp[0], pp[1], pp[2] + g["raise"])
        if g["raise"]:
            tp = cfg.table.init_state.pos
            cfg.table.init_state.pos = (tp[0], tp[1], tp[2] + g["raise"])
        self._hole_d = cfg.set_clearance(float(knobs["clearance"]))
        self._light = cfg.set_light_scale(float(knobs["light_scale"]))
        return cfg

    def apply_collision(self, h) -> None:
        apply_ccd("/World/envs/env_0/peg")          # dropped-peg tunnelling guard
        n_sdf = len(apply_sdf_collision("/World/envs/env_0/peg"))
        n_tri = len(apply_triangle_mesh_collision("/World/envs/env_0/board"))
        print(f"STEP hole diameter {self._hole_d * 1000:.2f} mm, light_scale "
              f"{self._light:g}; sdf on peg: {n_sdf} mesh(es); board triangles: "
              f"{n_tri}", flush=True)

    def task_id(self, knobs: dict) -> str:
        # The id follows the knob, not the file: at any scale but 1.0 this is
        # `blind_peg_dim` and must not be pooled with lit runs.
        return "blind_peg_insertion" if float(knobs["light_scale"]) == 1.0 else "blind_peg_dim"

    # -- episode --------------------------------------------------------------
    def reset(self, h, rng, ep, knobs) -> dict:
        g = self._geom(knobs)
        # Alternate sides: balanced by construction, reproducible, and the
        # policy still has to read which side from the observation.
        side = ("right", "left")[ep % 2]
        h.select_arm(side)
        # The hole's TRUE pose, redrawn per episode. Everything downstream
        # reads it from `geom`; this is the one place it is decided.
        jit = float(knobs["hole_jitter"])
        sgn = -1.0 if side == "right" else 1.0
        hole = _np.array(g["hole"]) + _np.array([0.0, sgn * float(knobs["hole_side_offset"])])
        hole = hole + (rng.uniform(-jit, jit, 2) if jit else 0.0)
        g = dict(g, hole=(float(hole[0]), float(hole[1])))
        h.place(h.scene["board"], hole,
                g["board_top"] - BOARD_THICKNESS / 2)
        p_xy = _np.array(PEG_SPAWN_XY[side]) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        e_max, e_min = float(knobs["pose_err"]), float(knobs["pose_err_min"])
        if e_min > 0.0:
            # Annulus, uniform in area: r = sqrt(U(rmin^2, rmax^2)), angle U(0, 2pi).
            r = _math.sqrt(rng.uniform(e_min ** 2, e_max ** 2))
            a = rng.uniform(0.0, 2.0 * _math.pi)
            err = _np.array([r * _math.cos(a), r * _math.sin(a)])
        else:
            err = rng.uniform(-e_max, e_max, 2)
        h.reset_robot()
        peg = h.scene["peg"]
        # A standing peg is metastable; re-drop until it stands.
        for attempt in range(5):
            h.place(peg, p_xy, g["table_top"] + PEG_LENGTH / 2 + 0.002)
            # Hold HOME exactly, jaws shut, while the peg settles: the first
            # recorded frame is the robot's HOME.
            h.hold_default(30)
            # The peg must settle near upright: the jaws descend on its base,
            # and past ~3 deg the top leans outside the open jaw.
            if self._tilt(h) < 3.0:
                break
            print(f"STEP ep{ep}: peg fell during settle (tilt {self._tilt(h):.0f} deg), "
                  f"re-dropping (attempt {attempt + 1})", flush=True)
        else:
            raise RetryReset("peg would not stand")
        # Grasp where the peg actually is after settling, not where it was
        # dropped: a settle bounce can walk it a few millimetres.
        p_xy = self._peg_pq(h)[0][:2].copy()
        return {
            "side": side, "peg_xy": p_xy, "pose_err": err,
            "hole_seen": (g["hole"][0] + err[0], g["hole"][1] + err[1]),
            "geom": g, "hole_d": self._hole_d,
            "heights": {"touch": g["touch_tcp"]},
            "offsets": _search_offsets(*self._lattice(knobs)),
            "wp_idx": 0, "t_wp": 0.0, "support": [], "support_baseline": 0.0,
            "wp_err_mm": [], "wp_settled": [], "wp_contacts": [], "wp_tip_mm": [],
            "wp_no_contact": 0, "wp_contact_t": None, "wp_contact_z": None,
            "wp_shear_free": None, "wp_shear_hop": [], "wp_shear_win": [],
            "wp_axial_free": None, "wp_axial_hop": [], "wp_axial_win": [], "contact_kinds": {},
            "wp_err_win": [], "area_support": [],
            "entered": False, "depth_now": 0.0, "upright": None,
            "pick_quat": None, "approach_cmd": None, "pick_goal": None,
            "lat_by_phase": {}, "area_by_phase": {}, "depth_at_release_mm": 0.0,
        }

    # -- probes ---------------------------------------------------------------
    @staticmethod
    def _peg_pq(h):
        peg = h.scene["peg"]
        return (peg.data.root_pos_w[0].detach().cpu().numpy(),
                peg.data.root_quat_w[0].detach().cpu().numpy())

    def _tilt(self, h) -> float:
        return _tilt(self._peg_pq(h)[1])

    def _tip(self, h):
        p, q = self._peg_pq(h)
        return _tip(p, q, PEG_LENGTH)

    def _depth(self, h, ctx) -> float:
        p, q = self._peg_pq(h)
        return _depth(p, q, length=PEG_LENGTH, hole_xy=ctx["geom"]["hole"],
                      mouth_z=ctx["geom"]["board_top"], hole_radius=ctx["hole_d"] / 2)

    def _lat(self, h) -> float:
        """Controller-internal support signal: the active pads' world-
        horizontal resultant. Relative units; only used as a ratio."""
        total = 0.0
        for p in h.ch["pads"]:
            d = h.sensors[p].data
            nf = getattr(d, "tactile_normal_force", None)
            sh = getattr(d, "tactile_shear_force", None)
            qw = getattr(d, "tactile_points_quat_w", None)
            if None in (nf, sh, qw):
                continue
            local = _torch.cat([sh, nf.unsqueeze(-1)], dim=-1)
            fw = _torch.nan_to_num(_quat_apply(qw, local)).sum(dim=1)[0]
            total += float(_torch.linalg.norm(fw[:2]))
        return total

    def _shear(self, h) -> float:
        """Summed tangential force magnitude on the active pads, x1e3. Unlike
        `_lat`, this sees an axial press on the gripped peg."""
        total = 0.0
        for p in h.ch["pads"]:
            sh = getattr(h.sensors[p].data, "tactile_shear_force", None)
            if sh is not None:
                total += float(_torch.linalg.norm(sh[0], dim=-1).sum()) * 1e3
        return total

    def _axial(self, h) -> float:
        """Signed world-vertical component of the active pads' resultant,
        x1e3. Free-hanging it is the peg's weight; when the tip lands on the
        board it collapses toward zero, a sustained contact signature."""
        total = 0.0
        for p in h.ch["pads"]:
            d = h.sensors[p].data
            nf = getattr(d, "tactile_normal_force", None)
            sh = getattr(d, "tactile_shear_force", None)
            qw = getattr(d, "tactile_points_quat_w", None)
            if None in (nf, sh, qw):
                continue
            local = _torch.cat([sh, nf.unsqueeze(-1)], dim=-1)
            fw = _torch.nan_to_num(_quat_apply(qw, local)).sum(dim=1)[0]
            total += float(fw[2]) * 1e3
        return total

    def _tip_above_board(self, h, ctx) -> float:
        """Peg tip height above the board's top face, metres, anywhere in xy.
        `_depth` is only defined inside the hole; this is the number that
        says whether a press reached the board."""
        return float(self._tip(h)[2] - ctx["geom"]["board_top"])

    def signals(self, h, ctx) -> dict:
        p, _ = self._peg_pq(h)
        d = self._depth(h, ctx)
        ctx["depth_now"] = d
        return {"areas": h.pad_areas(), "depth": d, "tilt": self._tilt(h),
                "lat": self._lat(h), "shear": self._shear(h), "axial": self._axial(h),
                "tip_above_mm": self._tip_above_board(h, ctx) * 1e3,
                "peg_pos": p.copy()}

    # -- pick geometry ---------------------------------------------------------
    @staticmethod
    def _pick_yaw_deg(ctx, knobs) -> float:
        """Blade heading at the pick: toward the midline, mirrored per arm."""
        sgn = 1.0 if ctx["side"] == "right" else -1.0
        return sgn * float(knobs["grasp_yaw_inward_deg"])

    def _blade(self, ctx, knobs) -> _np.ndarray:
        """Unit blade direction at the pick, from the pitch and heading knobs."""
        th = _math.radians(float(knobs["grasp_pitch_deg"]))
        psi = _math.radians(self._pick_yaw_deg(ctx, knobs))
        return _np.array([_math.cos(th) * _math.cos(psi), _math.cos(th) * _math.sin(psi),
                          -_math.sin(th)])

    @staticmethod
    def _wrist_err_deg(h, q_want) -> float:
        """Angle between the TCP orientation and `q_want`."""
        if q_want is None:
            return 180.0
        d = abs(float(_np.dot(h.tcp_quat(), _np.asarray(q_want, dtype=float))))
        return float(_np.degrees(2.0 * _np.arccos(min(1.0, d))))

    # -- phase machine --------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        g, xy, p_xy = ctx["geom"], ctx["hole_seen"], ctx["peg_xy"]
        # Before the peg is levelled, the pick orientation (None until the
        # jaws are open, i.e. position IK); after, the upright one.
        grip = CLOSE_WIDTH
        quat = ctx["upright"] if ctx["upright"] is not None else ctx["pick_quat"]
        grasp = _np.array([p_xy[0], p_xy[1], g["grasp_tcp"]])
        pregrasp = grasp - self._blade(ctx, knobs) * APPROACH_BACKOFF
        if name == "home":
            # Jaws still shut, as HOME leaves them; `above_peg` opens them.
            grip = 0.0
            tgt = [pregrasp[0], pregrasp[1], pregrasp[2] + HOME_ABOVE_PREGRASP]
        elif name in ("above_peg", "orient"):
            grip, tgt = OPEN_WIDTH, pregrasp.tolist()
            if (name == "orient" and ctx["pick_quat"] is None
                    and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP):
                ctx["pick_quat"] = quat = oblique_grasp_quat(
                    h.tcp_quat(), h.jaw_axis(), h.approach_axis(),
                    float(knobs["grasp_pitch_deg"]), self._pick_yaw_deg(ctx, knobs))
        elif name == "descend":
            # In along the blade: the command point slides to the grasp point.
            grip = OPEN_WIDTH
            if ctx["approach_cmd"] is None:
                ctx["approach_cmd"] = h.tcp_pos().astype(float)
            d = grasp - ctx["approach_cmd"]
            n = float(_np.linalg.norm(d))
            from sim.harness import DT
            step = APPROACH_SPEED * DT
            ctx["approach_cmd"] = ctx["approach_cmd"] + d * (step / n) if n > step else grasp
            ctx["pick_goal"] = grasp.tolist()
            tgt = ctx["approach_cmd"].tolist()
        elif name == "close":
            grip = OPEN_WIDTH + (CLOSE_WIDTH - OPEN_WIDTH) * min(1.0, t_phase / 1.0)
            tgt = [p_xy[0], p_xy[1], g["grasp_tcp"]]
        elif name == "measure":
            tgt = [p_xy[0], p_xy[1], g["hover_tcp"]]
            hgt = float(h.tcp_pos()[2] - self._tip(h)[2])
            ctx["heights"]["grasp"] = hgt
            ctx["heights"]["touch"] = g["board_top"] + hgt
        elif name == "level":
            tgt = [p_xy[0], p_xy[1], g["hover_tcp"]]
            ctx["upright"] = quat = h.level_wrist(h.scene["peg"])
        elif name == "lift":
            tgt = [p_xy[0], p_xy[1], g["hover_tcp"]]
        elif name == "above_hole":
            tgt = [xy[0], xy[1], g["hover_tcp"]]
        elif name == "probe":
            tgt = [xy[0], xy[1], ctx["heights"]["touch"] + 0.002]
        elif name == "search":
            dx, dy = ctx["offsets"][min(ctx["wp_idx"], len(ctx["offsets"]) - 1)]
            hop_s = float(knobs["search_hop_s"])
            # Heights are commanded through the live tip position, not a
            # grasp height measured once: the peg walks up the gel under
            # repeated presses.
            tip_above = sig["tip_above_mm"] * 1e-3
            tcp = h.tcp_pos()
            tcp_z = float(tcp[2])
            tip = self._tip(h)
            # Lattice offsets are tip positions, so the TCP is offset by the
            # peg's current lean. Re-level during the hop past
            # SEARCH_LEVEL_TOL: an off-axis press levers the peg in the jaws.
            lever = (float(tcp[0] - tip[0]), float(tcp[1] - tip[1]))
            if ctx["t_wp"] < hop_s:
                tip_goal = HOP_CLEAR                       # clear of the board
                if ctx.get("wp_levelled") != ctx["wp_idx"] and sig["tilt"] > SEARCH_LEVEL_TOL:
                    ctx["upright"] = quat = h.level_wrist(h.scene["peg"])
                    ctx["wp_levelled"] = ctx["wp_idx"]
                    ctx["search_relevels"] = ctx.get("search_relevels", 0) + 1
            elif ctx.get("wp_contact_z") is not None:
                # Contact felt. Follow through by FOLLOW_MM: on solid board
                # the tip cannot move, on the chamfer it slides down the
                # lead-in. Stopping at first touch cannot tell the two apart.
                tip_goal = None
                since = ctx["t_wp"] - ctx["wp_contact_t"]
                follow = min(FOLLOW_MM, since * FOLLOW_RATE)
            else:
                # Descend at a fixed rate from the hop clearance; the floor is
                # a safety, not a target -- the press ends on contact.
                tip_goal = max(HOP_CLEAR - (ctx["t_wp"] - hop_s) * PRESS_RATE,
                               -PRESS_FLOOR)
            if tip_goal is None:
                # After contact, steer the TCP over the tip rather than the
                # lattice point, so a tip on the chamfer can slide down the
                # lead-in and the arm goes with it.
                tgt = [float(tip[0]), float(tip[1]), ctx["wp_contact_z"] - follow]
            else:
                tgt = [xy[0] + dx + lever[0], xy[1] + dy + lever[1],
                       tcp_z + (tip_goal - tip_above)]
        elif name in ("insert", "release") and not ctx.get("entered"):
            # Lattice exhausted, peg still on solid board. Do not insert
            # (would drive 27 mm into the face) and do not release (would
            # drop it onto the board). Hold; `retreat` lifts it out.
            tp = h.tcp_pos()
            tgt = [tp[0], tp[1], float(tp[2])]
        elif name == "insert":
            # Absolute goal, fixed when the phase starts. A relative "10 mm
            # below the current TCP" re-issues full authority against a jam
            # every step.
            tp = h.tcp_pos()
            if ctx.get("insert_goal_z") is None:
                ctx["insert_goal_z"] = float(tp[2]) - (TARGET_DEPTH - ctx["depth_now"])
                ctx["insert_t0"] = t_phase
                ctx["insert_hist"] = []
            # Keep the TCP over the peg tip (remote-centre-of-compliance):
            # a TCP descending beside the hole axis levers the peg about the
            # jaws and the tilt grows with every millimetre.
            tip = self._tip(h)
            tgt = [float(tip[0]), float(tip[1]),
                   max(ctx["insert_goal_z"], float(tp[2]) - 0.010)]
        elif name == "release":
            grip = OPEN_WIDTH
            tp = h.tcp_pos()
            tgt = [tp[0], tp[1], float(tp[2])]
        else:                                          # retreat
            # Open only if the peg was placed. A peg that never entered is
            # still in the jaws; opening at hover would drop it onto the board.
            grip = OPEN_WIDTH if ctx.get("entered") else CLOSE_WIDTH
            tp = h.tcp_pos()
            tgt = [tp[0], tp[1], g["hover_tcp"]]
        return tgt, grip, quat

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        areas, depth, lat = sig["areas"], sig["depth"], sig["lat"]
        ctx["lat_by_phase"][name] = max(ctx["lat_by_phase"].get(name, 0.0), lat)
        ctx["area_by_phase"][name] = max(ctx["area_by_phase"].get(name, 0.0),
                                         sum(areas.values()))
        if name == "insert":
            ctx["depth_at_release_mm"] = round(depth * 1e3, 2)
        if name in ("home", "above_peg", "lift", "above_hole", "retreat"):
            return h.tcp_err(target) < 0.015
        if name == "orient":
            return (t_phase > 0.3 and self._wrist_err_deg(h, ctx["pick_quat"]) < ORIENT_TOL_DEG
                    and h.tcp_err(target) < 0.015)
        if name == "measure":
            return t_phase > 0.3
        if name == "level":
            return t_phase > 0.5 and sig["tilt"] < 2.0
        if name == "descend":
            # The true grasp point, not the sliding command point.
            return ctx["pick_goal"] is not None and h.tcp_err(ctx["pick_goal"]) < 0.004
        if name == "close":
            return t_phase > 1.1 and sum(areas.values()) > 5.0
        if name == "probe":
            return h.tcp_err(target) < 0.004
        if name == "search":
            from sim.harness import DT
            ctx["t_wp"] += DT
            hop_s = float(knobs["search_hop_s"])
            shear, axial = sig["shear"], sig["axial"]
            pressing = ctx["t_wp"] >= hop_s
            if not pressing:
                # Free-hanging reference for this waypoint; contact is a rise
                # over it. Median of the hop, since summed shear flickers
                # frame to frame and grip shear drifts as the peg settles.
                ctx["wp_shear_hop"].append(shear)
                ctx["wp_shear_free"] = float(_np.median(ctx["wp_shear_hop"]))
                ctx["wp_axial_hop"].append(axial)
                ctx["wp_axial_free"] = float(_np.median(ctx["wp_axial_hop"]))
            else:
                free = ctx.get("wp_shear_free") or shear
                win = ctx["wp_shear_win"]
                win.append(shear)
                del win[:-PRESS_MEAN_N]
                mean = float(_np.mean(win))
                awin = ctx["wp_axial_win"]
                awin.append(axial)
                del awin[:-PRESS_AXIAL_N]
                afree = ctx.get("wp_axial_free") or 0.0
                amean = float(_np.mean(awin))
                # spike: summed shear rises over the hop median (impact).
                # unload: vertical resultant drops (the peg's weight leaves
                # the gel). A soft landing misses the spike but still unloads.
                spike = len(win) >= PRESS_MEAN_N and \
                    mean >= max(free * PRESS_SHEAR_RATIO, free + PRESS_SHEAR_MIN_RISE)
                unload = len(awin) >= PRESS_AXIAL_N and abs(afree) > PRESS_AXIAL_MIN and \
                    amean / afree <= PRESS_AXIAL_FRAC
                # Proprioceptive fallback, counted separately: the TCP falling
                # behind its command means something is holding the peg up.
                ewin = ctx["wp_err_win"]
                ewin.append(float(h.tcp_err(target)))
                del ewin[:-PRESS_BLOCKED_N]
                blocked = len(ewin) >= PRESS_BLOCKED_N and min(ewin) > PRESS_BLOCKED_ERR
                if ctx.get("wp_contact_t") is None and (spike or unload or blocked):
                    ctx["wp_contact_t"] = ctx["t_wp"]
                    ctx["wp_contact_z"] = float(h.tcp_pos()[2])
                    ctx["wp_contact_rise"] = mean / max(free, 1e-6)
                    kind = "spike" if spike else ("unload" if unload else "blocked")
                    ctx["wp_contact_kind"] = kind
                    ctx["contact_kinds"][kind] = ctx["contact_kinds"].get(kind, 0) + 1
            # Support baseline: median of `lat`, the horizontal resultant,
            # which collapses when the peg rotates out of the jaws.
            if pressing:
                ctx["support"].append(lat)
            base = float(_np.median(ctx["support"])) if ctx["support"] else 0.0
            ctx["support_baseline"] = base
            # The watchdog reads contact area, not `lat`: touchdown dents
            # the horizontal resultant for a few frames, while a peg that has
            # left the jaws keeps the area collapsed.
            area_sum = float(sum(areas.values()))
            if pressing:
                ctx["area_support"].append(area_sum)
            abase = float(_np.median(ctx["area_support"])) if ctx["area_support"] else 0.0
            below = bool(pressing and len(ctx["area_support"]) >= 6 and abase > 1e-6
                         and area_sum < LOST_AREA_FRAC * abase)
            ctx["lost_n"] = ctx.get("lost_n", 0) + 1 if below else 0
            lost = ctx["lost_n"] >= LOST_N
            # Entered: the tip is below the hole mouth.
            felt_now = ctx.get("wp_contact_t") is not None
            # ENTRY_DEPTH is deeper than the tip can sink into the rim, so a
            # missed contact cannot count as entered.
            if pressing and depth > ENTRY_DEPTH:
                ctx["entered"] = True
                ctx["wp_contacts"].append(felt_now)
                ctx["entered_with_contact"] = felt_now
                return True
            if lost:
                ctx["entered"] = True          # judged downstream as a lost grasp
                return True
            # Creep guard: each press walks the peg up the gel and the TCP
            # follows lower, toward a joint limit. Measured as the change in
            # grasp height (TCP minus tip).
            hgt_now = float(h.tcp_pos()[2] - self._tip(h)[2])
            ctx["grasp_creep_mm"] = round((ctx["heights"]["grasp"] - hgt_now) * 1e3, 2)
            if pressing and ctx["heights"]["grasp"] - hgt_now > GRASP_CREEP_MAX:
                ctx["grasp_crept"] = True
                ctx["search_exhausted"] = True
                return True
            felt = ctx.get("wp_contact_t") is not None
            dwelt = felt and (ctx["t_wp"] - ctx["wp_contact_t"]) >= PRESS_DWELL
            timed_out = ctx["t_wp"] >= hop_s + WP_TIMEOUT
            if (dwelt or timed_out) and ctx["wp_idx"] >= len(ctx["offsets"]) - 1:
                # Lattice exhausted without entering. End now rather than
                # holding the last press until the phase timeout.
                ctx["wp_contacts"].append(felt)
                ctx["search_exhausted"] = True
                return True
            if (dwelt or timed_out) and ctx["wp_idx"] < len(ctx["offsets"]) - 1:
                ctx["wp_contacts"].append(felt)
                ctx["wp_err_mm"].append(round(h.tcp_err(target) * 1e3, 2))
                ctx["wp_settled"].append(bool(felt))
                ctx["wp_tip_mm"].append(round(sig["tip_above_mm"], 2))
                if not felt:
                    ctx["wp_no_contact"] += 1
                # Unreachable: "blocked" with the tip still well above the
                # board means joint limits, not contact. Several in a row
                # and the rest of the lattice is out of reach too.
                stuck = ctx.get("wp_contact_kind") == "blocked" and sig["tip_above_mm"] > 3.0
                ctx["unreach_n"] = ctx.get("unreach_n", 0) + 1 if stuck else 0
                if ctx["unreach_n"] >= UNREACH_WPS:
                    ctx["search_unreachable"] = True
                    ctx["search_exhausted"] = True
                    return True
                ctx["wp_idx"] += 1
                ctx["t_wp"] = 0.0
                ctx["wp_contact_t"] = None
                ctx["wp_contact_z"] = None
                ctx["wp_shear_hop"] = []
                ctx["wp_shear_win"] = []
                ctx["wp_axial_hop"] = []
                ctx["wp_axial_win"] = []
                ctx["wp_err_win"] = []
                ctx["wp_contact_kind"] = None
            return False
        if name in ("insert", "release") and not ctx.get("entered"):
            return True
        if name == "insert":
            if depth >= TARGET_DEPTH - 0.002:
                return True
            # Stall: no depth progress for INSERT_STALL_S, so release. Past
            # capture depth the jam is the gripper holding the peg off-axis
            # and gravity finishes the insertion.
            from sim.harness import DT
            hist = ctx.setdefault("insert_hist", [])
            hist.append(depth)
            n = int(round(INSERT_STALL_S / DT))
            # Progress is the window's best depth, not the latest sample.
            # insertion_depth is gated and returns 0 when a gate flickers.
            best = max(hist[-n:]) if len(hist) > n else depth
            if len(hist) > n and (best - hist[-n - 1]) < INSERT_STALL_PROGRESS:
                ctx["insert_stalled"] = True
                ctx["released_early"] = True
                print(f"STEP insert stalled at {best * 1e3:.1f} mm "
                      f"({'captured' if best >= INSERT_CAPTURE_DEPTH else 'not captured'}); "
                      f"releasing", flush=True)
                return True
            return False
        if name == "release":
            timeout = dict(self.phases)[name]
            return t_phase > timeout * 0.8
        return False

    def handover_ok(self, h, ctx, sig) -> bool:
        # Validate the grasp before the recorded / policy episode starts.
        return sum(sig["areas"][p] for p in h.ch["pads"]) > 5.0 and sig["tilt"] < 10.0

    def should_terminate(self, ctx, hist) -> bool:
        # The criterion judges FINAL depth; once the peg has sat past the bar
        # for 3 s the verdict cannot change, so stop.
        from sim.harness import CTRL_FPS
        d = hist["depth"]
        n = 3 * CTRL_FPS
        return len(d) >= n and all(x >= 0.025 for x in d[-n:])

    # -- scoring --------------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        from sim.envs.bimanual_tactile import TACTILE_SENSORS
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in TACTILE_SENSORS},
            object_pos={"peg": _np.asarray(hist["peg_pos"])},
            extra={"insertion_depth_m": _np.asarray(hist["depth"]),
                   "lateral_rel": _np.asarray(hist["lat"]),
                   "peg_tilt_deg": _np.asarray(hist["tilt"])},
        )

    def verdict(self, trace, ctx, knobs):
        from sim.harness import ARM_PADS
        return _verdict(trace, pads=ARM_PADS[ctx["side"]],
                        max_tilt_deg=float(knobs["max_tilt_deg"]))

    def episode_extra(self, ctx, hist) -> dict:
        return {
            "peg_xy": _np.asarray(ctx["peg_xy"]).tolist(),
            "pose_err": _np.asarray(ctx["pose_err"]).tolist(),
            # Episode's true hole, not the nominal one.
            "hole_xy": list(ctx["geom"]["hole"]),
            "entered_hole": bool(ctx["entered"]),
            "max_depth_mm": round(float(max(hist["depth"])) * 1e3, 2) if hist["depth"] else 0.0,
            "max_tilt_deg": round(float(max(hist["tilt"])), 2) if hist["tilt"] else 0.0,
            "depth_at_release_mm": ctx["depth_at_release_mm"],
            # A stalled-then-released success is a correct outcome but not a
            # clean demonstration.
            "insert_stalled": bool(ctx.get("insert_stalled", False)),
            "search_relevels": int(ctx.get("search_relevels", 0)),
            "grasp_crept": bool(ctx.get("grasp_crept", False)),
            "search_unreachable": bool(ctx.get("search_unreachable", False)),
            "grasp_creep_mm": float(ctx.get("grasp_creep_mm", 0.0)),
            "search_contact_kinds": dict(ctx.get("contact_kinds", {})),
            "released_early": bool(ctx.get("released_early", False)),
            "insert_s": (round(sum(1 for _ in ctx.get("insert_hist", [])) / 60.0, 2)),
            "waypoints_tried": ctx["wp_idx"] + 1,
            "lattice_n": len(ctx["offsets"]),
            # Fraction of completed waypoints where the board was felt.
            "search_contact_frac": (round(sum(ctx["wp_contacts"]) / len(ctx["wp_contacts"]), 2)
                                    if ctx["wp_contacts"] else None),
            "search_no_contact_wps": ctx["wp_no_contact"],
            "search_exhausted": bool(ctx.get("search_exhausted", False)),
            "entered_with_contact": ctx.get("entered_with_contact"),
            "search_tip_at_advance_mm": (round(float(_np.mean(ctx["wp_tip_mm"])), 2)
                                         if ctx["wp_tip_mm"] else None),
            "search_settled_frac": (round(sum(ctx["wp_settled"]) / len(ctx["wp_settled"]), 2)
                                    if ctx["wp_settled"] else None),
            "search_track_err_mm": (round(float(_np.mean(ctx["wp_err_mm"])), 2)
                                    if ctx["wp_err_mm"] else None),
            "search_track_err_max_mm": (max(ctx["wp_err_mm"]) if ctx["wp_err_mm"] else None),
            "support_baseline": round(ctx["support_baseline"], 2),
            "grasp_height_mm": round(ctx["heights"].get("grasp", float("nan")) * 1e3, 2),
            "lateral_peak_by_phase": {k: round(v, 2) for k, v in ctx["lat_by_phase"].items()},
            "area_peak_by_phase": {k: round(v, 1) for k, v in ctx["area_by_phase"].items()},
        }

    def report_extra(self, knobs) -> dict:
        g = self._geom(knobs)
        # Nominal hole; each episode's redrawn hole is in its `hole_xy`.
        pitch, r = self._lattice(knobs)
        return {"hole_xy_nominal_m": list(g["hole"]),
                "search_spacing_m": pitch, "search_r_max_m": r,
                "search_lattice_n": len(_search_offsets(pitch, r)),
                "hole_jitter_m": float(knobs["hole_jitter"]),
                "table_raise_m": g["raise"],
                "board_top_z_m": g["board_top"], "clearance_m": float(knobs["clearance"]),
                "hole_diameter_m": self._hole_d, "light_scale": self._light,
                "pose_err_m": float(knobs["pose_err"]),
                "grasp_pitch_deg": float(knobs["grasp_pitch_deg"]),
                "grasp_yaw_inward_deg": float(knobs["grasp_yaw_inward_deg"])}


TASK = BlindPegInsertionTask()
