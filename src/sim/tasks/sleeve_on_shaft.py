"""sleeve_on_shaft expert.

Phases::

    unfold -> home -> above_pick -> orient -> descend -> probe -> close -> settle
        -> lift -> above_shaft -> align -> insert -> open -> retreat
"""

from __future__ import annotations

import numpy as _np

from sim.criteria.probes import axis_from_quat, seated_on_shaft, tilt_deg, tip_position
from sim.criteria.tasks import sleeve_on_shaft as _verdict
from sim.criteria.trace import EpisodeTrace as _Trace
from sim.dims.sleeve_on_shaft import (
    JOINT_TRAVEL,
    SLEEVE_HEIGHT,
    SLEEVE_INNER_R,
    SPLIT_GAP,
)
from sim.envs.bimanual_tactile import ARM_LIFTED_JOINT_POS, RIGHT_ARM_SENSORS, TABLE_TOP_Z, gel_gap
from sim.envs.sleeve_on_shaft import (
    MATERIALS,
    PARKED_POS,
    PICK_XY,
    SHAFT_TOP_Z,
    SHAFT_XY,
    SLEEVE_SPAWN_Z,
    SPAWN_JITTER,
    SleeveOnShaftSceneCfg,
    material_prims,
)
from sim.tactile import apply_sdf_collision, apply_triangle_mesh_collision
from sim.tasks.scene_utils import join_split_shells, retarget_pads, top_down_grasp_quat
from sim.tasks.spec import Knob, Task

__all__ = ["TASK", "SleeveOnShaftTask"]

DT = 1.0 / 60.0
PADS = RIGHT_ARM_SENSORS
MATERIAL_BY_NAME = {m.name: m for m in MATERIALS}

# Controller parameters.
# Jaw command while approaching and releasing, metres per jaw: a 46 mm gel gap
# around the 40.5 mm split sleeve.
OPEN_WIDTH = 0.026
# Runaway floor for the closing ramp, metres per jaw: reaching it means the
# sleeve is not between the jaws.
GRIP_MIN_WIDTH = 0.018

# --- heights ---------------------------------------------------------------
# Gel centre 12 mm below the sleeve's top: the finger housing, 34 mm below the
# gel, then stays clear of the table at the pick and of the shaft base once
# the sleeve is seated.
GRASP_BELOW_TOP = 0.012
GRASP_TCP_Z = TABLE_TOP_Z + SLEEVE_HEIGHT - GRASP_BELOW_TOP
# Transit height: the carried sleeve's rim clears the shaft top by ~19 mm,
# low enough that the levelled wrist stays inside its joint range.
HOVER_TCP_Z = TABLE_TOP_Z + 0.105
# Where the wrist is levelled, just above the grasp.
ORIENT_TCP_Z = GRASP_TCP_Z + 0.030
# The sleeve's rim this far above the shaft top before the insertion starts,
# and this far below it as the insertion's goal.
INSERT_START_GAP = 0.008
SEAT_TARGET = 0.020

# --- grip ------------------------------------------------------------------
# The grip is set in jaw travel past first contact, read on the jaw encoder:
# the drive is a spring, so squeeze past contact is what sets the grip force,
# and gel penetration saturates long before the force does.
CLOSE_RATE_FAST = 0.006       # m/s per jaw, until both pads touch
CLOSE_RATE_SLOW = 0.002       # m/s per jaw, from the probe to the grip
AREA_CONTACT = 5.0            # mm2 per pad; below this a pad is not bearing load
# How long the probe's squeeze is held before the jaw position is read, s.
PROBE_HOLD_S = 0.4
# Squeeze past contact per called material, metres per jaw: the soft sleeve's
# bore stays open at its setting, the stiff sleeve needs the force of its own.
GRIP_SQUEEZE = {"soft": 0.0020, "stiff": 0.0055}

# --- motion ----------------------------------------------------------------
# Commanded TCP speed limits, m/s. The command point advances from the
# previous command, so the position term keeps pulling while pose IK turns
# the wrist.
FREE_SPEED = 0.25
CARRY_SPEED = 0.10
INSERT_SPEED = 0.02
CARRY_PHASES = ("settle", "lift", "above_shaft", "align")

# --- phase completion ------------------------------------------------------
ORIENT_TILT_DEG = 6.0         # blade tilt that counts as levelled above the grasp
# Measured gel gap, m, before the level orientation is computed. The jaw axis
# is the difference of the two gel centroids, which has no reliable sign while
# the gels overlap.
LEVEL_MIN_GEL_GAP = 0.010
ALIGN_XY_TOL = 0.0015         # m, rim centre to shaft axis before inserting
SETTLE_STEPS = 30             # steps held before t = 0


class SleeveOnShaftTask(Task):
    name = "sleeve_on_shaft"
    # Names neither material: the instruction must not give away what the
    # policy is supposed to feel for.
    instruction = "slide the sleeve onto the shaft"
    report_name = "sleeve_on_shaft_scripted.json"
    knobs = {
        "clearance_mm": Knob(1.0, "radial bore-to-shaft clearance, mm: 0.5, 1.0 or 1.5"),
        "materials": Knob("soft,stiff", "hidden materials to draw from, comma-separated",
                          type=str),
        "probe_squeeze_mm": Knob(1.5, "jaw command past first contact during the probe, mm per jaw"),
        "probe_thresh_mm": Knob(0.85,
                                "jaw travel past contact above which the sleeve is soft, mm"),
        "force_grip": Knob("", "grip as 'soft' or 'stiff' regardless of the probe", type=str),
        "stiff_half_mass": Knob(MATERIAL_BY_NAME["stiff"].half_mass,
                                "mass of each stiff shell, kg; the stiff sleeve weighs twice this"),
        "pen_quantile": Knob(0.98,
                             "taxel penetration quantile used as a pad's reading"),
        "sdf_resolution": Knob(128, "PhysX SDF grid per shell; four are cooked at sim.reset()",
                               type=int),
    }
    phases = [
        # Joint space, not IK. Episode starts at HOME.
        ("unfold", 2.0),
        # home 3 s: from HOME the hands start low and forward.
        ("home", 3.0), ("above_pick", 2.5), ("orient", 3.0), ("descend", 4.0),
        ("probe", 4.0), ("close", 4.0), ("settle", 1.0), ("lift", 3.0),
        ("above_shaft", 4.0), ("align", 3.0), ("insert", 5.0), ("open", 1.0),
        ("retreat", 1.5),
    ]
    # The policy takes control at t = 0: no prologue, the whole episode is
    # recorded, HOME included.
    handover_phase = None
    joint_phases = ("unfold",)
    horizon_s = 38.0
    scene_cam = ((0.70, -0.42, 0.34), (0.26, -0.085, 0.24))
    _sdf_resolution = 128
    _clearance = 0.001

    def _timeout(self, name: str) -> float:
        return dict(self.phases)[name]

    # ---- scene -------------------------------------------------------------
    def make_scene_cfg(self, knobs: dict):
        self._sdf_resolution = int(knobs["sdf_resolution"])
        cfg = SleeveOnShaftSceneCfg(num_envs=1, env_spacing=3.0)
        self._clearance = cfg.set_clearance(round(float(knobs["clearance_mm"]) * 1e-3, 6))
        self._stiff_half_mass = cfg.set_material_mass("stiff", float(knobs["stiff_half_mass"]))
        return cfg

    def apply_collision(self, h) -> None:
        for m in MATERIALS:
            for prim in material_prims(m.name):
                apply_sdf_collision(f"/World/envs/env_0/{prim}",
                                    sdf_resolution=self._sdf_resolution)
        apply_triangle_mesh_collision("/World/envs/env_0/shaft")

    # ---- episode -----------------------------------------------------------
    def _place_material(self, h, material, xy) -> None:
        """Stand the drawn material's sleeve on the table, park the other, and
        author its shell joint afresh."""
        for m in MATERIALS:
            a, b = material_prims(m.name)
            if m.name == material.name:
                h.place(h.scene[a], (xy[0], xy[1] + SPLIT_GAP / 2), SLEEVE_SPAWN_Z)
                h.place(h.scene[b], (xy[0], xy[1] - SPLIT_GAP / 2), SLEEVE_SPAWN_Z)
            else:
                h.place(h.scene[a], PARKED_POS[:2], PARKED_POS[2])
                h.place(h.scene[b], PARKED_POS[:2], PARKED_POS[2])
        a, b = material_prims(material.name)
        join_split_shells(f"/World/envs/env_0/{a}", f"/World/envs/env_0/{b}",
                        stiffness=material.stiffness, damping=material.damping,
                        break_force=float("inf"), travel=JOINT_TRAVEL,
                        gap=SPLIT_GAP, name="squash_joint")

    def _retarget(self, h, material) -> None:
        """Each right pad senses the shell it faces (`_a` is +y); the idle
        arm's pads must name some SDF prim, so they take the near shell."""
        a, b = material_prims(material.name)
        names = list(h.robot.data.body_names)
        ys = [float(h.robot.data.body_state_w[0, names.index(f"openarm_{p}_finger"), 1])
              for p in ("right_left", "right_right")]
        plus_first = ys[0] > ys[1]
        retarget_pads({PADS[0]: h.sensors[PADS[0]]}, a if plus_first else b)
        retarget_pads({PADS[1]: h.sensors[PADS[1]]}, b if plus_first else a)
        retarget_pads({p: s for p, s in h.sensors.items() if p not in PADS}, a)

    def reset(self, h, rng, ep, knobs) -> dict:
        h.select_arm("right")
        pool = [MATERIAL_BY_NAME[n] for n in str(knobs["materials"]).split(",") if n]
        # The hidden draw. `material` is never read by the controller after this.
        material = pool[int(rng.integers(len(pool)))]
        pick = _np.asarray(PICK_XY) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        shaft = _np.asarray(SHAFT_XY) + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
        h.place(h.scene["shaft"], shaft, TABLE_TOP_Z)
        self._place_material(h, material, pick)
        self._retarget(h, material)
        h.reset_robot()
        # Hold HOME, jaws still shut, while the sleeve settles.
        for _ in range(SETTLE_STEPS):
            h.step_ik_toward(h.tcp_pos(), 0.0)
            h.step()
        return {
            "side": "right",
            "true_material": material.name,
            "prims": material_prims(material.name),
            "pick": pick, "shaft": shaft,
            "pen_q": float(knobs["pen_quantile"]),
            "level_quat": None,
            "grip": 0.0,
            "goal": None,
            "cmd_pos": None,
            "release_xy": None,
            # Measured grasp: rim centre minus TCP in xy, and TCP height above
            # the rim, latched when the lift ends.
            "grasp_off_xy": None,
            "rim_below_tcp": None,
            "grip_log": [],
            # What the probe learns, and the only channel by which it reaches
            # the rest of the episode.
            "probe": {"g_contact": None, "t_contact": None, "travel": None, "called": None},
            "probe_failed": False,
            "margin_at_insert": None,
        }

    # ---- signals -----------------------------------------------------------
    def signals(self, h, ctx) -> dict:
        pen = h.pad_penetration(quantile=ctx["pen_q"])
        a, b = ctx["prims"]
        pa = h.scene[a].data.root_pos_w[0].detach().cpu().numpy()
        pb = h.scene[b].data.root_pos_w[0].detach().cpu().numpy()
        qa = h.scene[a].data.root_quat_w[0].detach().cpu().numpy()
        pos = (pa + pb) / 2.0
        # Signed shell separation along the sleeve's own y axis (`_a` is +y):
        # the bore's width along the squeeze is 2 * inner radius plus this.
        sep = float(_np.dot(pa - pb, axis_from_quat(qa, (0.0, 1.0, 0.0))))
        rim = tip_position(pos, qa, SLEEVE_HEIGHT)
        areas = h.pad_areas()
        return {
            "areas": areas,
            "pen": pen,
            "pen_mean": float(sum(pen.values()) / max(1, len(pen))),
            "min_area": min(areas[p] for p in PADS),
            "tilt": h.blade_tilt_deg(),
            "sleeve_pos": pos,
            "rim": rim,
            "sep": sep,
            "bore_margin": sep + 2.0 * self._clearance,
            "seat": seated_on_shaft(pos, qa, height=SLEEVE_HEIGHT, shaft_xy=ctx["shaft"],
                                    shaft_top_z=SHAFT_TOP_Z, bore_r=SLEEVE_INNER_R),
            "sleeve_tilt": tilt_deg(qa),
        }

    # ---- phases ------------------------------------------------------------
    def phase_target(self, h, ctx, name, t_phase, sig, knobs):
        if name == "unfold":
            # Joint space: the working configuration, jaws still shut as the
            # robot's HOME leaves them. `home` opens them.
            return dict(ARM_LIFTED_JOINT_POS), 0.0, None
        pick, shaft = ctx["pick"], ctx["shaft"]
        grip = ctx["grip"]
        probe = ctx["probe"]
        quat = ctx["level_quat"]
        tcp = h.tcp_pos()
        # Where the TCP must be for the sleeve's rim centre to sit over the
        # shaft axis. The grasp offset is the one measured when the lift
        # ended; while the pads still bear load the live rim position refines
        # it, and once they do not, nothing chases a dropped sleeve.
        held = sig["min_area"] > AREA_CONTACT
        off = ctx["grasp_off_xy"] if ctx["grasp_off_xy"] is not None else _np.zeros(2)
        over_shaft = shaft - off
        if held and ctx["grasp_off_xy"] is not None:
            over_shaft = tcp[:2] + (shaft - sig["rim"][:2])
        rim_below_tcp = (ctx["rim_below_tcp"] if ctx["rim_below_tcp"] is not None
                         else SLEEVE_HEIGHT - GRASP_BELOW_TOP)
        if name == "home":
            grip = OPEN_WIDTH
            tgt = [pick[0], pick[1], HOVER_TCP_Z]
        elif name == "above_pick":
            grip = OPEN_WIDTH
            tgt = [pick[0], pick[1], ORIENT_TCP_Z]
        elif name == "orient":
            grip = OPEN_WIDTH
            # Approach on position IK, then level the wrist in place and keep
            # that orientation for the rest of the episode.
            if quat is None and gel_gap(h.grip_measured()) >= LEVEL_MIN_GEL_GAP:
                quat = ctx["level_quat"] = top_down_grasp_quat(
                    h.tcp_quat(), h.jaw_axis(), h.approach_axis())
            tgt = [pick[0], pick[1], ORIENT_TCP_Z]
        elif name == "descend":
            grip = OPEN_WIDTH
            tgt = [pick[0], pick[1], GRASP_TCP_Z]
        elif name == "probe":
            # Close until both pads touch, then hold a fixed squeeze past that
            # contact; how far the jaws actually follow says how much the
            # sleeve gives.
            if probe["g_contact"] is None:
                grip = max(GRIP_MIN_WIDTH, grip - CLOSE_RATE_FAST * DT)
            else:
                grip = probe["g_contact"] - float(knobs["probe_squeeze_mm"]) * 1e-3
            tgt = [pick[0], pick[1], GRASP_TCP_Z]
        elif name == "close":
            want = max(GRIP_MIN_WIDTH, probe["g_contact"] - GRIP_SQUEEZE[probe["called"]])
            grip = max(want, grip - CLOSE_RATE_SLOW * DT) if grip > want else want
            tgt = [pick[0], pick[1], GRASP_TCP_Z]
        elif name == "settle":
            tgt = [pick[0], pick[1], GRASP_TCP_Z]
        elif name == "lift":
            tgt = [pick[0], pick[1], HOVER_TCP_Z]
        elif name == "above_shaft":
            tgt = [over_shaft[0], over_shaft[1], HOVER_TCP_Z]
        elif name == "align":
            tgt = [over_shaft[0], over_shaft[1], SHAFT_TOP_Z + INSERT_START_GAP + rim_below_tcp]
        elif name == "insert":
            if ctx["margin_at_insert"] is None:
                ctx["margin_at_insert"] = sig["bore_margin"]
            tgt = [over_shaft[0], over_shaft[1], SHAFT_TOP_Z - SEAT_TARGET + rim_below_tcp]
        elif name == "open":
            grip = OPEN_WIDTH
            if ctx["release_xy"] is None:
                ctx["release_xy"] = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
            tgt = list(ctx["release_xy"])
        else:                                        # retreat
            grip = OPEN_WIDTH
            rel = ctx["release_xy"] or [float(tcp[0]), float(tcp[1]), float(tcp[2])]
            tgt = [rel[0], rel[1], HOVER_TCP_Z]
        ctx["grip"] = grip
        ctx["goal"] = list(tgt)
        speed = (INSERT_SPEED if name == "insert"
                 else CARRY_SPEED if name in CARRY_PHASES else FREE_SPEED)
        return self._rate_limited(h, ctx, tgt, speed), grip, quat

    def _rate_limited(self, h, ctx, goal, speed: float) -> list:
        """Advance the commanded point toward `goal` by at most `speed` * DT."""
        if ctx["cmd_pos"] is None:
            ctx["cmd_pos"] = h.tcp_pos().astype(float)
        delta = _np.asarray(goal, dtype=float) - ctx["cmd_pos"]
        dist = float(_np.linalg.norm(delta))
        step = speed * DT
        ctx["cmd_pos"] = (ctx["cmd_pos"] + delta * (step / dist) if dist > step
                          else _np.asarray(goal, dtype=float))
        return ctx["cmd_pos"].tolist()

    def phase_done(self, h, ctx, name, t_phase, sig, target, knobs) -> bool:
        if name == "unfold":
            # Every commanded joint within a degree of its target.
            names = h._joint_name_index
            q = h.robot.data.joint_pos[0]
            return all(abs(float(q[names[n]]) - v) < 0.017 for n, v in target.items())
        if name in ("probe", "close", "settle", "lift") and int(round(t_phase / DT)) % 3 == 0:
            ctx["grip_log"].append([
                name, round(t_phase, 3), round(ctx["grip"] * 1e3, 3),
                round(h.grip_measured() * 1e3, 3),
                [round(v * 1e3, 3) for v in sig["pen"].values()],
                [round(sig["areas"][p], 1) for p in PADS],
                round(sig["sep"] * 1e3, 3)])
        # Completion reads the phase's true goal, never the rate-limited
        # point that was commanded.
        goal = ctx["goal"]
        probe = ctx["probe"]
        if name in ("home", "above_pick", "above_shaft", "retreat"):
            return h.tcp_err(goal) < 0.015
        if name == "orient":
            return (ctx["level_quat"] is not None and t_phase > 0.3
                    and sig["tilt"] < ORIENT_TILT_DEG and h.tcp_err(goal) < 0.015)
        if name == "descend":
            return h.tcp_err(goal) < 0.004
        if name == "probe":
            if probe["g_contact"] is None:
                if all(sig["areas"][p] > AREA_CONTACT for p in PADS):
                    probe["g_contact"] = h.grip_measured()
                    probe["t_contact"] = t_phase
                return ctx["grip"] <= GRIP_MIN_WIDTH
            if t_phase - probe["t_contact"] < PROBE_HOLD_S:
                return False
            probe["travel"] = probe["g_contact"] - h.grip_measured()
            called = ("soft" if probe["travel"] > float(knobs["probe_thresh_mm"]) * 1e-3
                      else "stiff")
            forced = str(knobs["force_grip"]).strip()
            probe["called"] = forced if forced else called
            print(f"STEP probe travel {probe['travel'] * 1e3:.3f} mm -> called {called}"
                  + (f", gripping as {forced} (forced)" if forced else ""), flush=True)
            return True
        if name == "close":
            want = max(GRIP_MIN_WIDTH, probe["g_contact"] - GRIP_SQUEEZE[probe["called"]])
            return t_phase > 0.3 and ctx["grip"] <= want + 1e-9
        if name == "settle":
            return t_phase > 0.5
        if name == "lift":
            done = abs(float(h.tcp_pos()[2]) - goal[2]) < 0.010
            if done:
                tcp = h.tcp_pos()
                ctx["grasp_off_xy"] = sig["rim"][:2] - tcp[:2]
                ctx["rim_below_tcp"] = float(tcp[2] - sig["rim"][2])
            return done
        if name == "align":
            off = float(_np.linalg.norm(sig["rim"][:2] - ctx["shaft"]))
            return t_phase > 0.3 and off < ALIGN_XY_TOL and h.tcp_err(goal) < 0.004
        if name == "insert":
            return sig["seat"] >= SEAT_TARGET - 0.002
        if name == "open":
            return t_phase > 0.8 * self._timeout("open")
        return False

    def next_phase(self, ctx, name):
        probe = ctx["probe"]
        if name == "probe" and probe["called"] is None:
            # The probe never completed: grip as the GENTLER material, so a
            # wrong guess slips rather than shutting the bore.
            if probe["g_contact"] is None:
                probe["g_contact"] = ctx["grip"]
            probe["called"] = "soft"
            ctx["probe_failed"] = True
            print("STEP probe did not complete; defaulting to soft", flush=True)
        return None

    # ---- scoring -----------------------------------------------------------
    def build_trace(self, h, ctx, hist) -> _Trace:
        return _Trace(
            t=_np.asarray(hist["t"]),
            pad_area={p: _np.asarray([a[p] for a in hist["areas"]]) for p in PADS},
            object_pos={"sleeve": _np.asarray(hist["sleeve_pos"])},
            extra={"seat_depth_m": _np.asarray(hist["seat"]),
                   "bore_margin_m": _np.asarray(hist["bore_margin"]),
                   "sleeve_tilt_deg": _np.asarray(hist["sleeve_tilt"]),
                   "pen_m": _np.asarray(hist["pen_mean"])},
        )

    def verdict(self, trace, ctx, knobs):
        return _verdict(trace, pads=PADS)

    def episode_extra(self, ctx, hist) -> dict:
        probe = ctx["probe"]
        return {
            "true_material": ctx["true_material"],
            "called_material": probe["called"],
            "material_correct": probe["called"] == ctx["true_material"],
            "probe_travel_mm": (round(probe["travel"] * 1e3, 3)
                                if probe["travel"] is not None else None),
            "probe_failed": ctx["probe_failed"],
            "peak_pen_mm": round(float(max(hist["pen_mean"])) * 1e3, 3),
            "final_seat_mm": round(float(hist["seat"][-1]) * 1e3, 2),
            "min_bore_margin_mm": round(float(min(hist["bore_margin"])) * 1e3, 3),
            "margin_at_insert_mm": (None if ctx["margin_at_insert"] is None
                                    else round(ctx["margin_at_insert"] * 1e3, 3)),
            "final_tilt_deg": round(float(hist["sleeve_tilt"][-1]), 1),
            "pick": [round(float(v), 4) for v in ctx["pick"]],
            "shaft": [round(float(v), 4) for v in ctx["shaft"]],
            "grasp_off_xy_mm": (None if ctx["grasp_off_xy"] is None
                                else [round(float(v) * 1e3, 2) for v in ctx["grasp_off_xy"]]),
            # phase, t, grip cmd mm, grip measured mm, pen per pad mm, area per pad mm2, shell sep mm
            "grip_log": ctx["grip_log"],
        }

    def report_extra(self, knobs) -> dict:
        return {
            "materials": {m.name: {"youngs_modulus_pa": m.youngs_modulus,
                                   "stiffness_n_m": round(m.stiffness, 1),
                                   "half_mass_kg": m.half_mass}
                          for m in MATERIALS},
            "clearance_m": self._clearance,
            "probe_squeeze_mm": float(knobs["probe_squeeze_mm"]),
            "probe_thresh_mm": float(knobs["probe_thresh_mm"]),
            "grip_squeeze_m": GRIP_SQUEEZE,
            "force_grip": str(knobs["force_grip"]),
            "stiff_half_mass_kg": float(knobs["stiff_half_mass"]),
        }


TASK = SleeveOnShaftTask()
