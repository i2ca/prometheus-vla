"""Diagnostic only: known cup pose, free-body grasp from the table, no visual control.

The cup is positioned ONCE during initialization. No pin, weld, teleport, or
direct object force is used after stepping starts. Every invocation keeps a
new directory. This precedes the camera-controlled experiment pipeline.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS, ROOT, load_seed
from kinematics import ArmIK, PALM
from recorder import Recorder

MODEL = "gpt-6-astra (Codex orchestrator)"


def trial(params, directory, record=False):
    sim = G1Sim()
    m, d = sim.m, sim.d
    arm, hand, fps = load_seed(0)
    gi = int(np.flatnonzero(hand[:, 4] > 0.5)[0])
    for names, values in [(ARM_JOINTS, arm[0]), (HAND_JOINTS, hand[0])]:
        for name, value in zip(names, values):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    sim.q_des[:] = d.qpos[sim.qadr]
    sim.set_targets(HAND_JOINTS, hand[0], kp=params.get("kp", 8), kd=1)
    sim.place_cup([*params.get("cup_xy", [0.33, -0.13]), 0.752])
    ik = ArmIK(sim)
    reference = ik.reference_grasp(arm, hand, gi, params.get("offset", [-0.02, -0.02, -0.02]), params.get("gain", 1.5))
    rec = Recorder(m, str(directory / "video.mp4")) if record else None
    samples, ik_errors = [], []
    frames = 0
    phase = "settle"
    initial = None
    table_substeps = 0
    worst_penetration = 0.0

    def contacts():
        table, max_force, min_dist = [], 0.0, 0.0
        for contact in d.contact:
            if m.geom("tampo").id in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == m.geom("tampo").id else contact.geom1
                body = m.body(int(m.geom_bodyid[other])).name
                if body.startswith("right_hand") or body == PALM:
                    table.append(body)
                    min_dist = min(min_dist, float(contact.dist))
        return sorted(set(table)), min_dist

    def advance(nframes):
        nonlocal frames, table_substeps, worst_penetration
        for _ in range(nframes):
            until = (frames + 1) / fps
            while d.time < until - m.opt.timestep / 2:
                tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr]
                if params.get("gravity_compensation", False):
                    tau += d.qfrc_bias[sim.vadr]
                d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
                mujoco.mj_step(m, d)
                sub_contacts, sub_depth = contacts()
                table_substeps += bool(sub_contacts)
                worst_penetration = min(worst_penetration, sub_depth)
            mujoco.mj_forward(m, d)
            table, distance = contacts()
            links = sorted(sim.finger_contacts())
            samples.append({"t": float(d.time), "phase": phase, "cup": sim.cup_pos().tolist(),
                            "palm": sim.body_pos(PALM).tolist(), "finger_links": links,
                            "fingers": sorted({n.split("_")[0] for n in links}),
                            "hand_table": table, "table_penetration_m": distance,
                            "cup_table": sim.cup_table_contact()})
            if rec:
                rec.frame(d)
            frames += 1

    def stage(name):
        nonlocal phase
        phase = name
        if rec:
            rec.event(name)

    def move(position, rotation, seconds):
        start, start_r = ik.fk(sim.q(ARM_JOINTS))
        q = sim.q(ARM_JOINTS)
        rv = cv2.Rodrigues(rotation @ start_r.T)[0].ravel()
        count = max(1, round(seconds * fps))
        for i in range(count):
            u = (i + 1) / count
            u = u * u * (3 - 2 * u)
            target_r = cv2.Rodrigues(rv * u)[0] @ start_r
            q, error = ik.solve(start + u * (np.asarray(position) - start), target_r, q, arm[gi], iterations=35)
            ik_errors.append({"phase": phase, **error})
            sim.set_targets(ARM_JOINTS, q)
            advance(1)

    try:
        advance(20)
        initial = sim.cup_pos().copy()
        rotation = cv2.Rodrigues(np.array([0.0, params.get("palm_pitch_rad", 0.0), 0.0]))[0] @ reference["palm_rotation"]
        target = initial - rotation @ reference["palm_to_cup_translation"]
        if "center_height" in params:
            target = initial + np.array([*params.get("center_xy_bias", [0, 0]), params["center_height"]]) - rotation @ reference["palm_to_center_translation"]
        if params.get("home_lift", 0.0):
            stage("clear_table")
            current, current_r = ik.fk(sim.q(ARM_JOINTS))
            move(current + [0, 0, params["home_lift"]], current_r, 1.0)
        stage("approach_above")
        move(target + [0, 0, params.get("approach_height", 0.09)], rotation, 1.5)
        fraction = params.get("preshape_fraction", 0.0)
        if fraction:
            stage("preshape_above_table")
            sim.set_targets(HAND_JOINTS, reference["open"] + fraction * (reference["close"] - reference["open"]))
            advance(30)
        stage("approach_cup")
        move(target, rotation, 1.5)
        advance(10)
        stage("close_free_cup")
        count = params.get("close_frames", 35)
        for i in range(count):
            u = min(1, (i + 1) / max(1, params.get("ramp_frames", count)))
            u = fraction + (1 - fraction) * u
            close = reference["open"] + u * (reference["close"] - reference["open"])
            sim.set_targets(HAND_JOINTS, close)
            advance(1)
        advance(15)
        stage("lift")
        move(target + [0, 0, params.get("lift_height", 0.16)], rotation, 2.0)
        stage("hold")
        advance(60)
        held = [s for s in samples if s["phase"] == "hold"]
        retained = all(s["cup"][2] > initial[2] + 0.08 and len(s["fingers"]) >= 2 and not s["cup_table"] for s in held)
        report = {"controller_model": MODEL, "mode": "known-pose diagnostic; no camera control",
                  "object_repositions_after_start": 0, "parameters": params, "initial_cup": initial.tolist(),
                  "final_cup": sim.cup_pos().tolist(), "lift_m": float(sim.cup_pos()[2] - initial[2]),
                  "retained_for_2s": bool(retained), "warnings": sim.warnings(),
                  "max_distinct_fingers": max(len(s["fingers"]) for s in samples),
                  "hand_table_frames": sum(bool(s["hand_table"]) for s in samples),
                  "hand_table_substeps": table_substeps,
                  "max_substep_table_penetration_m": -worst_penetration,
                  "max_table_penetration_m": -min(s["table_penetration_m"] for s in samples),
                  "max_ik_position_error_m": max(e["position_error_m"] for e in ik_errors),
                  "max_ik_orientation_error_rad": max(e["orientation_error_rad"] for e in ik_errors)}
        (directory / "physics-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "trajectory.json").write_text(json.dumps(samples) + "\n")
        (directory / "ik.json").write_text(json.dumps(ik_errors) + "\n")
        return report
    finally:
        if rec:
            rec.snapshot("head_camera", str(directory / "camera.png"))
            rec.close(str(directory / "timeline.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    params = json.loads(args.parameters.read_text()) if args.parameters else {"reason": "First unpinned table grasp; correct cup base height; test transfer of historical grasp."}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory = Path(ROOT) / "results" / ("free-cup-" + stamp)
    directory.mkdir()
    (directory / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (directory / "provenance.json").write_text(json.dumps({"model": MODEL, "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}, indent=2) + "\n")
    report = trial(params, directory, args.record)
    print(json.dumps({"directory": str(directory), **report}, indent=2))


if __name__ == "__main__":
    main()
