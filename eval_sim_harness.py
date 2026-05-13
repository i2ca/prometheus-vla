#!/usr/bin/env python3
# Closed-loop eval orchestrator: launches unitree_sim_isaaclab + inference_sim_pi05d
# in subprocesses, waits for the inference to finish, collects sim_eval.json.
#
# Run on hercules with the g1 conda env active. Sim runs in its own venv
# (~/Prometheus/isaac/.venv) — the shell launcher has to source it.

from __future__ import annotations
import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
SIM_DIR = Path.home() / "Prometheus" / "shared" / "unitree_sim_isaaclab"
SIM_VENV_PY = Path.home() / "Prometheus" / "isaac" / ".venv" / "bin" / "python"
G1_ENV = "g1"  # conda env for inference

REPO = Path(__file__).resolve().parent


def launch_sim(task: str, no_render: bool, log_path: Path) -> subprocess.Popen:
    cmd = [
        str(SIM_VENV_PY), str(SIM_DIR / "sim_main.py"),
        "--task", task,
        "--robot_type", "g129",
        "--enable_dex3_dds",
        "--enable_pressure_dds",
        "--action_source", "dds",
        "--device", "cuda",
    ]
    if no_render:
        cmd.append("--no_render")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(SIM_DIR), preexec_fn=os.setsid)


def launch_inference(checkpoint: Path, task_str: str, max_episodes: int, threshold: float,
                     frames: int, results_out: Path, log_path: Path, fusion_mode: str,
                     ablation: str = "none", occlude_fraction: float = 0.5,
                     ablation_seed: int = 0) -> subprocess.Popen:
    bash = (
        f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate {G1_ENV} && "
        f"cd {REPO} && python inference_sim_pi05d.py "
        f"--checkpoint {shlex.quote(str(checkpoint))} --task {shlex.quote(task_str)} "
        f"--max-episodes {max_episodes} "
        f"--reward-success-threshold {threshold} "
        f"--reward-success-frames {frames} "
        f"--fusion-mode {fusion_mode} "
        f"--ablation {shlex.quote(ablation)} "
        f"--occlude-fraction {occlude_fraction} "
        f"--ablation-seed {ablation_seed} "
        f"--results-out {results_out}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    return subprocess.Popen(["bash", "-c", bash], stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def kill_proc_tree(p: subprocess.Popen):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to best/pretrained_model")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--task-str", default="Pick up the cup", help="text prompt for policy")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--fusion-mode", choices=["none","depth_only","full"], default="none")
    ap.add_argument("--ablation",
                    choices=["none","drop-rgb","drop-depth","drop-proprio","drop-pressure",
                             "occlude-rgb","occlude-depth","swap-depth-noise"],
                    default="none")
    ap.add_argument("--occlude-fraction", type=float, default=0.5)
    ap.add_argument("--ablation-seed", type=int, default=0)
    ap.add_argument("--reward-threshold", type=float, default=0.8)
    ap.add_argument("--reward-success-frames", type=int, default=5)
    ap.add_argument("--no-render", action="store_true", default=True)
    ap.add_argument("--sim-warmup-sec", type=float, default=20.0,
                    help="Seconds to wait for sim to be ready before launching inference")
    ap.add_argument("--out-dir", required=True, help="dir to write sim_eval.json + logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_log = out_dir / "sim.log"
    inf_log = out_dir / "inference.log"
    results_out = out_dir / "sim_eval.json"

    print(f"[harness] launching sim ({args.task})")
    sim_p = launch_sim(args.task, args.no_render, sim_log)
    print(f"[harness] sim pid={sim_p.pid}; warmup {args.sim_warmup_sec}s")
    time.sleep(args.sim_warmup_sec)

    if sim_p.poll() is not None:
        print(f"[harness] FATAL: sim exited early. Log: {sim_log}")
        sys.exit(1)

    print(f"[harness] launching inference (checkpoint={args.checkpoint})")
    inf_p = launch_inference(
        Path(args.checkpoint), args.task_str, args.max_episodes,
        args.reward_threshold, args.reward_success_frames, results_out, inf_log, args.fusion_mode,
        ablation=args.ablation, occlude_fraction=args.occlude_fraction,
        ablation_seed=args.ablation_seed,
    )
    print(f"[harness] inference pid={inf_p.pid}")

    try:
        rc = inf_p.wait()
        print(f"[harness] inference exited rc={rc}")
    except KeyboardInterrupt:
        print("[harness] Ctrl+C; tearing down")
        kill_proc_tree(inf_p)

    print("[harness] killing sim")
    kill_proc_tree(sim_p)
    try:
        sim_p.wait(timeout=10)
    except Exception:
        pass

    if results_out.exists():
        data = json.loads(results_out.read_text())
        print(f"[harness] RESULT: success_rate={data.get('success_rate', 0):.2%}  "
              f"({data.get('successes', 0)}/{data.get('n_episodes', 0)})")
    else:
        print(f"[harness] WARN: {results_out} not produced")
        sys.exit(2)


if __name__ == "__main__":
    main()
