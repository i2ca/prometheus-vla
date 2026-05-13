#!/usr/bin/env python3
# Sweep wrapper around eval_sim_harness.py: runs (checkpoint, fusion_mode, ablation)
# combinations sequentially, collects sim_eval.json from each, writes a matrix.

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
HARNESS = REPO / "eval_sim_harness.py"

DEFAULT_ABLATIONS = [
    "none",
    "drop-rgb",
    "drop-depth",
    "drop-proprio",
    "drop-pressure",
    "occlude-rgb",
    "occlude-depth",
    "swap-depth-noise",
]


def run_one(ckpt: Path, fusion_mode: str, ablation: str, out_dir: Path,
            max_episodes: int, occlude_fraction: float, sim_warmup_sec: float,
            task: str, task_str: str, reward_threshold: float, reward_frames: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(HARNESS),
        "--checkpoint", str(ckpt),
        "--task", task,
        "--task-str", task_str,
        "--fusion-mode", fusion_mode,
        "--ablation", ablation,
        "--occlude-fraction", str(occlude_fraction),
        "--max-episodes", str(max_episodes),
        "--reward-threshold", str(reward_threshold),
        "--reward-success-frames", str(reward_frames),
        "--sim-warmup-sec", str(sim_warmup_sec),
        "--out-dir", str(out_dir),
    ]
    print(f"[sweep] >>> {ablation} on {ckpt.name} (fusion={fusion_mode})")
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    res_path = out_dir / "sim_eval.json"
    if rc != 0 or not res_path.exists():
        print(f"[sweep] !!! run failed (rc={rc}, json_exists={res_path.exists()}); see {out_dir}")
        return {
            "checkpoint": str(ckpt), "fusion_mode": fusion_mode, "ablation": ablation,
            "success_rate": None, "successes": 0, "n_episodes": 0,
            "rc": rc, "duration_sec": dt, "out_dir": str(out_dir),
        }
    data = json.loads(res_path.read_text())
    data["rc"] = rc
    data["duration_sec"] = dt
    data["out_dir"] = str(out_dir)
    sr = data.get("success_rate")
    print(f"[sweep] <<< {ablation} on {ckpt.name}: success_rate={sr:.2%} ({data.get('successes',0)}/{data.get('n_episodes',0)}) in {dt:.0f}s")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more checkpoint dirs (path to best/ or numbered step)")
    ap.add_argument("--fusion-modes", nargs="+", default=None,
                    help="Parallel list to --checkpoints. If single value, broadcast to all. "
                         "Default: 'depth_only' for paths containing 'depth', else 'none'.")
    ap.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS,
                    help=f"Ablations to sweep. Default: {DEFAULT_ABLATIONS}")
    ap.add_argument("--occlude-fraction", type=float, default=0.5)
    ap.add_argument("--max-episodes", type=int, default=10)
    ap.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
    ap.add_argument("--task-str", default="Pick up the cup")
    ap.add_argument("--reward-threshold", type=float, default=0.8)
    ap.add_argument("--reward-success-frames", type=int, default=5)
    ap.add_argument("--sim-warmup-sec", type=float, default=20.0)
    ap.add_argument("--out-dir", required=True, help="Root dir; per-run subdirs are created")
    ap.add_argument("--tag", default=None, help="Optional tag appended to run subdir names")
    args = ap.parse_args()

    ckpts = [Path(c) for c in args.checkpoints]

    if args.fusion_modes is None:
        fusion_modes = ["depth_only" if "depth" in str(c).lower() else "none" for c in ckpts]
    elif len(args.fusion_modes) == 1:
        fusion_modes = args.fusion_modes * len(ckpts)
    elif len(args.fusion_modes) == len(ckpts):
        fusion_modes = list(args.fusion_modes)
    else:
        print(f"--fusion-modes must be 1 value or match --checkpoints length ({len(ckpts)})")
        sys.exit(1)

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = len(ckpts) * len(args.ablations)
    run_i = 0
    sweep_t0 = time.time()
    for ckpt, fusion in zip(ckpts, fusion_modes):
        # Tag: walk up until we find the run name (parent of "checkpoints/")
        ckpt_tag = None
        for parent in ckpt.parents:
            if parent.name == "checkpoints":
                ckpt_tag = parent.parent.name
                break
        if ckpt_tag is None:
            ckpt_tag = ckpt.name
        for abl in args.ablations:
            run_i += 1
            sub = f"{ckpt_tag}__{abl}"
            if args.tag:
                sub = f"{sub}__{args.tag}"
            out_dir = root / sub
            print(f"[sweep] === run {run_i}/{total_runs} ===")
            res = run_one(
                ckpt, fusion, abl, out_dir,
                max_episodes=args.max_episodes,
                occlude_fraction=args.occlude_fraction,
                sim_warmup_sec=args.sim_warmup_sec,
                task=args.task, task_str=args.task_str,
                reward_threshold=args.reward_threshold,
                reward_frames=args.reward_success_frames,
            )
            res["ckpt_tag"] = ckpt_tag
            results.append(res)
            (root / "sweep_progress.json").write_text(json.dumps({
                "completed": run_i, "total": total_runs,
                "elapsed_sec": time.time() - sweep_t0,
                "results_so_far": results,
            }, indent=2))

    summary_path = root / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] wrote {summary_path}")

    # Markdown matrix: rows = checkpoints, cols = ablations
    md_lines = ["# Eval sweep results", "", f"Total runs: {len(results)}  ",
                f"Total wall time: {time.time() - sweep_t0:.0f}s", ""]
    by_ckpt = {}
    for r in results:
        by_ckpt.setdefault(r["ckpt_tag"], {})[r["ablation"]] = r
    md_lines.append("| checkpoint | " + " | ".join(args.ablations) + " |")
    md_lines.append("|" + "---|" * (len(args.ablations) + 1))
    for ckpt_tag, by_abl in by_ckpt.items():
        cells = []
        for abl in args.ablations:
            r = by_abl.get(abl)
            if r is None or r.get("success_rate") is None:
                cells.append("—")
            else:
                cells.append(f"{r['success_rate']:.0%} ({r.get('successes',0)}/{r.get('n_episodes',0)})")
        md_lines.append(f"| `{ckpt_tag}` | " + " | ".join(cells) + " |")
    md_path = root / "sweep_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[sweep] wrote {md_path}")


if __name__ == "__main__":
    main()
