"""CLI that alternates GPT-6 Astra decisions and persisted MuJoCo steps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from openai import OpenAI

from .policy import AstraPolicy


def _load_harness(root: Path):
    scripts = root / "runtime" / "g1-cup-grasp" / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("direct_env", scripts / "direct_env.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Direct MuJoCo harness is not installed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = importlib.import_module("direct_cases")
    return module.DirectEpisode, cases


def _record_policy_call(episode: Path, index: int, decision) -> None:
    path = episode / "policy-calls"
    path.mkdir(exist_ok=True)
    (path / f"call-{index:03d}.json").write_text(json.dumps({
        "model": "gpt-6-astra",
        "response_id": decision.response_id,
        "tool": decision.tool,
        "arguments": decision.arguments,
        "usage": decision.usage,
    }, indent=2) + "\n")


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    DirectEpisode, cases = _load_harness(root)
    x, y = (float(v) for v in args.cup.split(","))
    case = cases.case((x, y), cases.home_arm_pose())
    episode = Path(args.episode).resolve()
    env = DirectEpisode(episode)
    observation = env.start(case, max_calls=args.max_calls,
                            max_sim_seconds=args.max_sim_seconds, video=not args.no_video)
    meta_path = episode / "episode.json"
    meta = json.loads(meta_path.read_text())
    meta.update({"policy_model": args.model, "reasoning_effort": args.reasoning_effort})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    policy = AstraPolicy(OpenAI(), model=args.model, reasoning_effort=args.reasoning_effort)
    while not observation["finished"]:
        decision = policy.decide(observation)
        _record_policy_call(episode, observation["step_id"] + 1, decision)
        if decision.tool == "finish_episode":
            (episode / "policy-finish.json").write_text(json.dumps(decision.arguments, indent=2) + "\n")
            break
        action = {
            "reason": decision.arguments["reason"],
            "steps": decision.arguments["steps"],
            "target": {
                "position": decision.arguments["position"],
                "quaternion_wxyz": decision.arguments["quaternion_wxyz"],
                "gripper": decision.arguments["gripper"],
            },
        }
        observation = DirectEpisode(episode).load().act(action)

    report = DirectEpisode(episode).load().finish()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("pass") else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--cup", required=True, help="x,y; used only by environment setup")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--max-sim-seconds", type=float, default=40.0)
    parser.add_argument("--no-video", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
