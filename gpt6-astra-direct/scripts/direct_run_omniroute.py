"""Roda um episodio Direct com um modelo do OmniRoute como politica (API formato Anthropic).

Uso:
  set -a; source ~/.config/omniroute/omniroute.env; set +a
  .venv/bin/python scripts/direct_run_omniroute.py results/direct-omniroute-01/x0.40_y-0.20 \
      --cup 0.40,-0.20 --model cx/gpt-5.6-sol-high

Cada chamada fica em <ep>/policy-calls/call-NNN.json e uma linha em <ep>/policy-log.jsonl:
id da resposta do gateway, uso de tokens, texto e tool call do modelo, e o que o arnes fez.
"""
import argparse, base64, datetime, hashlib, json, os, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from direct_env import DirectEpisode
import direct_cases

GATEWAY = os.environ.get("OMNIROUTE_URL", "http://127.0.0.1:20128")

SYSTEM = """You are the visuomotor policy for a Unitree G1 humanoid in MuJoCo, right arm, Dex3 hand.
Use only the RGB images and measured proprioception in this message. You do not know the cup pose.
Each call: choose ONE absolute target pose for the right palm, within 5 cm and 0.35 rad of the
current measured palm pose, then look at the next observation. Frame: x forward, y left, z up (meters).
Keep the hand above the table in transit, approach the mug from the side at mid height, open before
contact, close only when the mug is between thumb and fingers, then lift. Touching the table with the
hand or the robot body aborts the episode. If a command is rejected, read the error and fix it.
In `what_i_see`, describe what each camera shows right now (mug, hand, fingers, table) before deciding.
Put your immediate purpose in `reason` (one or two sentences).
Call finish_episode when the mug is lifted and held, or if continuing is unsafe."""

TOOLS = [
    {"name": "actuate", "description": "Execute one bounded Cartesian command for the right palm.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "what_i_see": {"type": "string", "description": "what each camera shows now"},
         "reason": {"type": "string"},
         "steps": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1 step = 0.2 s"},
         "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
         "gripper": {"type": "string", "enum": ["keep", "open", "closed"]}},
         "required": ["what_i_see", "reason", "steps", "position", "quaternion_wxyz", "gripper"]}},
    {"name": "finish_episode", "description": "Stop: task complete, or unsafe/impossible to continue.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "what_i_see": {"type": "string", "description": "what each camera shows now"},
         "reason": {"type": "string"},
         "outcome": {"type": "string", "enum": ["complete", "blocked", "unsafe"]}},
         "required": ["what_i_see", "reason", "outcome"]}},
]


def content_of(obs):
    public = {k: v for k, v in obs.items() if k not in ("images", "next_call", "episode")}
    blocks = [{"type": "text", "text": "Current measured observation:\n" + json.dumps(public)}]
    for im in obs["images"]:
        blocks.append({"type": "text", "text": f"Camera: {im['camera']}"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": base64.b64encode(Path(im["path"]).read_bytes()).decode()}})
    return blocks


def ask(model, obs, system=SYSTEM, retries=3):
    body = {"model": model, "max_tokens": 2000, "system": system, "tools": TOOLS,
            "tool_choice": {"type": "any"}, "messages": [{"role": "user", "content": content_of(obs)}]}
    req = urllib.request.Request(f"{GATEWAY}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                  "x-api-key": os.environ["OMNIROUTE_API_KEY"]})
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = json.load(urllib.request.urlopen(req, timeout=300))
            return resp, round(time.time() - t0, 2)
        except Exception as exc:  # erro de gateway nao e decisao do modelo: tenta de novo e registra
            err = f"{exc} {getattr(exc, 'read', lambda: b'')()[:300]!r}"
            print("gateway:", err, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gateway falhou {retries}x: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--cup", required=True)
    ap.add_argument("--model", default="cx/gpt-5.6-sol-high")
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--lessons", help="memoria de licoes do direct_reflect.py, injetada no system prompt")
    a = ap.parse_args()

    system = SYSTEM
    if a.lessons:
        from direct_reflect import lessons_text
        block = lessons_text(a.lessons)
        if block:
            system = SYSTEM + "\n\nLessons from your previous attempts (follow them):\n" + block

    ep_dir = Path(a.episode_dir)
    x, y = (float(v) for v in a.cup.split(","))
    obs = DirectEpisode(ep_dir).start(direct_cases.case((x, y), direct_cases.home_arm_pose()),
                                      max_calls=a.max_calls, video=not a.no_video)
    meta = json.loads((ep_dir / "episode.json").read_text())
    meta["policy_model"] = f"{a.model} via OmniRoute"
    meta["system_prompt"] = system
    meta["lessons_file"] = a.lessons
    (ep_dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ep_dir / "policy-calls").mkdir()
    log = open(ep_dir / "policy-log.jsonl", "a")

    n = 0
    while not obs["finished"]:
        n += 1
        sent_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
        images = [{"camera": im["camera"], "file": str(Path(im["path"]).relative_to(ep_dir.resolve())),
                   "sha256": hashlib.sha256(Path(im["path"]).read_bytes()).hexdigest()} for im in obs["images"]]
        resp, latency = ask(a.model, obs, system)
        uses = [c for c in resp.get("content", []) if c.get("type") == "tool_use"]
        text = " ".join(c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text").strip()
        call = {"call": n, "step_id": obs["step_id"], "model": resp.get("model"), "response_id": resp.get("id"),
                "sent_at": sent_at,
                "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "sim_time_s": obs["elapsed_sim_seconds"], "images_sent": images,
                "latency_s": latency, "usage": resp.get("usage"), "model_text": text,
                "observed": {"palm": obs["current_palm"], "arm_joints_rad": obs["arm_joints_rad"],
                             "waist_yaw_rad": obs["waist_yaw_rad"], "gripper_closure": obs["gripper_closure"]}}
        if len(uses) != 1:
            call["error"] = f"modelo devolveu {len(uses)} tool calls"
            obs = DirectEpisode(ep_dir).load().act({"reason": "", "steps": 1, "target": {}})  # conta como rejeicao
        else:
            tool, args = uses[0]["name"], uses[0]["input"]
            call.update(tool=tool, what_i_see=args.get("what_i_see"), arguments=args)
            if tool == "finish_episode":
                (ep_dir / "policy-finish.json").write_text(json.dumps(args, indent=2) + "\n")
                call["result"] = {"status": "finished_by_policy"}
            else:
                obs = DirectEpisode(ep_dir).load().act({
                    "reason": args.get("reason", ""), "steps": args.get("steps", 1),
                    "target": {"position": args.get("position"), "quaternion_wxyz": args.get("quaternion_wxyz"),
                               "gripper": args.get("gripper", "keep")}})
                call["result"] = obs["previous_execution"]
        (ep_dir / "policy-calls" / f"call-{n:03d}.json").write_text(json.dumps(call, indent=2) + "\n")
        log.write(json.dumps(call) + "\n"); log.flush()
        r = call.get("result") or {}
        print(f"[{n:02d}] {call.get('tool')} {r.get('status')} gap={r.get('position_gap_m')} "
              f"{(r.get('errors') or '')} | {call.get('arguments', {}).get('reason', '')[:110]}", flush=True)
        if call.get("tool") == "finish_episode":
            break

    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "direct_finish.py"), str(ep_dir)], check=False)


if __name__ == "__main__":
    main()
