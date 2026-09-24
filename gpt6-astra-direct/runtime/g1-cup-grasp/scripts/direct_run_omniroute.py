"""Roda um episodio Direct com um modelo do OmniRoute como politica (API formato Anthropic).

Uso:
  set -a; source ~/.config/omniroute/omniroute.env; set +a
  .venv/bin/python scripts/direct_run_omniroute.py results/direct-omniroute-01/x0.40_y-0.20 \
      --cup 0.40,-0.20 --model cx/gpt-5.6-sol-high

Cada chamada fica em <ep>/policy-calls/call-NNN.json e uma linha em <ep>/policy-log.jsonl:
id da resposta do gateway, uso de tokens, texto e tool call do modelo, e o que o arnes fez.
"""
import argparse, base64, datetime, hashlib, io, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw
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
Cameras: head_camera (848x480) on the head looking down at the workspace; left_wrist_camera and
right_wrist_camera (424x240) mounted on top of each wrist, looking forward along the hand. All three are
RGB-D. Call probe_depth with a camera and pixel coordinates (u, v) (a labeled grid is drawn on each image)
to get the 3D point under each pixel, in the same frame as the palm pose. It does not move the robot.
The observation includes hand_geometry: approximate fingertip positions and the center of the grasp aperture
between thumb and opposing fingers, from your measured joint angles. The palm is not the leading part of the
hand; plan with these points. Rotating the hand swings the fingertips; call preview_pose to check where they
would go before committing a motion.
gripper_closed_for_s and palm_still_for_s are clocks of your own state: how long the hand has been closed and
how long the palm has been still, in simulated seconds.
In `what_i_see`, describe what each camera shows right now (mug, hand, fingers, table) before deciding.
Put your immediate purpose in `reason` (one or two sentences).
Write `what_i_see` and `reason` in Brazilian Portuguese, the language of the task.
There is no limit on the number of calls: keep working until the task is done. Still, every call costs
time and tokens, so prefer the fewest calls that reliably complete the task.
Call finish_episode only when the mug is lifted and held. The controller rejects unsafe commands (you get
the reason back), so a rejection is information, not a reason to stop."""

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
    {"name": "probe_depth", "description": "Depth camera: 3D point (same frame as the palm) under each pixel (u, v) of "
                                           "the chosen camera image (head 848x480, wrists 424x240). Does not move the robot.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "camera": {"type": "string", "enum": ["head_camera", "left_wrist_camera", "right_wrist_camera"]},
         "points": {"type": "array", "minItems": 1, "maxItems": 8,
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}},
         "required": ["camera", "points"]}},
    {"name": "preview_pose", "description": "Plan without moving: for a candidate palm target, returns where the "
                                            "fingertips and grasp aperture would end up, and the lowest point of the hand "
                                            "along the whole motion (rotations swing the fingers). Does not move the robot.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
         "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
         "steps": {"type": "integer", "minimum": 1, "maximum": 5},
         "gripper": {"type": "string", "enum": ["keep", "open", "closed"]}},
         "required": ["position", "quaternion_wxyz"]}},
    {"name": "finish_episode", "description": "Stop only when the task is complete. Safety is enforced by the "
                                              "controller (unsafe commands are rejected), so keep trying otherwise.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "what_i_see": {"type": "string", "description": "what each camera shows now"},
         "reason": {"type": "string"},
         "outcome": {"type": "string", "enum": ["complete"]}},
         "required": ["what_i_see", "reason", "outcome"]}},
]


def with_grid(path):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    step = 100 if im.width > 500 else 50
    for x in range(step, im.width, step):
        d.line((x, 0, x, im.height), fill=(255, 255, 0), width=1); d.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(step, im.height, step):
        d.line((0, y, im.width, y), fill=(255, 255, 0), width=1); d.text((2, y + 2), str(y), fill=(255, 255, 0))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def content_of(obs):
    hidden = ("images", "next_call", "episode", "remaining_calls", "remaining_sim_seconds", "finished")
    public = {k: v for k, v in obs.items() if k not in hidden}
    blocks = [{"type": "text", "text": "Current measured observation:\n" + json.dumps(public)}]
    for im in obs["images"]:
        blocks.append({"type": "text", "text": f"Camera: {im['camera']}"})
        data = with_grid(im["path"])
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    return blocks


def post(body, retries=3):
    req = urllib.request.Request(f"{GATEWAY}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                  "x-api-key": os.environ["OMNIROUTE_API_KEY"]})
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as exc:  # erro de gateway nao e decisao do modelo: tenta de novo e registra
            err = f"{exc} {getattr(exc, 'read', lambda: b'')()[:300]!r}"
            print("gateway:", err, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gateway falhou {retries}x: {err}")


MAX_PROBE_ROUNDS = 4
TOPPLED_DEG = 80.0
TILT_BEFORE_CLOSE_DEG = 15.0   # mesmo limite do direct_finish.py


INFO_TOOLS = ("probe_depth", "preview_pose")
RAW_DIR = None   # definido no main: <ep>/raw-runner


def ask(model, obs, system, probe, effort=None, preview=None):
    """Uma decisao: o modelo pode sondar profundidade algumas vezes antes de agir. Devolve a resposta final
    (com actuate/finish), a latencia somada, as sondagens e o uso de tokens somado."""
    msgs = [{"role": "user", "content": content_of(obs)}]
    probes, usage, t0 = [], {"input_tokens": 0, "output_tokens": 0}, time.time()
    for rnd in range(MAX_PROBE_ROUNDS + 1):
        last = rnd == MAX_PROBE_ROUNDS
        body = {"model": model, "max_tokens": 2000 if not effort else 16000, "system": system,
                "tools": TOOLS if not last else [t for t in TOOLS if t["name"] not in INFO_TOOLS],
                "tool_choice": {"type": "any", "disable_parallel_tool_use": True}, "messages": msgs}
        if effort:  # nivel de raciocinio da politica (o gateway traduz output_config.effort para o upstream)
            body["output_config"] = {"effort": effort}
        resp = post(body)
        if RAW_DIR is not None:   # log bruto proprio: nao depende da rotacao do gateway (que apaga os antigos)
            from export_raw_logs import strip_images
            RAW_DIR.mkdir(exist_ok=True)
            n = len(list(RAW_DIR.glob("*.json"))) + 1
            (RAW_DIR / f"{n:04d}.json").write_text(json.dumps(
                {"requestBody": strip_images(body), "responseBody": resp}, indent=1, ensure_ascii=False) + "\n")
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0)
        uses = [c for c in resp.get("content", []) if c.get("type") == "tool_use"]
        acts = [u for u in uses if u["name"] not in INFO_TOOLS]
        asked = [u for u in uses if u["name"] in INFO_TOOLS]
        if (len(acts) == 1 and not asked) or last or not uses:
            resp["usage"] = usage
            return resp, round(time.time() - t0, 2), probes
        # sondas pedidas (com ou sem acao junto): executa as sondas e pede UMA acao depois de ler os resultados
        results = []
        for u in uses:
            if u["name"] == "probe_depth":
                pts = u["input"].get("points", [])[:8]
                cam = u["input"].get("camera", "head_camera")
                out = probe(cam, pts)
                probes.append({"camera": cam, "points": pts, "result": out, "response_id": resp.get("id")})
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(out)})
            elif u["name"] == "preview_pose":
                out = preview(u["input"])
                probes.append({"preview": u["input"], "result": out, "response_id": resp.get("id")})
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(out)})
            else:
                results.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": True,
                                "content": "not executed: call exactly one action per turn, after reading the probe results"})
        msgs += [{"role": "assistant", "content": resp["content"]}, {"role": "user", "content": results}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--cup", required=True)
    ap.add_argument("--model", default="cx/gpt-5.6-sol-high")
    # teto de seguranca do operador (custo), invisivel para o modelo; ele nao sabe que existe
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--effort", help="esforco de raciocinio da politica (low|medium|high|xhigh|max); padrao do modelo se omitido")
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
                                      max_calls=a.max_calls, max_sim_seconds=3600.0, video=not a.no_video)
    meta = json.loads((ep_dir / "episode.json").read_text())
    meta["policy_model"] = f"{a.model} via OmniRoute"
    meta["system_prompt"] = system
    meta["lessons_file"] = a.lessons
    meta["policy_effort"] = a.effort
    (ep_dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ep_dir / "policy-calls").mkdir()
    global RAW_DIR
    RAW_DIR = ep_dir / "raw-runner"
    log = open(ep_dir / "policy-log.jsonl", "a")

    n = 0
    while not obs["finished"]:
        n += 1
        sent_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
        images = [{"camera": im["camera"], "file": str(Path(im["path"]).relative_to(ep_dir.resolve())),
                   "sha256": hashlib.sha256(Path(im["path"]).read_bytes()).hexdigest()} for im in obs["images"]]
        resp, latency, probes = ask(a.model, obs, system, effort=a.effort,
                                    preview=lambda x: DirectEpisode(ep_dir).load().preview(x["position"], x["quaternion_wxyz"],
                                                                                         x.get("steps", 3), x.get("gripper", "keep")),
                                    probe=lambda cam, pts: DirectEpisode(ep_dir).load().depth_probe(pts, cam=cam,
                                                                             size=(848, 480) if cam == "head_camera" else (424, 240)))
        uses = [c for c in resp.get("content", []) if c.get("type") == "tool_use" and c["name"] not in INFO_TOOLS][:1]
        text = " ".join(c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text").strip()
        call = {"call": n, "step_id": obs["step_id"], "model": resp.get("model"), "response_id": resp.get("id"),
                "sent_at": sent_at,
                "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "sim_time_s": obs["elapsed_sim_seconds"], "images_sent": images,
                "latency_s": latency, "usage": resp.get("usage"), "model_text": text, "depth_probes": probes,
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
        # regra do operador (o modelo nao ve): objeto tombado e fim de episodio, a tarefa exige ele em pe
        truth = ep_dir / "truth" / f"step-{obs['step_id']:03d}.json"
        if truth.exists() and json.loads(truth.read_text()).get("cup_tilt_deg", 0) > TOPPLED_DEG and call.get("tool") != "finish_episode":
            call["terminated_by_operator"] = f"objeto tombado (inclinacao acima de {TOPPLED_DEG} graus)"
        # criterio do avaliador que nao tem volta: caneca inclinada acima do limite sem nenhum dedo nela (aproximacao)
        if not call.get("terminated_by_operator") and call.get("tool") != "finish_episode":
            s_path = ep_dir / "actions" / f"samples-{obs['step_id']:03d}.json"
            if s_path.exists() and any(x["cup_tilt_deg"] >= TILT_BEFORE_CLOSE_DEG and not x["fingers"]
                                       for x in json.loads(s_path.read_text())):
                call["terminated_by_operator"] = (f"inclinacao acima de {TILT_BEFORE_CLOSE_DEG} graus antes da pega: "
                                                  "o avaliador ja reprova, continuar so gasta chamadas")
        (ep_dir / "policy-calls" / f"call-{n:03d}.json").write_text(json.dumps(call, indent=2) + "\n")
        log.write(json.dumps(call) + "\n"); log.flush()
        r = call.get("result") or {}
        print(f"[{n:02d}] {call.get('tool')} {r.get('status')} gap={r.get('position_gap_m')} "
              f"{(r.get('errors') or '')} | {call.get('arguments', {}).get('reason', '')[:110]}", flush=True)
        if call.get("tool") == "finish_episode":
            break
        if call.get("terminated_by_operator"):
            (ep_dir / "interrupted.json").write_text(json.dumps({"interrupted_by": "regra do operador", "at_call": n,
                                                                  "why": call["terminated_by_operator"]}) + "\n")
            break

    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "direct_finish.py"), str(ep_dir)], check=False)


if __name__ == "__main__":
    main()
