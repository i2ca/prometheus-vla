"""Reflexao pos-episodio: um critico ve tudo do episodio e escreve licoes gerais para a politica.

Uso:
  set -a; source ~/.config/omniroute/omniroute.env; set +a
  .venv/bin/python scripts/direct_reflect.py results/direct-astra-01/x0.40_y-0.20 --memory results/direct-lessons.json

O critico recebe o que a politica nao recebe: verdade do simulador por step (contato dos dedos, inclinacao
e altura da caneca, distancia real palma-caneca) e o veredito do avaliador. Por isso as licoes passam por
um filtro: nada de coordenada ou numero da cena, senao a politica do proximo episodio receberia verdade
privilegiada por tabela. A memoria (--memory) acumula licoes; o runner injeta com --lessons.
Estilo Reflexion (Shinn et al., 2023): nao muda pesos, muda o que o modelo le.
"""
import argparse, base64, datetime, io, json, os, re, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

GATEWAY = os.environ.get("OMNIROUTE_URL", "http://127.0.0.1:20128")

CRITIC = """You are reviewing one episode of a vision-language policy that controls a Unitree G1 right arm
(Dex3 hand) in MuJoCo to grasp a mug and lift it. The policy only sees 3 RGB cameras (head, left wrist,
right wrist) plus measured palm pose, joints and gripper closure, and it can query the head depth camera
(probe_depth: 3D point under a pixel). Judge whether it used that sensor well. It does NOT know the mug position.
You see more than the policy did: the simulator truth per step and the evaluator verdict.

Your job: explain why the episode failed (or succeeded) and write lessons that will be added to the
policy's system prompt in FUTURE episodes, where the mug may be somewhere else.

Hard rules for lessons (they become permanent rules in the policy's system prompt):
- Generic manipulation principles, valid for ANY object and ANY later stage of a kitchen task
  (mug, jug, kettle, filter, spoon...). Say "the object" / "the target", never name this object.
- Never describe this episode: no step numbers, no "you did X", no which finger touched what, no
  "last time". The policy must not be told what went wrong; it must receive a rule that would have
  prevented it and that also holds elsewhere.
- No coordinates, distances, angles or numbers from the scene. Qualitative guidance only.
- Each lesson must be checkable from the policy's own inputs (images + proprioception) and must say
  what to DO next, not only when to stop. A rule that only makes the policy stop, wait or retreat
  will make it stall; pair any "stop" with the alternative action to try.
- At most 4 lessons, most important first. Put the evidence (steps) in evidence_steps/why, which the
  policy never sees.

Always write lessons, whether the episode was accepted or not. The policy has no call limit, so
efficiency is part of quality: every call costs time and tokens (see COST). If the episode was
ACCEPTED, focus on efficiency: which calls were unnecessary, which detours could be skipped, how to
reach the same result with fewer calls without losing reliability. If it FAILED, focus on what would
have made it succeed, and add efficiency only when waste was significant.

If the policy already had lessons in its prompt (listed as ACTIVE LESSONS), judge them against what
happened: which helped, which caused harmful behavior (for example stalling, repeated retreat, or new
collisions). Return the COMPLETE revised set that replaces them: keep what helped, rewrite or drop what hurt.

Answer only with JSON:
{"diagnosis": "...", "failure_mode": "...",
 "lessons_review": [{"lesson": "...", "verdict": "helped|hurt|neutral", "why": "..."}],
 "lessons": [{"lesson": "...", "evidence_steps": [..], "why": "..."}]}"""

# numero com unidade de cena, ou decimal com 2+ casas: cheira a coordenada vazada
LEAK = re.compile(r"(-?\d+[.,]\d{2,})|(\d+(?:[.,]\d+)?\s?(?:cm|mm|m\b|°|deg|degrees|rad))"
                  r"|\b(mugs?|cups?|caneca|steps? \d|episode|last time|previous attempt|middle finger|you (?:did|tipped|pushed))\b", re.I)


def b64(path, size):
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def truth_summary(ep, call):
    """Verdade do simulador durante a execucao dessa chamada (so o critico ve)."""
    s = ep / "actions" / f"samples-{call['step_id'] + 1:03d}.json"
    if call.get("tool") != "actuate" or not s.exists():
        return None
    samples = json.loads(s.read_text())
    if not samples:
        return None
    dist = [float(np.linalg.norm(np.array(x["palm"][:2]) - np.array(x["cup"][:2]))) for x in samples]
    return {"fingers_touching_mug": sorted({f for x in samples for f in x["fingers"]}),
            "mug_tilt_deg_max": round(max(x["cup_tilt_deg"] for x in samples), 1),
            "mug_height_change_cm": round((samples[-1]["cup"][2] - samples[0]["cup"][2]) * 100, 1),
            "palm_to_mug_horizontal_cm_end": round(dist[-1] * 100, 1),
            "palm_height_above_mug_base_cm_end": round((samples[-1]["palm"][2] - samples[-1]["cup"][2]) * 100, 1),
            "mug_touches_table_end": samples[-1]["cup_table"]}


def build(ep):
    calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
    report = json.loads((ep / "report.json").read_text())
    meta = json.loads((ep / "episode.json").read_text())
    active = meta.get("system_prompt", "").split("Lessons from your previous attempts (follow them):\n")
    content = [{"type": "text", "text": "TASK: " + meta["case"]["instruction"] +
                "\nPOLICY SYSTEM PROMPT RULES: palm target within 5 cm / 0.35 rad per call; gripper open/closed/keep."
                "\nACTIVE LESSONS in the policy prompt during this episode:\n" + (active[1] if len(active) > 1 else "(none)")}]
    for c in calls:
        a = c.get("arguments", {})
        step = {"step": c["call"], "tool": c.get("tool"),
                "policy_saw": c.get("what_i_see"), "policy_reason": a.get("reason"),
                "action": {k: v for k, v in a.items() if k not in ("what_i_see", "reason")},
                "measured_palm_before": c.get("observed", {}).get("palm"),
                "hand_geometry_before": json.loads((ep / "obs" / f"step-{c['step_id']:03d}" / "observation.json").read_text()).get("hand_geometry")
                if (ep / "obs" / f"step-{c['step_id']:03d}" / "observation.json").exists() else None,
                "gripper_closure_before": c.get("observed", {}).get("gripper_closure"),
                "harness_result": c.get("result"),
                "depth_probes_by_policy": [{"camera": pr.get("camera"), "pixels": pr["points"],
                                            "points_3d": [r.get("point") for r in pr["result"]]}
                                           for pr in c.get("depth_probes", []) if "points" in pr],
                "pose_previews_by_policy": [{"asked": pr["preview"], "lowest_hand_point": pr["result"].get("lowest_hand_point_along_path")}
                                            for pr in c.get("depth_probes", []) if "preview" in pr],
                "SIM_TRUTH_hidden_from_policy": truth_summary(ep, c)}
        content.append({"type": "text", "text": json.dumps(step, ensure_ascii=False)})
        head = ep / "obs" / f"step-{c['step_id']:03d}" / "head_camera.png"
        if head.exists():
            content.append({"type": "text", "text": f"head_camera the policy saw at step {c['call']}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": b64(head, (640, 360))}})
    frames = sorted((ep / "frames").glob("*.jpg"))
    if frames:
        content.append({"type": "text", "text": "External 6-camera view at the end of the episode (policy never saw this):"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": b64(frames[-1], (1280, 512))}})
    usage = [c.get("usage") or {} for c in calls]
    path = [c["observed"]["palm"]["position"] for c in calls if c.get("observed")]
    content.append({"type": "text", "text": "COST: " + json.dumps({
        "calls": len(calls), "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "model_wait_s": round(sum(c.get("latency_s") or 0 for c in calls), 1),
        "sim_seconds": report.get("sim_seconds"),
        "palm_path_length_cm": round(sum(float(np.linalg.norm(np.array(b) - np.array(a))) for a, b in zip(path, path[1:])) * 100, 1),
        "gripper_close_commands": sum(1 for c in calls if c.get("arguments", {}).get("gripper") == "closed")})})
    content.append({"type": "text", "text": "EVALUATOR: " + json.dumps(
        {k: report.get(k) for k in ("accepted", "lift_m", "retained_for_2s", "upright_grasp",
                                    "cup_tilt_deg_before_close", "calls_used", "aborted")})})
    return content, report


def ask(model, content, effort="max"):
    # esforco pedido no corpo: na spark o gateway nao aceita mais o sufixo -xhigh para o gpt-6-astra (catalogo
    # sincronizado sem niveis de esforco para ele); output_config.effort funciona e aparece em tokens_reasoning
    body = {"model": model, "max_tokens": 16000, "system": CRITIC, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(f"{GATEWAY}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                  "x-api-key": os.environ["OMNIROUTE_API_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=600))


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--memory", required=True, help="arquivo de licoes acumuladas")
    ap.add_argument("--model", default="cx/gpt-6-astra")
    a = ap.parse_args()
    ep = Path(a.episode_dir)
    content, report = build(ep)
    resp = ask(a.model, content)
    text = " ".join(c.get("text", "") for c in resp["content"] if c.get("type") == "text")
    out = parse(text)
    kept, rejected = [], []
    for L in out.get("lessons", [])[:4]:
        hit = LEAK.search(L.get("lesson", ""))
        (rejected if hit else kept).append({**L, **({"rejected_for": hit.group(0)} if hit else {})})
    record = {"episode": str(ep), "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "critic_model": resp.get("model"), "response_id": resp.get("id"), "usage": resp.get("usage"),
              "accepted": report.get("accepted"), "diagnosis": out.get("diagnosis"),
              "failure_mode": out.get("failure_mode"), "lessons_review": out.get("lessons_review", []),
              "lessons": kept, "rejected_lessons": rejected}
    (ep / "reflection.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    mem_path = Path(a.memory)
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else {"reflections": []}
    mem["reflections"].append(record)
    mem_path.write_text(json.dumps(mem, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


def lessons_text(memory_path):
    """Bloco que o runner cola no system prompt: o conjunto revisado mais recente (substitui os anteriores)."""
    mem = json.loads(Path(memory_path).read_text())
    return "\n".join(f"- {L['lesson'].strip()}" for L in mem["reflections"][-1]["lessons"])


if __name__ == "__main__":
    main()
