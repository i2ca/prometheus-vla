"""Narracao de TODOS os passos de um episodio Direct: o que o robo disse que viu, o que decidiu e por que.

Uso: .venv/bin/python scripts/make_narration.py results/<ep>/x0.40_y-0.20 [--model agy/gemini-3.7-flash-high]

- "ve" e "porque": traducao curta em PT de `what_i_see` e `reason` do proprio modelo, feita por um modelo de
  texto do OmniRoute com instrucao de nao acrescentar nada.
- a acao (direcao, centimetros, rotacao, mao, recusa) sai do log pelo codigo, sem passar por modelo.
- se ja existir narration.json, intro/outro sao mantidos e as notas escritas a mao para um passo (verdade do
  simulador) sao anexadas depois da narracao automatica daquele passo. O original vira narration-notas.json.
"""
import argparse, json, os, re, urllib.request
from pathlib import Path
import numpy as np

GATEWAY = os.environ.get("OMNIROUTE_URL", "http://127.0.0.1:20128")
PARTS = {"shoulder_roll": "o ombro", "shoulder_yaw": "o ombro", "shoulder_pitch": "o ombro", "elbow": "o cotovelo",
         "wrist": "o punho", "thumb": "o polegar", "index": "o indicador", "middle": "o dedo médio", "palm": "a palma",
         "torso": "o tronco", "pelvis": "a pélvis", "left_": "o braço esquerdo"}


def part(name):
    return next((v for k, v in PARTS.items() if k in name), name.replace("_link", "").replace("_", " "))


def speakable(t):
    """Coordenadas e simbolos de um jeito que a voz le bem."""
    t = re.sub(r"\bz\s*≈\s*", "altura de cerca de ", t)
    t = re.sub(r"\bz\s*=\s*", "altura ", t)
    t = t.replace("≈", "cerca de ").replace("–", " a ").replace("/", " ou ")
    return t


def cm(v):
    return f"{abs(v) * 100:.1f}".replace(".0", "").replace(".", ",")


def action_text(c):
    a, res = c.get("arguments", {}), c.get("result") or {}
    if c.get("tool") == "finish_episode":
        return "declara a tarefa completa"
    if c.get("error"):
        return "não devolve uma decisão válida"
    if res.get("status") == "rejected":
        err = (res.get("errors") or [""])[0]
        m = re.search(r"(right_\w+|left_\w+) (?:encosta em|x) (\w+)", err)
        if m:
            a_, b_ = part(m.group(1)), part(m.group(2))
            em = ("no " + b_[2:]) if b_.startswith("o ") else ("na " + b_[2:]) if b_.startswith("a ") else "em " + b_
            de = ("do " + b_[2:]) if b_.startswith("o ") else ("da " + b_[2:]) if b_.startswith("a ") else "de " + b_
            if "ensaio" in err:
                return f"pede um movimento que o controlador recusa, porque no ensaio {a_} encostaria {em}"
            return f"pede um movimento que o controlador recusa, porque {a_} passaria perto demais {de}"
        return "pede um movimento que o controlador recusa: " + err.split(";")[0].split(" nada foi executado")[0].rstrip(". ,")
    obs = c.get("observed", {})
    before = np.array(obs.get("palm", {}).get("position", a.get("position")))
    d = np.array(a["position"]) - before
    parts = []
    for val, pos, neg in ((d[0], "para a frente", "para trás"), (d[1], "para a esquerda", "para a direita"),
                          (d[2], "para cima", "para baixo")):
        if abs(val) >= 0.005:
            parts.append(f"{cm(val)} centímetros {pos if val > 0 else neg}")
    q0 = np.array(obs.get("palm", {}).get("quaternion_wxyz", a["quaternion_wxyz"]), float)
    q1 = np.array(a["quaternion_wxyz"], float)
    ang = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))))))
    move = ("move a mão " + ", ".join(parts)) if parts else ""
    if ang >= 3:
        move = (move + " e " if move else "") + f"gira a mão {ang:.0f} graus"
    closed_before = (obs.get("gripper_closure") or 0) > 0.5
    g = a.get("gripper")
    hand = {"closed": "mantém a mão fechada" if closed_before else "fecha a mão",
            "open": "abre a mão" if closed_before else "", "keep": ""}.get(g, "")
    if not move:
        move = "fica parado" if not hand or hand.startswith("mantém") else ""
    txt = ", ".join(x for x in (move, hand) if x)
    if res.get("status") == "aborted_on_contact":
        txt += "; o movimento é interrompido por contato"
    return txt


def translate(model, calls, batch=8, tries=3):
    """Em lotes pequenos: resposta grande de uma vez volta com JSON quebrado."""
    out = {}
    for i in range(0, len(calls), batch):
        chunk = calls[i:i + batch]
        for attempt in range(tries):
            try:
                out.update(_translate(model, chunk)); break
            except (ValueError, AttributeError, RuntimeError) as exc:
                if attempt == tries - 1:
                    raise
                print(f"lote {i // batch}: tentativa {attempt + 1} falhou ({exc}); de novo")
    return out


def _translate(model, calls):
    items = [{"n": c["call"], "see": c.get("what_i_see") or "", "reason": c.get("arguments", {}).get("reason", "")}
             for c in calls]
    prompt = ("Traduza para português do Brasil, de forma curta e fiel, SEM acrescentar informação. Para cada item "
              "devolva 've' (o que o robô diz que vê, no máximo 22 palavras, escrito para completar a frase 'Ele vê ...', "
              "começando por minúscula; ex.: 'a caneca em pé à esquerda da mão, e o punho direito mostrando a borda') "
              "e 'porque' (a intenção declarada, no máximo 16 palavras, começando por 'para' quando couber). "
              "Devolva também 've_completo' e 'porque_completo': a tradução COMPLETA e fiel de 'see' e 'reason', frase "
              "por frase, sem resumir nem cortar, mantendo números e unidades. "
              "Responda só com JSON: {\"<n>\": {\"ve\": \"...\", \"porque\": \"...\", \"ve_completo\": \"...\", "
              "\"porque_completo\": \"...\"}}.\n\n" + json.dumps(items, ensure_ascii=False))
    body = {"model": model, "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{GATEWAY}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                  "x-api-key": os.environ["OMNIROUTE_API_KEY"]})
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    text = " ".join(c.get("text", "") for c in resp["content"] if c.get("type") == "text")
    out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    missing = [c["call"] for c in calls if str(c["call"]) not in out]
    if missing:
        raise RuntimeError(f"traducao sem os passos {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--model", default="agy/gemini-3.7-flash-high")
    a = ap.parse_args()
    ep = Path(a.episode_dir)
    calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
    old_path, notes_path = ep / "narration.json", ep / "narration-notas.json"
    if not notes_path.exists() and old_path.exists():
        notes_path.write_text(old_path.read_text())
    old = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    tr = translate(a.model, calls)
    steps = {}
    for c in calls:
        n = str(c["call"])
        t = tr[n]
        see = t.get("ve", "").strip().rstrip(".")
        why = t.get("porque", "").strip().rstrip(".")
        act = action_text(c)
        see, why = speakable(see), speakable(why)
        txt = f"Passo {n}. Ele vê {see}." if see else f"Passo {n}."
        if act:
            txt += f" Decide: {act}" + (f", {why}." if why and c.get("tool") != "finish_episode" else ".")
        note = old.get("steps", {}).get(n)
        if note:
            txt += " " + re.sub(r"^Passo [^.]+\.\s*", "", note)
        steps[n] = txt
    panel = {str(c["call"]): {"ve": tr[str(c["call"])].get("ve_completo", ""),
                              "porque": tr[str(c["call"])].get("porque_completo", "")} for c in calls}
    out = {"voice": old.get("voice", "pt-BR-AntonioNeural"), "intro": old.get("intro"), "steps": steps,
           "outro": old.get("outro"), "panel_pt": panel}
    old_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for n in list(steps)[:5]:
        print(steps[n])


if __name__ == "__main__":
    main()
