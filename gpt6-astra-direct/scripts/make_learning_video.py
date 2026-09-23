"""Video do ciclo de aprendizado: episodio -> critico (o que viu, diagnostico, o que muda no prompt)
-> proximo episodio -> validacao, repetido para cada rodada.

Uso: .venv/bin/python scripts/make_learning_video.py results/learning-video.json --out results/ciclo.mp4

O JSON lista a ordem: {"voice": ..., "intro": "...", "rounds": [{"episode": dir, "critic_text": [..]}, ...],
"validation_text": {"<ep>": "..."}, "outro": "..."}. Cada episodio usa o narration.json dele.
Metricas de validacao saem da verdade do simulador (actions/samples-*.json), nunca da fala do modelo.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from make_direct_story import (Story, Writer, W, H, FPS, BG, PANEL, CARD, INK, MUTED, LINE, ACCENT, OK, BAD, WARN,
                               font, wrap, fit, chip)

PURPLE, ORANGE = (192, 132, 252), (251, 146, 60)
ALARM = re.compile(r"\b(tipped|tipping|tilted|apparent tilt|unintended contact|possible (?:lateral )?(?:finger )?contact|"
                   r"contact[- ]relief|relieve)\b", re.I)
VERDICT = {"helped": OK, "hurt": BAD, "neutral": MUTED}


def metrics(ep):
    """Numeros de validacao a partir da verdade do simulador."""
    ep = Path(ep)
    calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
    rep = json.loads((ep / "report.json").read_text())
    dist, open_contact, tilt, retreats, moves, alarms = [], 0, 0.0, 0, 0, 0
    prev = None
    for c in calls:
        s = ep / "actions" / f"samples-{c['step_id'] + 1:03d}.json"
        samples = json.loads(s.read_text()) if s.exists() and c.get("tool") == "actuate" else []
        grip = c.get("arguments", {}).get("gripper")
        truth_touch = False
        for x in samples:
            d = float(np.linalg.norm(np.array(x["palm"][:2]) - np.array(x["cup"][:2])))
            dist.append(d)
            tilt = max(tilt, x["cup_tilt_deg"])
            if x["fingers"]:
                truth_touch = True
                if grip == "open":
                    open_contact += 1
        if samples:
            end = dist[-1]
            if prev is not None:
                moves += 1; retreats += end > prev + 0.002
            prev = end
        # alarme falso: a ACAO foi motivada por tombo/contato (campo reason) e a verdade nao tem nenhum dos dois
        if ALARM.search(c.get("arguments", {}).get("reason", "")) and not truth_touch and tilt < 3:
            alarms += 1
    ab = rep.get("aborted") or {}
    return {"accepted": bool(rep.get("accepted")), "lift_cm": rep.get("lift_m", 0) * 100,
            "min_dist_cm": min(dist) * 100 if dist else None, "start_dist_cm": dist[0] * 100 if dist else None,
            "open_contact_frames": open_contact, "max_tilt": tilt,
            "retreat_ratio": retreats / moves if moves else 0.0, "false_alarms": alarms,
            "self_collision": bool(ab.get("self_collision")), "calls": len(calls)}


class Learning:
    def __init__(self, cfg_path):
        self.cfg = json.loads(Path(cfg_path).read_text())
        self.root = Path(cfg_path).parent

    def header(self, d, label):
        d.rectangle((0, 0, W, 64), fill=(9, 10, 14))
        d.text((24, 12), "G1 · aprendendo com os próprios erros", font=font(24, True), fill=INK)
        d.text((W - 24 - d.textlength(label, font=font(16, True)), 22), label, font=font(16, True), fill=ACCENT)

    def base(self, label):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        self.header(d, label)
        return img, d

    def box(self, d, x0, y0, x1, y1, tone, title):
        fill = tuple(int(PANEL[i] * 0.82 + tone[i] * 0.10) for i in range(3))
        d.rounded_rectangle((x0, y0, x1, y1), 12, fill=fill, outline=tone, width=2)
        d.text((x0 + 18, y0 + 12), title, font=font(18, True), fill=tone)

    def para(self, d, x, y, text, width, fnt, color=INK, max_lines=99, gap=6):
        for ln in wrap(d, text, fnt, width)[:max_lines]:
            d.text((x, y), ln, font=fnt, fill=color); y += fnt.size + gap
        return y

    # ---------- capitulo do critico ----------
    def critic_saw(self, ep, refl, label):
        img, d = self.base(label)
        d.text((40, 88), "1 · O que o crítico vê", font=font(34, True), fill=PURPLE)
        d.text((40, 136), "Tudo o que a política viu, mais o que ela não pode ver: a verdade do simulador e a visão externa.",
               font=font(18), fill=MUTED)
        calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
        # miniaturas do que a politica viu
        self.box(d, 40, 180, 1000, 1040, ACCENT, "câmera da cabeça em cada step (o que a política recebeu)")
        cols = 3
        tw, th = 300, 169
        for i, c in enumerate(calls[:9]):
            p = ep / "obs" / f"step-{c['step_id']:03d}" / "head_camera.png"
            x, y = 60 + (i % cols) * (tw + 12), 222 + (i // cols) * (th + 44)
            if p.exists():
                img.paste(fit(Image.open(p).convert("RGB"), tw, th), (x, y))
            d.text((x, y + th + 6), f"step {c['call']} · {c.get('tool')}", font=font(14), fill=INK)
        # verdade por step
        self.box(d, 1030, 180, W - 40, 740, ORANGE, "verdade do simulador (a política não vê)")
        y = 222
        d.text((1050, y), "step  dedos na caneca   inclin.   distância mão-caneca", font=font(14, mono=True), fill=MUTED); y += 26
        for c in calls:
            s = ep / "actions" / f"samples-{c['step_id'] + 1:03d}.json"
            if c.get("tool") != "actuate" or not s.exists():
                d.text((1050, y), f"{c['call']:>4}  (encerrou)", font=font(15, mono=True), fill=INK); y += 26; continue
            smp = json.loads(s.read_text())
            fing = ",".join(sorted({f for x in smp for f in x["fingers"]})) or "nenhum"
            tl = max(x["cup_tilt_deg"] for x in smp)
            dist = float(np.linalg.norm(np.array(smp[-1]["palm"][:2]) - np.array(smp[-1]["cup"][:2]))) * 100
            col = WARN if fing != "nenhum" or tl > 5 else INK
            d.text((1050, y), f"{c['call']:>4}  {fing:<16} {tl:5.0f}°   {dist:6.1f} cm", font=font(15, mono=True), fill=col)
            y += 26
        frames = sorted((ep / "frames").glob("*.jpg"))
        self.box(d, 1030, 760, W - 40, 1040, MUTED, "visão externa no fim do episódio")
        if frames:
            im = Image.open(frames[-1]).convert("RGB")
            img.paste(fit(im.crop((0, 32, im.width, im.height)), 820, 230), (1050, 800))
        return img

    def critic_diag(self, refl, label):
        img, d = self.base(label)
        d.text((40, 88), "2 · O que o crítico entendeu", font=font(34, True), fill=PURPLE)
        d.text((40, 136), f"crítico: {refl.get('critic_model')} · original em inglês, sem edição", font=font(18), fill=MUTED)
        self.box(d, 40, 180, 1180, 700, PURPLE, "diagnóstico")
        self.para(d, 62, 222, refl.get("diagnosis", ""), 1090, font(19), max_lines=18)
        self.box(d, 40, 720, 1180, 1040, BAD, "modo de falha")
        self.para(d, 62, 762, refl.get("failure_mode", ""), 1090, font(21, True), max_lines=10)
        self.box(d, 1210, 180, W - 40, 1040, ACCENT, "lições que estavam no prompt")
        y = 222
        rev = refl.get("lessons_review") or []
        if not rev:
            self.para(d, 1232, y, "Nenhuma: era o primeiro episódio.", 640, font(18), MUTED)
        for L in rev:
            y = chip(d, 1232, y, {"helped": "AJUDOU", "hurt": "ATRAPALHOU", "neutral": "NEUTRA"}[L.get("verdict", "neutral")],
                     VERDICT.get(L.get("verdict"), MUTED)) and y + 34
            y = self.para(d, 1232, y, L["lesson"], 640, font(14), INK, max_lines=3, gap=4)
            y = self.para(d, 1232, y + 2, "por quê: " + L.get("why", ""), 640, font(13), MUTED, max_lines=3, gap=4) + 14
        return img

    def critic_change(self, refl, label):
        img, d = self.base(label)
        d.text((40, 88), "3 · O que muda no prompt do robô", font=font(34, True), fill=PURPLE)
        d.text((40, 136), "O robô NÃO é informado do erro. Recebe só princípios gerais, válidos para qualquer objeto.",
               font=font(18), fill=WARN)
        self.box(d, 40, 180, 620, 1040, MUTED, "prompt base (não muda)")
        self.para(d, 62, 222, "Você é a política visuomotora de um G1... Use só as imagens RGB e a propriocepção medida. "
                  "Você não sabe a pose da caneca. Cada chamada: UM alvo absoluto para a palma, até 5 cm e 0,35 rad "
                  "da pose medida; depois olhe a próxima observação. ...", 540, font(16), MUTED)
        self.box(d, 650, 180, W - 40, 1040, OK, "+ lições adicionadas (substituem as anteriores)")
        y = 222
        for i, L in enumerate(refl.get("lessons", []), 1):
            d.text((672, y), f"{i}.", font=font(20, True), fill=OK)
            y = self.para(d, 706, y, L["lesson"], 1150, font(17), INK, max_lines=6, gap=5) + 16
        rej = refl.get("rejected_lessons") or []
        d.text((672, 1000), f"filtro anti-vazamento: {len(rej)} lição(ões) barrada(s) por citar números, objeto ou o episódio",
               font=font(15), fill=MUTED)
        return img

    # ---------- validacao ----------
    def validation(self, a, b, label, names):
        ma, mb = metrics(a), metrics(b)
        img, d = self.base(label)
        d.text((40, 88), "Validação: o problema foi corrigido?", font=font(34, True), fill=INK)
        d.text((40, 136), "Números da verdade do simulador, não do que o modelo diz.", font=font(18), fill=MUTED)
        # (rotulo, formato, valor numerico, maior e melhor?)
        rows = [
            ("aceito pelo avaliador", lambda m: "sim" if m["accepted"] else "não", lambda m: m["accepted"], True),
            ("caneca levantada", lambda m: f"{m['lift_cm']:.1f} cm", lambda m: round(m["lift_cm"]), True),
            ("menor distância mão-caneca", lambda m: f"{m['min_dist_cm']:.1f} cm", lambda m: round(m["min_dist_cm"]), False),
            ("contato com a mão aberta (quadros)", lambda m: str(m["open_contact_frames"]), lambda m: m["open_contact_frames"], False),
            ("inclinação máx. da caneca", lambda m: f"{m['max_tilt']:.0f}°", lambda m: round(m["max_tilt"] / 3), False),
            ("comandos que afastaram a mão", lambda m: f"{m['retreat_ratio']*100:.0f}%", lambda m: round(m["retreat_ratio"] * 10), False),
            ("alarme falso de contato/tombo", lambda m: str(m["false_alarms"]), lambda m: m["false_alarms"], False),
            ("autocolisão", lambda m: "sim" if m["self_collision"] else "não", lambda m: m["self_collision"], False),
            ("chamadas ao modelo", lambda m: str(m["calls"]), None, None),
        ]
        x0, x1, x2, x3 = 120, 900, 1250, 1600
        d.text((x1, 200), names[0], font=font(20, True), fill=MUTED)
        d.text((x2, 200), names[1], font=font(20, True), fill=INK)
        y, fixed, worse = 250, [], []
        for label_, fmt, val, up in rows:
            d.line((x0, y - 8, W - 120, y - 8), fill=LINE)
            d.text((x0, y), label_, font=font(22), fill=INK)
            d.text((x1, y), fmt(ma), font=font(22, mono=True), fill=MUTED)
            if val is None:
                state, col = "", INK
            else:
                va, vb = float(val(ma)), float(val(mb))
                state = "igual" if va == vb else ("melhorou" if (vb > va) == up else "piorou")
                col = {"igual": MUTED, "melhorou": OK, "piorou": BAD}[state]
                (fixed if state == "melhorou" else worse if state == "piorou" else []).append(
                    f"{label_} ({fmt(ma)} para {fmt(mb)})")
            d.text((x2, y), fmt(mb), font=font(22, True, mono=True), fill=col)
            d.text((x3, y), state, font=font(18, True), fill=col)
            y += 60
        y += 20
        y = self.para(d, x0, y, "Corrigido: " + ("; ".join(fixed) or "nada"), W - 240, font(22, True), OK, max_lines=3)
        y = self.para(d, x0, y + 8, "Piorou: " + ("; ".join(worse) or "nada"), W - 240, font(22, True), BAD if worse else MUTED, max_lines=3)
        tarefa = "Tarefa cumprida." if mb["accepted"] else "Tarefa ainda não cumprida: a caneca não foi levantada."
        self.para(d, x0, y + 8, tarefa, W - 240, font(22, True), OK if mb["accepted"] else WARN)
        return img

    # ---------- montagem ----------
    def render(self, out):
        cfg = self.cfg
        wr = Writer(out, cfg.get("voice", "pt-BR-AntonioNeural"))
        n = max(4 * FPS, wr.speak(cfg.get("intro")))
        img, d = self.base("introdução")
        d.text((160, 300), "Um robô, um modelo de linguagem, e um crítico", font=font(50, True), fill=INK)
        self.para(d, 160, 400, "O GPT-6 Astra controla o braço do G1 olhando só as câmeras. Quando falha, um segundo modelo "
                  "analisa o episódio com a verdade do simulador e reescreve regras gerais para o prompt da próxima tentativa. "
                  "Não há treino de pesos: o que muda é o que o robô lê.", 1500, font(28), INK)
        wr.fade_in(img, n)
        rounds = cfg["rounds"]
        for k, rd in enumerate(rounds):
            ep = self.root / rd["episode"] if not Path(rd["episode"]).is_absolute() else Path(rd["episode"])
            Story(ep).play(wr, title=rd.get("title"))
            if k > 0:
                prev = self.root / rounds[k - 1]["episode"]
                img = self.validation(prev, ep, f"validação · {rounds[k-1].get('name')} vs {rd.get('name')}",
                                      (rounds[k - 1].get("name", "antes"), rd.get("name", "depois")))
                wr.fade_in(img, max(7 * FPS, wr.speak(rd.get("validation_text"))))
            if k + 1 < len(rounds):
                refl = json.loads((ep / "reflection.json").read_text())
                lab = f"crítico · depois do {rd.get('name', ep.parent.name)}"
                texts = rd.get("critic_text", ["", "", ""])
                for scene, txt in zip((self.critic_saw(ep, refl, lab), self.critic_diag(refl, lab),
                                       self.critic_change(refl, lab)), texts):
                    wr.fade_in(scene, max(7 * FPS, wr.speak(txt)))
        if cfg.get("outro"):
            img, d = self.base("conclusão")
            d.text((160, 280), cfg.get("outro_title", "O que aprendemos"), font=font(50, True), fill=INK)
            self.para(d, 160, 380, cfg.get("outro_screen", cfg["outro"]), 1500, font(28), INK)
            wr.fade_in(img, max(6 * FPS, wr.speak(cfg["outro"])))
        wr.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Learning(a.config).render(Path(a.out))
    print(a.out)
