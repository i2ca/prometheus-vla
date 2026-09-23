"""Video narrado de um episodio Direct: o que o robo ve, o que declara, o que decide e o que acontece.

Uso: .venv/bin/python scripts/make_direct_story.py results/direct-astra-01/x0.40_y-0.20 [--out story.mp4]

Tudo sai dos arquivos do episodio (policy-calls/, actions/, obs/, report.json). O modelo nao expoe
raciocinio oculto: o "raciocinio" mostrado e o campo `reason` que ele escreveu na tool call.
Se existir <ep>/narration.json ({voice, intro, steps: {"1": ...}, outro}), gera narracao com edge-tts
e estica cada trecho do video para caber na fala.
Custo: preco de tabela da API (PRICES), porque as chamadas passam pela assinatura do Codex via OmniRoute.
"""
import argparse, asyncio, json, subprocess, tempfile, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
PANEL_X, HEAD_H = 1280, 64
GRID_Y, GRID_W, GRID_H = HEAD_H, 1280, 480
SUB_Y, SUB_H = GRID_Y + GRID_H, 108
IN_Y = SUB_Y + SUB_H
THINK_S, AFTER_S, INTRO_S, OUTRO_S = 3.0, 0.6, 4.0, 6.0
SETTLE_FRAMES, SR = 15, 24000
# US$ por milhao de tokens (entrada, saida), preco de tabela da API em 09/2026
PRICES = {"gpt-6-astra": (10.0, 50.0), "gpt-5.6-sol": (4.0, 20.0)}

BG, PANEL, CARD, INK, MUTED, LINE = (13, 15, 20), (22, 25, 33), (30, 34, 45), (236, 238, 243), (146, 153, 168), (50, 55, 68)
ACCENT, OK, BAD, WARN, CUP = (96, 165, 250), (74, 222, 128), (248, 113, 113), (251, 191, 36), (230, 230, 230)
F = "/usr/share/fonts/truetype/noto/NotoSans-{}.ttf"
M = "/usr/share/fonts/truetype/noto/NotoSansMono-{}.ttf"


def font(size, bold=False, mono=False):
    return ImageFont.truetype((M if mono else F).format("Bold" if bold else "Regular"), size)


FT, FH, FB, FS, FXS, FMONO, FSUB = font(22, True), font(16, True), font(17), font(14), font(12), font(15, mono=True), font(24)
ARM = ["ombro pitch", "ombro roll", "ombro yaw", "cotovelo", "punho roll", "punho pitch", "punho yaw"]


def price_of(model):
    for k, v in PRICES.items():
        if k in (model or ""):
            return v
    return None


def wrap(draw, text, fnt, width):
    out = []
    for para in (text or "").split("\n"):
        line = ""
        for word in para.split():
            test = (line + " " + word).strip()
            if draw.textlength(test, font=fnt) <= width:
                line = test
            else:
                out.append(line); line = word
        out.append(line)
    return out


def chip(draw, x, y, text, color, fnt=FH):
    w = draw.textlength(text, font=fnt) + 22
    draw.rounded_rectangle((x, y, x + w, y + 28), 14, fill=color)
    draw.text((x + 11, y + 4), text, font=fnt, fill=(12, 12, 16))
    return x + w + 10


def grid_of(path):
    im = Image.open(path).convert("RGB")
    return fit(im.crop((0, 32, im.width, im.height)), GRID_W, GRID_H)


def fit(img, w, h):
    im = img.copy(); im.thumbnail((w, h), Image.LANCZOS); return im


# ---------------- audio ----------------
async def _tts(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate="+4%").save(str(path))


def tts_pcm(text, voice, tmp):
    mp3 = Path(tmp) / f"{abs(hash(text))}.mp3"
    asyncio.run(_tts(text, voice, mp3))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp3), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


class Story:
    def __init__(self, ep):
        self.ep = ep
        self.meta = json.loads((ep / "episode.json").read_text())
        self.task = self.meta["case"]["instruction"]
        self.model = self.meta.get("policy_model", "?")
        self.calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
        self.frames = sorted((ep / "frames").glob("*.jpg"))
        self.report = json.loads((ep / "report.json").read_text()) if (ep / "report.json").exists() else {}
        nar = ep / "narration.json"
        self.narration = json.loads(nar.read_text()) if nar.exists() else None
        start = SETTLE_FRAMES
        tin = tout = 0
        price = price_of(self.calls[0].get("model") if self.calls else "")
        self.price = price
        for c in self.calls:
            s = ep / "actions" / f"samples-{c['step_id'] + 1:03d}.json"
            samples = json.loads(s.read_text()) if s.exists() and c.get("tool") == "actuate" else []
            c["_frames"], c["_samples"] = self.frames[start:start + len(samples)], samples
            start += len(samples)
            u = c.get("usage") or {}
            tin += u.get("input_tokens", 0); tout += u.get("output_tokens", 0)
            c["_tin"], c["_tout"] = tin, tout
            c["_usd"] = (tin * price[0] + tout * price[1]) / 1e6 if price else None
        # trajetoria da palma (medida) e caneca (verdade, so para o espectador)
        self.palm_path = [c["observed"]["palm"]["position"] for c in self.calls if "observed" in c]
        self.cup0 = self.meta["case"]["cup_xy"]

    # ---------------- blocos ----------------
    def header(self, d, current):
        d.rectangle((0, 0, W, HEAD_H), fill=(9, 10, 14))
        d.text((24, 12), "G1 · política Direct", font=font(24, True), fill=INK)
        d.text((24 + d.textlength("G1 · política Direct", font=font(24, True)) + 16, 20),
               f"{self.model.replace(' via OmniRoute', '')}  ·  episódio {self.ep.parent.name}/{self.ep.name}", font=FS, fill=MUTED)
        x = W - 24 - len(self.calls) * 46
        d.text((x - 70, 22), "steps", font=FS, fill=MUTED)
        for c in self.calls:
            st = (c.get("result") or {}).get("status")
            col = BAD if st in ("rejected", "aborted_on_contact") else ACCENT if c.get("tool") == "finish_episode" else OK
            done = current is not None and c["call"] < current
            now = current == c["call"]
            fill = col if (done or now) else CARD
            d.rounded_rectangle((x, 16, x + 38, 48), 8, fill=fill, outline=INK if now else None, width=2)
            d.text((x + 19 - d.textlength(str(c["call"]), font=FH) / 2, 21), str(c["call"]), font=FH,
                   fill=(12, 12, 16) if (done or now) else MUTED)
            x += 46

    def subtitle(self, d, text):
        d.rectangle((0, SUB_Y, PANEL_X, SUB_Y + SUB_H), fill=(9, 10, 14))
        lines = wrap(d, text or "", FSUB, PANEL_X - 80)[:2]
        y = SUB_Y + (SUB_H - len(lines) * 32) // 2
        for ln in lines:
            d.text(((PANEL_X - d.textlength(ln, font=FSUB)) / 2, y), ln, font=FSUB, fill=INK); y += 32

    def inputs(self, img, d, call):
        d.rectangle((0, IN_Y, PANEL_X, H), fill=BG)
        d.text((24, IN_Y + 10), "O QUE O ROBÔ VÊ", font=FH, fill=ACCENT)
        d.text((24 + d.textlength("O QUE O ROBÔ VÊ", font=FH) + 12, IN_Y + 12),
               f"as 3 imagens enviadas ao modelo no step {call['call']}", font=FS, fill=MUTED)
        step_dir = self.ep / "obs" / f"step-{call['step_id']:03d}"
        y0 = IN_Y + 38
        h = fit(Image.open(step_dir / "head_camera.png").convert("RGB"), 640, 360)
        img.paste(h, (24, y0)); d.text((30, y0 + h.height - 20), "cabeça", font=FS, fill=INK)
        y = y0
        for cam, lab in (("left_wrist_camera", "punho esq."), ("right_wrist_camera", "punho dir.")):
            im = fit(Image.open(step_dir / f"{cam}.png").convert("RGB"), 310, 176)
            img.paste(im, (680, y)); d.text((686, y + im.height - 20), lab, font=FS, fill=INK)
            y += im.height + 8
        obs = call.get("observed", {})
        x0, yj = 1010, y0
        d.text((x0, yj), "juntas medidas (°)", font=FS, fill=MUTED); yj += 22
        for name, q in zip(ARM, obs.get("arm_joints_rad", [])):
            deg = float(np.degrees(q))
            d.text((x0, yj), f"{name:<11}{deg:6.1f}", font=font(14, mono=True), fill=INK)
            mid = x0 + 130
            d.line((x0, yj + 19, x0 + 250, yj + 19), fill=LINE)
            d.rectangle((min(mid, mid + deg), yj + 17, max(mid, mid + deg), yj + 21), fill=ACCENT)
            yj += 28
        if "waist_yaw_rad" in obs:
            d.text((x0, yj), f"{'cintura':<11}{np.degrees(obs['waist_yaw_rad']):6.1f}", font=font(14, mono=True), fill=INK); yj += 28
        if "gripper_closure" in obs:
            c = obs["gripper_closure"]
            d.text((x0, yj), f"{'mão fechada':<11}{c*100:5.0f}%", font=font(14, mono=True), fill=INK)
            d.rectangle((x0, yj + 20, x0 + 250, yj + 24), fill=LINE)
            d.rectangle((x0, yj + 20, x0 + int(c * 250), yj + 24), fill=WARN)

    # cor do cartao por secao (contorno, rotulo e fundo levemente tingido)
    TONES = {"TAREFA": (148, 155, 170), "O QUE ELE VÊ": (96, 165, 250), "RACIOCÍNIO DECLARADO": (192, 132, 252),
             "DECISÃO": (251, 191, 36), "RESULTADO": (74, 222, 128), "VERDADE": (251, 146, 60)}

    def section(self, d, y, label, text, color=INK, fnt=FB, max_lines=6, reveal=1.0, tone=None):
        tone = tone or next((v for k, v in self.TONES.items() if label.startswith(k)), ACCENT)
        x0, x1 = PANEL_X + 18, W - 18
        pad, width = 14, x1 - x0 - 28
        lines = wrap(d, text, fnt, width)
        shown_lines = lines[:max_lines]
        h = 12 + 22 + len(shown_lines) * (fnt.size + 6) + 10
        fill = tuple(int(PANEL[i] * 0.82 + tone[i] * 0.10) for i in range(3))
        d.rounded_rectangle((x0, y, x1, y + h), 10, fill=fill, outline=tone, width=2)
        d.rectangle((x0 + 2, y + 10, x0 + 6, y + h - 10), fill=tone)
        d.text((x0 + pad, y + 10), label, font=FH, fill=tone)
        ty = y + 12 + 22
        total = sum(len(l) + 1 for l in lines)
        budget = int(total * reveal)
        for i, ln in enumerate(shown_lines):
            if budget <= 0:
                break
            part = ln[:budget]; budget -= len(ln) + 1
            if i == max_lines - 1 and len(lines) > max_lines:
                part = part.rstrip() + " …"
            d.text((x0 + pad, ty), part, font=fnt, fill=color)
            ty += fnt.size + 6
        return y + h + 10

    def panel(self, d, call, phase, reveal=1.0, progress=0.0, sample=None):
        d.rectangle((PANEL_X, HEAD_H, W, H), fill=PANEL)
        x = PANEL_X + 26
        y = HEAD_H + 18
        d.text((x, y), f"STEP {call['call']} de {len(self.calls)}", font=FT, fill=INK)
        d.text((x + 170, y + 6), f"t = {call.get('sim_time_s', 0):.2f} s simulados", font=FS, fill=MUTED)
        y += 38
        res = call.get("result") or {}
        st = res.get("status", "")
        if phase == "think":
            cx = chip(d, x, y, "DECIDINDO", WARN)
        elif st == "rejected":
            cx = chip(d, x, y, "REJEITADO", BAD)
        elif st == "aborted_on_contact":
            cx = chip(d, x, y, "ABORTADO", BAD)
        elif call.get("tool") == "finish_episode":
            cx = chip(d, x, y, "ENCERROU", ACCENT)
        else:
            cx = chip(d, x, y, "EXECUTANDO", OK)
        if call.get("latency_s"):
            d.text((cx + 2, y + 6), f"resposta em {call['latency_s']:.1f} s  ·  {(call.get('sent_at') or '')[11:19]} UTC",
                   font=FS, fill=MUTED)
        y += 44
        y = self.section(d, y, "TAREFA", self.task, color=INK, fnt=FS, max_lines=2)
        args = call.get("arguments", {})
        if call.get("what_i_see"):
            y = self.section(d, y, "O QUE ELE VÊ", call["what_i_see"], fnt=FS, max_lines=6, reveal=min(1, reveal * 1.6))
        y = self.section(d, y, "RACIOCÍNIO DECLARADO", args.get("reason", call.get("error", "")),
                         max_lines=5, reveal=max(0, min(1, reveal * 1.6 - 0.6)))
        if reveal >= 0.999:
            if call.get("tool") == "actuate":
                before = call.get("observed", {}).get("palm", {}).get("position")
                delta = (np.array(args["position"]) - np.array(before)) * 100 if before else None
                dec = (f"actuate · dedos {args.get('gripper')} · {args.get('steps')} steps ({args.get('steps', 0) * 0.2:.1f} s)\n"
                       + (f"palma Δ = [{', '.join(f'{v:+.1f}' for v in delta)}] cm  (frente, esq., cima)" if delta is not None else ""))
            elif call.get("tool") == "finish_episode":
                dec = f"finish_episode · outcome = {args.get('outcome')}"
            else:
                dec = call.get("error", "sem decisão")
            y = self.section(d, y, "DECISÃO", dec, fnt=font(15, mono=True), max_lines=3)
        if phase == "act":
            if st == "rejected":
                y = self.section(d, y, "RESULTADO", "; ".join(res.get("errors", [])), color=BAD, fnt=FS, max_lines=3, tone=BAD)
            elif st == "executed":
                y = self.section(d, y, "RESULTADO", f"executado · erro final da palma {res.get('position_gap_m', 0)*1000:.1f} mm",
                                 color=OK, fnt=FS, max_lines=1)
            if sample:
                fingers = ", ".join(sample["fingers"]) or "nenhum"
                live = (f"dedos tocando a caneca: {fingers}\n"
                        f"caneca: altura {sample['cup'][2]*100:.1f} cm · inclinação {sample['cup_tilt_deg']:.0f}°")
                y = self.section(d, y, "VERDADE DO SIMULADOR · o modelo não vê",
                                 live, color=WARN if sample["fingers"] else INK, fnt=FS, max_lines=2)
        self.cost_box(d, call, reveal)
        self.minimap(d, call, sample)
        if phase == "act":
            d.rectangle((PANEL_X, H - 6, W, H), fill=LINE)
            d.rectangle((PANEL_X, H - 6, PANEL_X + int((W - PANEL_X) * progress), H), fill=OK)

    def cost_box(self, d, call, reveal):
        x0, y0, x1, y1 = PANEL_X + 26, H - 250, PANEL_X + 300, H - 22
        d.rounded_rectangle((x0, y0, x1, y1), 10, fill=CARD)
        d.text((x0 + 14, y0 + 10), "CUSTO ACUMULADO", font=FH, fill=ACCENT)
        prev = self.calls[call["call"] - 2] if call["call"] > 1 else {"_tin": 0, "_tout": 0, "_usd": 0}
        k = 1.0 if reveal >= 0.999 else 0.0
        tin = prev["_tin"] + k * (call["_tin"] - prev["_tin"])
        tout = prev["_tout"] + k * (call["_tout"] - prev["_tout"])
        usd = (prev["_usd"] or 0) + k * ((call["_usd"] or 0) - (prev["_usd"] or 0))
        d.text((x0 + 14, y0 + 38), f"US$ {usd:.4f}", font=font(32, True), fill=INK)
        d.text((x0 + 14, y0 + 84), f"entrada  {int(tin):>7,} tok".replace(",", "."), font=font(15, mono=True), fill=INK)
        d.text((x0 + 14, y0 + 106), f"saída    {int(tout):>7,} tok".replace(",", "."), font=font(15, mono=True), fill=INK)
        lat = sum(c.get("latency_s") or 0 for c in self.calls[:call["call"] - 1 + int(k)])
        d.text((x0 + 14, y0 + 128), f"espera   {lat:>7.1f} s", font=font(15, mono=True), fill=INK)
        d.text((x0 + 14, y0 + 150), f"chamadas {call['call'] - 1 + int(k):>3} / {self.meta['max_calls']}", font=font(15, mono=True), fill=INK)
        if self.price:
            d.text((x0 + 14, y0 + 182), f"tabela API: US$ {self.price[0]:g} / {self.price[1]:g} por Mtok", font=FXS, fill=MUTED)
            d.text((x0 + 14, y0 + 198), "equivalente; rodou via assinatura Codex", font=FXS, fill=MUTED)

    def minimap(self, d, call, sample):
        """Vista de cima: x (frente) para cima, y (esquerda) para a esquerda."""
        x0, y0, x1, y1 = PANEL_X + 314, H - 250, W - 26, H - 22
        d.rounded_rectangle((x0, y0, x1, y1), 10, fill=CARD)
        d.text((x0 + 14, y0 + 10), "VISTA DE CIMA", font=FH, fill=ACCENT)
        xs, ys = (0.12, 0.48), (-0.34, -0.06)
        px0, py0, px1, py1 = x0 + 20, y0 + 40, x1 - 20, y1 - 30
        to = lambda p: (px0 + (ys[1] - p[1]) / (ys[1] - ys[0]) * (px1 - px0),
                        py1 - (p[0] - xs[0]) / (xs[1] - xs[0]) * (py1 - py0))
        for gx in np.arange(0.15, 0.48, 0.05):
            a, b = to((gx, ys[0])), to((gx, ys[1])); d.line((a[0], a[1], b[0], b[1]), fill=LINE)
        cup = sample["cup"] if sample else self.cup0
        cx, cy = to(cup)
        r = 0.04 / (ys[1] - ys[0]) * (px1 - px0)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=CUP, width=2)
        d.text((cx + r + 4, cy - 8), "caneca", font=FXS, fill=CUP)
        path = [p for p in self.palm_path[:call["call"]]]
        if sample:
            path = path + [sample["palm"]]
        pts = [to(p) for p in path]
        if len(pts) > 1:
            d.line(pts, fill=ACCENT, width=3)
        for p in pts:
            d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=ACCENT)
        if pts:
            p = pts[-1]; d.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), outline=INK, width=2)
        ly = y1 - 20
        d.ellipse((x0 + 14, ly + 3, x0 + 22, ly + 11), fill=ACCENT)
        d.text((x0 + 28, ly), "palma (medida)", font=FXS, fill=MUTED)
        d.ellipse((x0 + 128, ly + 2, x0 + 138, ly + 12), outline=CUP, width=2)
        d.text((x0 + 144, ly), "caneca (verdade)", font=FXS, fill=MUTED)
        d.polygon([(x0 + 262, ly + 12), (x0 + 267, ly + 2), (x0 + 272, ly + 12)], fill=MUTED)
        d.text((x0 + 278, ly), "frente", font=FXS, fill=MUTED)

    def frame(self, grid, call, phase, sub, **kw):
        img = Image.new("RGB", (W, H), BG)
        img.paste(grid, (0, GRID_Y))
        if phase == "think":
            img.paste(Image.blend(grid, Image.new("RGB", grid.size, (0, 0, 0)), 0.35), (0, GRID_Y))
        d = ImageDraw.Draw(img)
        self.header(d, call["call"])
        self.subtitle(d, sub)
        self.inputs(img, d, call)
        self.panel(d, call, phase, **kw)
        return img

    def card(self, title, lines, color=INK, sub=None, current=None):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        self.header(d, current)
        d.text((160, 250), title, font=font(52, True), fill=color)
        y = 350
        for ln in lines:
            fnt = font(26) if not ln.startswith("·") else font(22)
            for part in wrap(d, ln.lstrip("· "), fnt, W - 320):
                d.text((160, y), part, font=fnt, fill=INK if not ln.startswith("·") else MUTED); y += fnt.size + 14
            y += 12
        if sub:
            lines = wrap(d, sub, FSUB, W - 320)[:3]
            yy = H - 60 - len(lines) * 34
            for ln in lines:
                d.text(((W - d.textlength(ln, font=FSUB)) / 2, yy), ln, font=FSUB, fill=MUTED); yy += 34
        return img

    # ---------------- render ----------------
    def play(self, wr, title=None):
        """Toca o episodio inteiro no Writer (intro, steps, veredito)."""
        put, nar = wr.put, self.narration or {}
        case = self.meta["case"]
        n = max(int(INTRO_S * FPS), wr.speak(nar.get("intro")))
        intro = self.card(title or "O robô decide sozinho, olhando", [
            f"Modelo no controle: {self.model.replace(' via OmniRoute', '')} (chamado direto, via OmniRoute)",
            f"Tarefa: {self.task}",
            "· Entrada: 3 câmeras RGB + pose medida da palma e juntas. A posição da caneca não é informada.",
            "· Saída: um alvo para a palma (até 5 cm e 0,35 rad por step) e o comando dos dedos. IK e PD executam.",
            f"· Caneca em x = {case['cup_xy'][0]:.2f} m, y = {case['cup_xy'][1]:.2f} m. Orçamento: {self.meta['max_calls']} chamadas."
            + ("  · Prompt com lições do crítico." if self.meta.get("lessons_file") else "")])
        wr.fade_in(intro, n)
        last_grid = grid_of(self.frames[SETTLE_FRAMES - 1])
        for call in self.calls:
            text = nar.get("steps", {}).get(str(call["call"]), "")
            need = wr.speak(text)
            frames = call["_frames"]
            act_n = len(frames) + int(AFTER_S * FPS) if frames else int(2.0 * FPS)
            think_n = max(int(THINK_S * FPS), need - act_n)
            reveal_n = int(THINK_S * FPS * 0.8)
            for i in range(think_n):
                put(self.frame(last_grid, call, "think", text, reveal=min(1.0, (i + 1) / reveal_n)))
            if not frames:
                put(self.frame(last_grid, call, "act", text, progress=1.0), act_n)
                continue
            sample = None
            for k, fp in enumerate(frames):
                last_grid = grid_of(fp)
                sample = call["_samples"][k] if k < len(call["_samples"]) else sample
                put(self.frame(last_grid, call, "act", text, progress=(k + 1) / len(frames), sample=sample))
            put(self.frame(last_grid, call, "act", text, progress=1.0, sample=sample), int(AFTER_S * FPS))
        r = self.report
        ok = r.get("accepted")
        last = self.calls[-1]
        ab = r.get("aborted") or {}
        why = (" · abortado por autocolisão" if ab.get("self_collision") else
               " · abortado por mão na mesa" if ab.get("hand_table") else "")
        n = max(int(OUTRO_S * FPS), wr.speak(nar.get("outro")))
        put(self.card("ACEITO" if ok else "REPROVADO", [
            f"Caneca levantada {r.get('lift_m', 0)*100:.1f} cm (mínimo 4,8) · retida 2 s: {'sim' if r.get('retained_for_2s') else 'não'} · "
            f"em pé: {'sim' if r.get('upright_grasp') else 'não'} (inclinação máx. antes do fecho {r.get('cup_tilt_deg_before_close', 0):.0f}°){why}",
            f"{len(self.calls)} chamadas ao modelo · {r.get('rejected_calls', 0)} rejeitadas · {r.get('sim_seconds')} s simulados · "
            f"{sum(c.get('latency_s') or 0 for c in self.calls):.0f} s esperando o modelo",
            f"{last['_tin']:,} tokens de entrada · {last['_tout']:,} de saída · US$ {last['_usd'] or 0:.3f} em preço de API".replace(",", "."),
            "· Avaliação do simulador com os mesmos critérios do controlador scriptado: caneca acima de 4,8 cm por 2 s, "
            "2 dedos em contato, inclinação abaixo de 15°, sem tocar a mesa."],
            color=OK if ok else BAD, current=len(self.calls) + 1), n)

    def render(self, out):
        wr = Writer(out, (self.narration or {}).get("voice", "pt-BR-AntonioNeural"))
        self.play(wr)
        wr.close()


class Writer:
    """Pipe de video para o ffmpeg + trilha de narracao alinhada por quadro."""

    def __init__(self, out, voice="pt-BR-AntonioNeural"):
        self.out, self.voice, self.tmp = Path(out), voice, tempfile.mkdtemp()
        self.audio, self.cursor = [], 0
        self.silent = self.out.with_suffix(".silent.mp4")
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-preset", "medium", "-crf", "19", str(self.silent)], stdin=subprocess.PIPE)

    def put(self, im, n=1):
        b = np.asarray(im, np.uint8).tobytes()
        for _ in range(n):
            self.ff.stdin.write(b)
        self.cursor += n

    def fade_in(self, im, n):
        for i in range(n):
            self.put(Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), im, min(1.0, i / 12)))

    def speak(self, text):
        """Agenda a fala a partir do quadro atual; devolve quantos quadros ela ocupa."""
        if not text:
            return 0
        pcm = tts_pcm(text, self.voice, self.tmp)
        self.audio.append((self.cursor + int(0.3 * FPS), pcm))
        return int(len(pcm) / SR * FPS) + int(0.6 * FPS)

    def close(self):
        self.ff.stdin.close(); self.ff.wait()
        if not self.audio:
            self.silent.rename(self.out); return
        total = np.zeros(int(self.cursor / FPS * SR) + SR, np.int32)
        for start, pcm in self.audio:
            s = int(start / FPS * SR)
            total[s:s + len(pcm)] += pcm[: max(0, len(total) - s)]
        wav = Path(self.tmp) / "voice.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(np.clip(total, -32768, 32767).astype(np.int16).tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.silent), "-i", str(wav),
                        "-af", "loudnorm=I=-16:TP=-1.5", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                        "-shortest", str(self.out)], check=True)
        self.silent.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    ep = Path(a.episode_dir)
    out = Path(a.out) if a.out else ep / "direct-story.mp4"
    Story(ep).render(out)
    print(out)
