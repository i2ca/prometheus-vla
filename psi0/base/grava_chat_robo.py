"""Grava o app do orquestrador (chat_local :8090) e a câmera do robô (sim_hibrido :8782/quadro) ao mesmo tempo, lado a lado.

Abre o chat num Chromium sem janela, manda o objetivo, tira foto da página e do robô ~4x por segundo até o orquestrador
encerrar (objetivo concluído, finish ou parada) e monta o vídeo com ffmpeg.
Uso (python com playwright): python grava_chat_robo.py --out <pasta> [--objetivo "..."] [--max-s 480]
"""
import argparse, io, subprocess, threading, time, urllib.request
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from playwright.sync_api import sync_playwright

LOG = Path("/tmp/prometheus_chat/orquestrador.log")
FIM = ("status=goal_completed", "Plano Gerado (Tool): finish", "Parei")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--objetivo", default="Pegue a caneca branca e coloque-a sob o coador de café.")
    ap.add_argument("--max-s", type=float, default=480); ap.add_argument("--fps", type=float, default=4); a = ap.parse_args()
    (a.out / "app").mkdir(parents=True, exist_ok=True); (a.out / "robo").mkdir(exist_ok=True)
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8782/reset_cena", data=b"{}", method="POST"), timeout=10).read()
    n0 = len(LOG.read_text().splitlines())
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1000, "height": 900}); pg.goto("http://127.0.0.1:8090")
        pg.get_by_role("button", name="Nova conversa").click(); time.sleep(1)
        caixa = pg.get_by_placeholder("Descreva a tarefa…"); caixa.fill(a.objetivo); caixa.press("Enter")
        t0, i, fim_em = time.time(), 0, None; T_APP, T_ROBO, parar = [], [], threading.Event()

        fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        ER1 = Path("/tmp/prometheus_chat/er1_ultimo.jpg"); er1_t0 = ER1.stat().st_mtime if ER1.exists() else 0

        def pega(rota):
            try: return Image.open(io.BytesIO(urllib.request.urlopen("http://127.0.0.1:8782" + rota, timeout=2).read())).convert("RGB").resize((640, 480))
            except Exception: return None

        def robo():   # grade 2x2 a ~10 Hz: cabeça | o que o ER-1 marcou / punho esquerdo | punho direito
            j, ultimo = 0, {}
            while not parar.is_set():
                t = time.time(); g = Image.new("RGB", (1280, 960), (30, 30, 30)); d = ImageDraw.Draw(g)
                for rota, pos, nome in (("/quadro", (0, 0), "câmera da cabeça"), ("/quadro/left_wrist", (0, 480), "punho esquerdo"),
                                        ("/quadro/right_wrist", (640, 480), "punho direito")):
                    im = pega(rota) or ultimo.get(rota)
                    if im: ultimo[rota] = im; g.paste(im, pos)
                    d.text((pos[0] + 10, pos[1] + 8), nome, font=fonte, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
                if ER1.exists() and ER1.stat().st_mtime > er1_t0:
                    try: g.paste(Image.open(ER1).convert("RGB").resize((640, 480)), (640, 0)); rot = "o que o ER-1 marcou " + time.strftime("%H:%M:%S", time.localtime(ER1.stat().st_mtime))
                    except Exception: rot = "ER-1"
                else: rot = "ER-1: ainda não chamado"
                d.text((650, 8), rot, font=fonte, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
                g.save(a.out / "robo" / f"{j:05d}.jpg", quality=88); T_ROBO.append(t - t0); j += 1
                time.sleep(max(0, 0.1 - (time.time() - t)))
        th = threading.Thread(target=robo, daemon=True); th.start()
        while time.time() - t0 < a.max_s:
            t = time.time()
            for det in pg.locator("details:not([open])").all():   # abre os blocos de raciocínio para aparecerem no vídeo
                try: det.evaluate("e => e.open = true")
                except Exception: pass
            pg.evaluate("() => { const m = document.querySelector('main') || document.scrollingElement; m.scrollTop = m.scrollHeight; window.scrollTo(0, document.body.scrollHeight); }")
            T_APP.append(time.time() - t0); pg.screenshot(path=str(a.out / "app" / f"{i:05d}.png")); i += 1
            novo = "\n".join(LOG.read_text().splitlines()[n0:])
            if fim_em is None and (any(f in novo for f in FIM) or pg.get_by_text("Parei:").count() or pg.get_by_text("Encerrou o objetivo").count()):
                fim_em = time.time()
            if fim_em and time.time() - fim_em > 12: break
            time.sleep(max(0, 1 / a.fps - (time.time() - t)))
        b.close(); parar.set(); th.join()
    dur = time.time() - t0

    def lista(nome, pasta, ext, T):   # concat do ffmpeg: cada quadro dura até o próximo, no tempo real da gravação
        L = [f"file '{a.out / pasta / f'{k:05d}.{ext}'}'\nduration {(T[k + 1] if k + 1 < len(T) else dur) - T[k]:.3f}" for k in range(len(T))]
        (a.out / nome).write_text("\n".join(L) + f"\nfile '{a.out / pasta / f'{len(T) - 1:05d}.{ext}'}'\n")
    lista("app.txt", "app", "png", T_APP); lista("robo.txt", "robo", "jpg", T_ROBO)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(a.out / "app.txt"),
                    "-f", "concat", "-safe", "0", "-i", str(a.out / "robo.txt"),
                    "-filter_complex", f"[0]scale=-2:960,fps=10,tpad=start_duration={T_APP[0]:.2f}:start_mode=clone[a];[1]fps=10,tpad=start_duration={T_ROBO[0]:.2f}:start_mode=clone[r];[a][r]hstack=shortest=1,format=yuv420p",
                    "-c:v", "libx264", "-crf", "23", "-r", "20", str(a.out / "chat_e_robo.mp4")], check=True)
    print(f"{len(T_APP)} fotos do chat, {len(T_ROBO)} quadros do robô em {dur:.0f} s -> {a.out / 'chat_e_robo.mp4'}")


if __name__ == "__main__":
    main()
