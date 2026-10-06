"""Vídeo 1920x1080 da avaliação em malha aberta do Psi0 (task3) no estilo escuro do vídeo da Jacobiana:
câmera egocêntrica gravada à esquerda, curvas Psi0 (roxo) x teleoperação gravada (cinza) à direita, cursor no tempo.
Uso: python render_video.py <ep_gravado.npz> <ep_auto.npz> <episode.mp4> <saida.mp4>
"gravado": prefixo RTC = ações gravadas; "auto": prefixo RTC = previsões do próprio modelo sobre observações gravadas."""
import sys, json, subprocess
import numpy as np, av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz, npz_auto, mp4, out = sys.argv[1:5]
Ea = np.load(npz_auto)["executed"]
z = np.load(npz); E, G = z["executed"], z["gt_action"]; ts = z["timestamp"]; ct = z["chunk_t"]; pf = z["prefixo"]
Ep = E.copy(); Ep[pf] = np.nan   # passos de prefixo são cópia da execução gravada: não desenhar como previsão
erro = np.abs(E - G)[~pf, :28].mean()
hold = np.mean([np.abs(G[t0 + 6:t0 + 15, :28] - G[t0 + 5, :28]).mean() for t0 in ct[1:] if t0 + 15 <= len(G)])
log = json.loads(str(z["log"])); lat = np.median([l["lat_s"] for l in log[1:]])
with av.open(mp4) as c: F = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
n = min(len(E), len(F))

BG, FG, DIM, ROXO, CINZA = "#0b0b10", "#e8e6f0", "#6f6b80", "#a970ff", "#8a8a96"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM})
PAINEIS = [("cotovelo direito (rad)", 24), ("fechamento da mão direita (rad)", 11), ("ombro direito, pitch (rad)", 21), ("giro da base, alvo de yaw (rad)", 35)]

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
ax_img = fig.add_axes([0.03, 0.19, 0.50, 0.64]); ax_img.axis("off")
im = ax_img.imshow(F[0])
fig.text(0.03, 0.92, "Ψ₀ · Task 3", fontsize=34, weight="bold", color=FG)
fig.text(0.03, 0.875, "Pick bottle, turn and pour into cup  ·  checkpoint real oficial (ckpt 40000)", fontsize=17, color=DIM)
fig.text(0.03, 0.018, "Replay em malha aberta de um episódio GRAVADO de teleoperação: a câmera é a gravação, não o nosso robô. O Ψ₀ recebe as 6 ações já em execução (prefixo RTC) e prevê o resto.\n"
         f"Nos passos previstos, erro médio de {erro:.3f} rad em mãos e braços; repetir a última ação conhecida dá {hold:.3f}.".replace("0.0", "0,0") + "\n"
         "Realimentando a própria previsão (tracejado), ele trava. Replay offline não mostra ganho; só malha fechada decide.\n"
         f"Controlador RTC do servidor oficial simulado (replaneja a cada 15 passos, atraso 6). Inferência na A100: {lat*1000:.0f} ms por bloco de 30 ações.",
         fontsize=12.5, color=DIM, linespacing=1.5)
t_txt = fig.text(0.03, 0.84, "", fontsize=15, color=ROXO, family="DejaVu Sans Mono")
axs, cur = [], []
for i, (titulo, d) in enumerate(PAINEIS):
    a = fig.add_axes([0.58, 0.72 - i * 0.19, 0.39, 0.14], facecolor=BG)
    for s in a.spines.values(): s.set_color("#25232e")
    a.plot(ts[:n], G[:n, d], color=CINZA, lw=2.0, label="teleoperação gravada")
    a.plot(ts[:n], Ea[:n, d], color=ROXO, lw=1.4, ls="--", alpha=0.45, label="Ψ₀ realimentando a si mesmo")
    a.plot(ts[:n], Ep[:n, d], color=ROXO, lw=2.4, label="Ψ₀, passos previstos")
    a.set_xlim(ts[0], ts[n - 1]); a.set_title(titulo, loc="left", fontsize=14, color=FG, pad=6)
    a.tick_params(labelsize=10); a.grid(color="#1c1a24", lw=0.8)
    if i == 0: fig.legend(*a.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.555, 0.965), ncol=3, frameon=False, fontsize=11, labelcolor=FG)
    cur.append(a.axvline(ts[0], color=FG, lw=1.2, alpha=0.8)); axs.append(a)
for a in axs:
    for t0 in ct: a.axvline(ts[min(t0, n - 1)], color=ROXO, lw=0.5, alpha=0.12)

W, H = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
for t in range(n):
    im.set_data(F[t])
    for c_ in cur: c_.set_xdata([ts[t], ts[t]])
    t_txt.set_text(f"t = {ts[t]:5.2f} s   quadro {t:4d}/{n}")
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
ff.stdin.close(); ff.wait(); print("ok", out, n, "quadros")
