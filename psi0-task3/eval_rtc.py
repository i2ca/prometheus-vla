"""Psi0 task3 em malha aberta com o controlador RTC do servidor oficial (serve_psi0_amo.RealTimeChunkController) simulado passo a passo:
o 1o bloco sai de predict_action; depois, a cada s passos (s >= 15), um bloco novo com o resto do anterior como prefixo e atraso d,
rpy/altura do estado trocados pela ação anterior (A_cur[s-2]). Observações = episódio gravado (imagem egocêntrica + estado).
Grava a ação executada a cada passo, os blocos, latências e timestamps num npz. Uso: python eval_rtc.py <saida> <d> <s> <modo auto|gravado> <ep>...
Modo gravado: o prefixo RTC vem das ações gravadas (o que o robô estaria executando se seguisse a demo), não das previsões."""
import os
import sys, json, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch, av
from PIL import Image
sys.argv, args = sys.argv[:1], sys.argv[1:]
from psi.deploy.serve_psi0_amo import Server
from psi.utils import pad_to_len
D = Path(os.environ["PSI0_DATA"])
out, d, s_exec, modo, eps = Path(args[0]), int(args[1]), int(args[2]), args[3], list(map(int, args[4:])); out.mkdir(parents=True, exist_ok=True)
srv = Server(policy="psi", run_dir=Path(os.environ["PSI0_CKPT"]), ckpt_step=40000, device="cuda:0", enable_rtc=True)
lo, hi = np.array(srv.maxmin.action_min, np.float32), np.array(srv.maxmin.action_max, np.float32)
I = "g1/pick_bottle_and_turn_and_pour_into_cup"  # o que o cliente oficial manda (deploy_psi0-rtc.sh), em minúsculas pelo servidor
M = srv.model

def obs(S_t, rpy, h):
    o = pad_to_len(np.concatenate([S_t[:28], rpy, h]).astype(np.float32), 36, dim=0)[0]
    return torch.from_numpy(srv.maxmin.normalize_state_func(o)[None, None].astype(np.float32)).to(srv.device)

for ep in eps:
    df = pd.read_parquet(D / f"data/chunk-000/episode_{ep:06d}.parquet"); S = np.stack(df.states.values); A = np.stack(df.action.values)
    with av.open(str(D / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4")) as c: F = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n = min(len(S), len(F)); img = lambda t: [srv.preprocess_image({"cam0": Image.fromarray(F[t])})["cam0"]]
    log = []; executed = np.zeros((n, 36), np.float32); prefixo = np.zeros(n, bool)
    def infer(t, prev=None):
        w = dt.datetime.now().isoformat(timespec="milliseconds"); t0 = time.perf_counter()
        rpy, h = (np.zeros(3, np.float32), np.array([0.75], np.float32)) if prev is None else (prev[1][28:31], prev[1][31:32])
        with torch.inference_mode():
            if prev is None:
                a = M.predict_action(observations=img(t), states=obs(S[t], rpy, h), traj2ds=None, instructions=[I], num_inference_steps=8)
            else:
                a = M.predict_action_with_training_rtc_flow(observations=img(t), states=obs(S[t], rpy, h), traj2ds=None, instructions=[I],
                        num_inference_steps=8, prev_actions=torch.from_numpy(prev[0][None]).to(srv.device), inference_delay=d, max_delay=8)
        a = a[0].float().cpu().numpy(); torch.cuda.synchronize()
        log.append({"t": t, "wall": w, "lat_s": round(time.perf_counter() - t0, 4)}); return a
    A_cur = infer(0); k = 0; chunks = [(0, A_cur.copy())]
    for t in range(n):
        if k >= s_exec and t < n - 1:   # replaneja: ações já executadas saem, o resto vira prefixo
            A_prev = np.concatenate([A_cur[k:], np.zeros((k, A_cur.shape[1]), A_cur.dtype)])
            if modo == "gravado":
                g = A[t:t + d]; g = np.concatenate([g, np.repeat(g[-1:], d - len(g), 0)])   # só as d ações que estariam na fila
                A_prev = np.zeros_like(A_prev); A_prev[:d] = ((g - lo) / np.maximum(hi - lo, 1e-8) * 2 - 1).astype(np.float32)
            A_new = infer(t, (A_prev, A_cur[k - 2]))
            A_cur, k = A_new, 0; chunks.append((t, A_cur.copy()))
        executed[t] = srv._postprocess_action(A_cur[min(k, len(A_cur) - 1)]); prefixo[t] = len(chunks) > 1 and k < d; k += 1
    err = np.abs(executed - A[:n])[~prefixo]   # só passos previstos de fato (fora do prefixo RTC)
    np.savez_compressed(out / f"ep{ep:03d}_d{d}_s{s_exec}_{modo}.npz", executed=executed, prefixo=prefixo, gt_action=A[:n], states=S[:n], timestamp=df.timestamp.values[:n],
                        chunk_t=np.array([c[0] for c in chunks]), chunks=np.stack([srv._postprocess_action(c[1]) for c in chunks]),
                        log=json.dumps(log), instruction=I, d=d, s=s_exec)
    hold = float(np.mean([np.abs(A[t0 + d:t0 + s_exec, :28] - A[t0 + d - 1, :28]).mean() for t0 in range(0, n - s_exec, s_exec)]))  # repetir a última ação conhecida
    print(json.dumps({"ep": ep, "d": d, "s": s_exec, "modo": modo, "frames": n, "chunks": len(chunks), "lat_med": float(np.median([l["lat_s"] for l in log[1:]])),
                      "mae_maos": round(float(err[:, :14].mean()), 3), "mae_bracos": round(float(err[:, 14:28].mean()), 3),
                      "mae_base": round(float(err[:, 28:].mean()), 3), "ref_segurar": round(hold, 3)}), flush=True)
