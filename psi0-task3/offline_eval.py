"""Avaliação em malha aberta do Psi0 (checkpoint real task3) nos episódios reais do dataset Pick_bottle_and_turn_and_pour_into_cup.
Reproduz o pré-processamento do servidor oficial (serve_psi0_amo.Server._parse_obs_payload): estado = mãos(14) + braços(14)
+ rpy zerado + altura 0,75, imagem egocêntrica 480x640 -> resize/crop 240x320, instrução em minúsculas.
A cada 30 quadros pede um bloco de 30 ações e compara com a ação gravada. Grava tudo em npz para re-renderizar depois.
Uso: python offline_eval.py <saida> <ep> [<ep> ...]"""
import os
import sys, time, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch, av
from PIL import Image
sys.argv, args = sys.argv[:1], sys.argv[1:]
from psi.deploy.serve_psi0_amo import Server
from psi.utils import pad_to_len

D = Path(os.environ["PSI0_DATA"])
out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
srv = Server(policy="psi", run_dir=Path(os.environ["PSI0_CKPT"]), ckpt_step=40000, device="cuda:0")
instr = json.loads((D / "meta/tasks.jsonl").read_text().splitlines()[0])["task"].lower()  # o cliente oficial manda o nome da tarefa (g1/Pick_...), não a descrição
H = srv.Tp

def frames(ep):
    with av.open(str(D / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4")) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]

for ep in map(int, args[1:]):
    df = pd.read_parquet(D / f"data/chunk-000/episode_{ep:06d}.parquet")
    S, A = np.stack(df["states"].values), np.stack(df["action"].values)
    imgs = frames(ep); n = min(len(df), len(imgs))
    starts, preds, lat, wall = [], [], [], []
    for t0 in range(0, n - 1, H):
        obs = np.concatenate([S[t0, :28], np.zeros(3, np.float32), np.array([0.75], np.float32)])
        obs = pad_to_len(obs, srv.maxmin.pad_state_dim, dim=0)[0] if srv.maxmin.pad_state_dim != len(obs) else obs
        obs = srv.maxmin.normalize_state_func(obs)[np.newaxis, np.newaxis, :].astype(np.float32)
        o = {"imgs": [srv.preprocess_image({"cam0": Image.fromarray(imgs[t0])})["cam0"]], "obs": obs, "text_instructions": [instr]}
        wall.append(dt.datetime.now().isoformat(timespec="milliseconds")); t = time.perf_counter()
        with torch.inference_mode():
            a = srv.model.predict_action(observations=o["imgs"], states=torch.from_numpy(o["obs"]).to(srv.device), traj2ds=None,
                                         instructions=o["text_instructions"], num_inference_steps=8)[0].float().cpu().numpy()
        torch.cuda.synchronize(); lat.append(time.perf_counter() - t)
        starts.append(t0); preds.append(srv._postprocess_action(a))
    P = np.stack(preds)                                   # (nchunks, H, 36)
    pred_seq = P.reshape(-1, P.shape[-1])[:n]             # execução em malha aberta: bloco inteiro a cada H quadros
    err = np.abs(pred_seq - A[:len(pred_seq)])
    np.savez_compressed(out / f"ep{ep:03d}.npz", starts=np.array(starts), pred_chunks=P, pred_seq=pred_seq, gt_action=A[:n], states=S[:n],
                        timestamp=df["timestamp"].values[:n], latency_s=np.array(lat), wall=np.array(wall), instruction=instr)
    print(json.dumps({"ep": ep, "frames": n, "chunks": len(starts), "lat_med_s": float(np.median(lat[1:] or lat)),
                      "mae_hands": float(err[:, :14].mean()), "mae_arms": float(err[:, 14:28].mean()), "mae_base": float(err[:, 28:].mean())}), flush=True)
