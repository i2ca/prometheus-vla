"""Converte as demos aceitas de gera_demos.py para LeRobot v2.1 no mesmo molde do dataset da Task 3 do Ψ0
(data/chunk-000/*.parquet com `states` e `action` em lista float32, vídeos h264 da cabeça e dos dois punhos em videos/chunk-000/, meta/ com
info, tasks, episodes, episodes_stats e stats_psi0.json, este último lido pelo finetune com action_norm_type=bounds).
Estado e ação têm 28 valores (mãos 14 + braços 14) ou 31 (+ cintura roll, pitch, yaw, nas posições do rpy do tronco do Ψ0);
o treino completa até 36 com zeros, que a normalização ignora.
O texto da tarefa é a instrução (o Ψ0 usa a coluna `task` como instrução).

Uso: python para_lerobot.py <pasta das demos> <raiz>/<nome_do_dataset>
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

FPS = 30


def estat(x):
    return dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(), min=x.min(0).tolist(), max=x.max(0).tolist(),
                q01=np.quantile(x, 0.01, 0).tolist(), q99=np.quantile(x, 0.99, 0).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos"); ap.add_argument("saida")
    ap.add_argument("--excluir", nargs="*", default=[], help="demos a deixar de fora mesmo aceitas pelo roteiro (ex. reprovadas na revisão visual)")
    ap.add_argument("--incluir", nargs="*", help="quando usado, converte somente estas demos")
    a = ap.parse_args(); src, out = Path(a.demos), Path(a.saida)
    eps = sorted(p for p in src.glob("ep*") if (p / "info.json").exists() and json.loads((p / "info.json").read_text())["aceita"] and p.name not in a.excluir and (a.incluir is None or p.name in a.incluir))
    assert eps, "nenhuma demo aceita"
    instr = json.loads((eps[0] / "info.json").read_text())["instrucao"]
    for sub in ("data/chunk-000", "videos/chunk-000/observation.images.head", "videos/chunk-000/observation.images.left_wrist", "videos/chunk-000/observation.images.right_wrist", "meta"): (out / sub).mkdir(parents=True, exist_ok=True)
    todos_s, todos_a, linhas, est_ep, idx = [], [], [], [], 0
    for i, ep in enumerate(eps):
        z = np.load(ep / "dados.npz"); S, A = z["states"].astype(np.float32), z["action"].astype(np.float32); n = len(S)
        assert S.shape == A.shape and S.shape[1] in (28, 31); dim = S.shape[1]
        df = pd.DataFrame({"states": list(S), "action": list(A), "timestamp": (np.arange(n) / FPS).astype(np.float32),
                           "frame_index": np.arange(n), "episode_index": np.full(n, i), "index": np.arange(idx, idx + n),
                           "task_index": np.zeros(n, np.int64), "next.done": np.arange(n) == n - 1})
        df.to_parquet(out / f"data/chunk-000/episode_{i:06d}.parquet")
        for cam in ("head", "left_wrist", "right_wrist"):
            shutil.copy(ep / f"{cam}.mp4", out / f"videos/chunk-000/observation.images.{cam}/episode_{i:06d}.mp4")
        info = json.loads((ep / "info.json").read_text())
        linhas.append(dict(episode_index=i, tasks=[instr], length=n, origem=ep.name, copo_xy=info["xy"], copo_yaw_deg=info["yaw_deg"]))
        t = (np.arange(n) / FPS)[:, None]
        est_ep.append(dict(episode_index=i, stats={"action": dict(estat(A), count=[n]), "states": dict(estat(S), count=[n]),
                                                   "timestamp": dict(estat(t), count=[n])}))
        todos_s.append(S); todos_a.append(A); idx += n
    S, A = np.concatenate(todos_s), np.concatenate(todos_a)
    video = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"],
             "video_info": {"video.fps": 30.0, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    feats = {**{f"observation.images.{cam}": video for cam in ("head", "left_wrist", "right_wrist")},
             "states": {"dtype": "float32", "shape": [dim]}, "action": {"dtype": "float32", "shape": [dim]},
             **{k: {"dtype": t, "shape": [1]} for k, t in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                                                          ("index", "int64"), ("next.done", "bool"), ("task_index", "int64"))}}
    meta = out / "meta"
    (meta / "info.json").write_text(json.dumps(dict(codebase_version="v2.1", robot_type="g1", total_episodes=len(eps), total_frames=int(idx),
        total_tasks=1, total_videos=3 * len(eps), total_chunks=1, chunks_size=1000, fps=FPS, splits={"train": f"0:{len(eps)}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", features=feats), indent=4))
    (meta / "tasks.jsonl").write_text(json.dumps(dict(task_index=0, task=instr)) + "\n")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(l) + "\n" for l in linhas))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in est_ep))
    (meta / "stats_psi0.json").write_text(json.dumps({"states": estat(S), "action": estat(A)}, indent=1))
    print(json.dumps(dict(episodios=len(eps), quadros=int(idx), instrucao=instr)))


if __name__ == "__main__":
    main()
