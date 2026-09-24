"""Monta um dataset LeRobot v3.0 (lerobot 0.4.4, o do conda `g1` do laboratorio) a partir das pastas <run>/dataset/ gravadas
pelo DatasetRecorder. Usa a API publica (create / add_frame / save_episode / finalize), sem as injecoes do
lerobot-ext/init_lerobot_record.py do prometheus-vla. Depth entra como PIL "I;16" (PNG de 16 bits), que o writer
do LeRobot salva sem conversao. Rodar com ~/miniconda3/envs/g1/bin/python.

Uso: build_lerobot_dataset.py --out results/dataset-01/lerobot --repo-id lewislf/g1_cup_sim_v1 run1/dataset run2/dataset ...
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def read_video(path, w, h):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--repo-id", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); out = Path(a.out)
    if out.exists(): shutil.rmtree(out)
    meta0 = json.load(open(Path(a.episodes[0]) / "meta.json")); names = meta0["names"]; fps = meta0["fps"]
    features = {
        "action": {"dtype": "float32", "shape": (29,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (29,), "names": names},
        "observation.images.head_camera": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
        "observation.images.head_camera_depth": {"dtype": "image", "shape": (480, 848, 1), "names": ["height", "width", "channels"]},
        "observation.images.right_wrist_camera": {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(a.repo_id, fps, features, root=out, robot_type="unitree_g1_dex3", use_videos=True, image_writer_threads=a.threads, vcodec="h264")
    log = []
    for ep in a.episodes:
        ep = Path(ep); meta = json.load(open(ep / "meta.json")); assert meta["names"] == names
        st = np.load(ep / "observation.state.npy"); ac = np.load(ep / "action.npy"); n = len(st)
        head = read_video(ep / "head_camera.mp4", 848, 480); wrist = read_video(ep / "right_wrist_camera.mp4", 224, 224)
        assert len(head) == n == len(wrist) == meta["frames"], (len(head), n, len(wrist))
        for i in range(n):
            depth = Image.open(ep / "head_depth" / f"frame-{i:06d}.png"); assert depth.mode == "I;16"
            ds.add_frame({"action": ac[i], "observation.state": st[i], "observation.images.head_camera": head[i],
                          "observation.images.head_camera_depth": depth, "observation.images.right_wrist_camera": wrist[i], "task": meta["task"]})
        ds.save_episode()
        log.append({"source": str(ep), "frames": n, "cup_xy": meta.get("cup_xy"), "cup_yaw_deg": meta.get("cup_yaw_deg")})
        print(f"episodio {len(log)-1}: {ep} ({n} quadros)", flush=True)
    ds.finalize()
    json.dump({"episodes": log, "names": names, "omitted": meta0["omitted"], "depth": meta0["depth"],
               "note": "sem pressao das maos: nao concatenar com o dataset real (aggregate.py recusa esquemas diferentes)"},
              open(out / "meta" / "sim_provenance.json", "w"), indent=2)
    print("pronto:", out)


if __name__ == "__main__":
    main()
