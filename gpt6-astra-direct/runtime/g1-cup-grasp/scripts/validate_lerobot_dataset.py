"""Le o dataset LeRobot de volta (API do LeRobot, nao os arquivos .npy) e reproduz um episodio no simulador a partir
do `action` do parquet: mesma cena, mesmo copo, mesmos ganhos, alvo PD = action[i] no quadro i. Confere nomes, formas,
dtypes, depth uint16, e que a trajectoria reproduzida bate com observation.state dentro de 1e-4 rad e que a pega
acontece (copo sobe). Rodar com .venv-lerobot/bin/python (tem lerobot + mujoco).

Uso: validate_lerobot_dataset.py <raiz do dataset> --episode 0 --parameters <parameters.json do episodio> [--video saida.mp4]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import mujoco
from g1_sim import G1Sim, HAND_JOINTS
from dataset_recorder import joint_for, NAMES

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--episode", type=int, default=0); ap.add_argument("--parameters", required=True)
ap.add_argument("--video"); a = ap.parse_args()
p = json.load(open(a.parameters)); root = Path(a.root)
info = json.load(open(root / "meta" / "info.json"))
ds = LeRobotDataset(info.get("repo_id", "local/x"), root=root, video_backend="pyav")   # torchcodec exige ffmpeg <= 7; o sistema tem o 9
assert ds.features["observation.state"]["names"] == NAMES == ds.features["action"]["names"], "nomes do esquema diferem"
assert ds.features["observation.state"]["shape"] == (29,) or list(ds.features["observation.state"]["shape"]) == [29]
ep = ds.meta.episodes[a.episode]; i0, i1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
f0 = ds[i0]
report = {"episode": a.episode, "frames": i1 - i0, "fps": ds.fps, "keys": sorted(f0.keys())}
for k in ("observation.images.head_camera", "observation.images.right_wrist_camera", "observation.images.head_camera_depth"):
    v = f0[k]; report[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "min": float(v.min()), "max": float(v.max())}
# depth: o LeRobot devolve tensor; o PNG original e uint16 em mm. Confere o valor bruto pela imagem embutida no parquet.
import pyarrow.parquet as pq, io
from PIL import Image
tbl = pq.read_table(root / ds.meta.data_path.format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
row = tbl.slice(i0 - int(tbl.column("index")[0].as_py()), 1).to_pylist()[0]
dimg = Image.open(io.BytesIO(row["observation.images.head_camera_depth"]["bytes"]))
darr = np.array(dimg); report["depth_png"] = {"mode": dimg.mode, "dtype": str(darr.dtype), "shape": list(darr.shape), "min_mm": int(darr.min()), "max_mm": int(darr.max()),
                                              "median_mm_center": int(np.median(darr[200:280, 380:470]))}
assert darr.dtype == np.uint16 and dimg.mode == "I;16", "depth nao esta em uint16"

state = np.stack([np.asarray(ds[i]["observation.state"]) for i in range(i0, i1)]); action = np.stack([np.asarray(ds[i]["action"]) for i in range(i0, i1)])
assert state.shape == (i1 - i0, 29) and action.dtype == np.float32
report["task"] = ds[i0]["task"]

# ---- replay no simulador ----
sim = G1Sim("scene_grasp.xml"); m, d = sim.m, sim.d
joints = [joint_for(n) for n in NAMES]; qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in joints]); aidx = np.array([sim.act_joint[j] for j in joints])
d.qpos[qadr] = action[0]                       # quadro 0 e a fase de repouso: alvo = pose inicial
sim.q_des[:] = d.qpos[sim.qadr]
sim.set_targets(HAND_JOINTS, action[0][[NAMES.index(f"{j}.q") for j in HAND_JOINTS]], kp=p["kp"], kd=1)
yaw = np.radians(p["cup_yaw_deg"]); sim.place_cup([*p["cup_xy"], 0.752], quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
cup0 = sim.cup_pos().copy(); fps = ds.fps; replay = []; frames = []
rend = mujoco.Renderer(m, 360, 640) if a.video else None
for i in range(i1 - i0):
    sim.q_des[aidx] = action[i]
    until = (i + 1) / fps
    while d.time < until - m.opt.timestep / 2:
        tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr] + d.qfrc_bias[sim.vadr]
        d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d); replay.append(d.qpos[qadr].copy())
    if rend: rend.update_scene(d, camera="view_center"); frames.append(rend.render().copy())
replay = np.array(replay); err = np.abs(replay - state); per_frame = err.max(1)
first_div = int(np.argmax(per_frame > 1e-5)) if (per_frame > 1e-5).any() else None
report["replay"] = {"max_abs_err_rad": float(err.max()), "max_abs_err_joint": NAMES[int(err.max(0).argmax())], "frame_of_max": int(err.max(1).argmax()),
                    "mean_abs_err_rad": float(err.mean()), "first_frame_err_gt_1e-5": first_div,
                    "max_err_per_50_frames_rad": [float(per_frame[i:i+50].max()) for i in range(0, len(per_frame), 50)], "cup_rise_m": float(sim.cup_pos()[2] - cup0[2]),
                    "cup_tilt_deg_end": float(np.degrees(np.arccos(np.clip(d.xmat[sim.cup_body].reshape(3, 3)[2, 2], -1, 1))))}
report["exact_replay"] = bool(err.max() < 1e-4)
report["grasp_reproduced"] = bool(report["replay"]["cup_rise_m"] > 0.05 and report["replay"]["cup_tilt_deg_end"] < 15)
report["ok"] = bool(report["grasp_reproduced"] and err.mean() < 1e-3 and err.max() < 1e-2)
if a.video:
    import subprocess
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", a.video], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close(); pr.wait()
print(json.dumps(report, indent=2))
