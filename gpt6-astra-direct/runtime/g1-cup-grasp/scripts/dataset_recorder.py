"""Grava, por quadro, o que o pipeline de treino do LCAD consome (SCHEMA_G1_V2): estado e acao de 29 dimensoes
por NOME de junta, camera da cabeca 848x480 RGB e depth uint16 em mm, camera do punho direito 424x240 com corte
central quadrado e resize para 224x224 (SCHEMA_G1_V2.md 3.1). Saida em <run>/dataset/. Sem pressao das maos (nao existe
no sim) e depth geometrico, sem o ruido de D435 do prometheus (decisoes declaradas no README do dataset)."""
import json, os, subprocess
import numpy as np, mujoco, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = json.load(open(os.path.join(ROOT, "scene", "schema_v2_names.json")))["names"]
# nome do dataset -> junta do MJCF
_ARM = {"ShoulderPitch": "shoulder_pitch", "ShoulderRoll": "shoulder_roll", "ShoulderYaw": "shoulder_yaw", "Elbow": "elbow",
        "WristRoll": "wrist_roll", "WristPitch": "wrist_pitch", "Wristyaw": "wrist_yaw", "WristYaw": "wrist_yaw"}


def joint_for(name):
    n = name[:-2]                                   # tira ".q"
    if n == "kWaistYaw": return "waist_yaw_joint"
    if n.startswith("kLeft") or n.startswith("kRight"):
        side = "left" if n.startswith("kLeft") else "right"; key = n[5:] if side == "left" else n[6:]
        return f"{side}_{_ARM[key]}_joint"
    return n                                        # left_hand_thumb_0_joint etc.


class DatasetRecorder:
    def __init__(self, sim, out_dir, fps=30):
        self.sim, self.m, self.d = sim, sim.m, sim.d; self.fps = fps; self.out = out_dir
        os.makedirs(os.path.join(out_dir, "head_depth"), exist_ok=True)
        self.joints = [joint_for(n) for n in NAMES]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in self.joints])
        self.act_idx = np.array([sim.act_joint[j] for j in self.joints])
        self.r_head = mujoco.Renderer(self.m, 480, 848)
        self.r_depth = mujoco.Renderer(self.m, 480, 848); self.r_depth.enable_depth_rendering()
        self.r_wrist = mujoco.Renderer(self.m, 240, 424)
        self.state, self.action, self.t = [], [], []
        self.head = self._ffmpeg(os.path.join(out_dir, "head_camera.mp4"), 848, 480)
        self.wrist = self._ffmpeg(os.path.join(out_dir, "right_wrist_camera.mp4"), 224, 224)
        self.n = 0

    def _ffmpeg(self, path, w, h):   # lossless, para o construtor do dataset reler sem perda
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                 "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)

    def frame(self):
        d = self.d
        self.state.append(d.qpos[self.qadr].astype(np.float32).copy())
        self.action.append(self.sim.q_des[self.act_idx].astype(np.float32).copy())   # alvo PD bruto, sem clipar (e o comando que o robo recebe)
        self.t.append(self.n / self.fps)
        self.r_head.update_scene(d, camera="head_camera"); self.head.stdin.write(self.r_head.render().tobytes())
        self.r_depth.update_scene(d, camera="head_camera"); depth_m = self.r_depth.render()
        depth_mm = np.clip(depth_m * 1000.0, 0, 32767).astype(np.uint16)          # README_RECORD.md: mm, sobrevive ao int16
        cv2.imwrite(os.path.join(self.out, "head_depth", f"frame-{self.n:06d}.png"), depth_mm)
        self.r_wrist.update_scene(d, camera="right_wrist_camera"); img = self.r_wrist.render()
        h, w = img.shape[:2]; lado = min(h, w); y0, x0 = (h - lado) // 2, (w - lado) // 2
        self.wrist.stdin.write(np.ascontiguousarray(cv2.resize(img[y0:y0+lado, x0:x0+lado], (224, 224), interpolation=cv2.INTER_AREA)).tobytes())
        self.n += 1

    def close(self, meta):
        for p in (self.head, self.wrist): p.stdin.close(); p.wait()
        for r in (self.r_head, self.r_depth, self.r_wrist): r.close()
        np.save(os.path.join(self.out, "observation.state.npy"), np.array(self.state)); np.save(os.path.join(self.out, "action.npy"), np.array(self.action))
        np.save(os.path.join(self.out, "timestamp.npy"), np.array(self.t, np.float32))
        json.dump({"names": NAMES, "joints": self.joints, "fps": self.fps, "frames": self.n, "task": "pick up the white cup",
                   "cameras": {"observation.images.head_camera": [480, 848, 3], "observation.images.head_camera_depth": [480, 848, 1],
                               "observation.images.right_wrist_camera": [224, 224, 3]},
                   "omitted": ["observation.left_hand_pressure", "observation.right_hand_pressure"],
                   "depth": "geometrico do MuJoCo em mm, sem o DEPTH_NOISE do prometheus", **meta}, open(os.path.join(self.out, "meta.json"), "w"), indent=2)
