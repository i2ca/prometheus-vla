"""Sonda: carrega a cena, mede copo, abertura da Dex3 e enquadramento da head_camera."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, cv2
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(root, "scene", sys.argv[1] if len(sys.argv) > 1 else "scene_gonogo.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("nq", m.nq, "nv", m.nv, "nu", m.nu, "timestep", m.opt.timestep)
print("actuators:", [m.actuator(i).name for i in range(m.nu)][:3], "...", m.nu)
g = m.geom("copo_geom"); print("copo aabb (half sizes)", np.round(m.geom_aabb[g.id], 4), "rbound", round(float(m.geom_rbound[g.id]),4))
b = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
print("palma", np.round(b("right_hand_palm_link"),3), "thumb2", np.round(b("right_hand_thumb_2_link"),3), "index1", np.round(b("right_hand_index_1_link"),3), "middle1", np.round(b("right_hand_middle_1_link"),3))
print("abertura polegar-indicador (home)", round(float(np.linalg.norm(b("right_hand_thumb_2_link")-b("right_hand_index_1_link"))),3))
print("pelvis", np.round(b("pelvis"),3), "copo", np.round(b("copo"),3))
# camera render
r = mujoco.Renderer(m, 480, 848)
for cam in ("head_camera", "global_view"):
    r.update_scene(d, camera=cam); img = r.render()
    cv2.imwrite(os.path.join(root, "results", f"probe-{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cam = m.camera("head_camera"); print("head_camera fovy", cam.fovy, "pos", cam.pos)
# settle 1s with zero ctrl to see if it stands (welded)
for _ in range(500): mujoco.mj_step(m, d)
print("apos 1s sem controle: pelvis", np.round(b("pelvis"),3), "copo", np.round(b("copo"),3), "warnings", int(d.warning.number.sum()))
