import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cv2
from g1_sim import G1Sim, load_seed, ARM_JOINTS, HAND_JOINTS, CLOSE_TGT, ROOT
arm, hand, fps = load_seed(0); gi = int(np.where(hand[:,4]>0.5)[0][0])
sim = G1Sim(); m, d = sim.m, sim.d
def pose(h, tag, cup=None):
    d.qpos[:] = 0
    for jn, q in zip(ARM_JOINTS, arm[gi]): d.qpos[sim.qadr[sim.act_joint[jn]]] = q
    for jn, q in zip(HAND_JOINTS, h): d.qpos[sim.qadr[sim.act_joint[jn]]] = q
    mujoco.mj_forward(m, d)
    palm = sim.body_pos("right_wrist_yaw_link"); R = d.xmat[m.body("right_wrist_yaw_link").id].reshape(3,3)
    print(f"== {tag}: hand q={np.round(h,2)}")
    for b in ("right_wrist_yaw_link","right_hand_thumb_0_link","right_hand_thumb_1_link","right_hand_thumb_2_link","right_hand_index_0_link","right_hand_index_1_link","right_hand_middle_0_link","right_hand_middle_1_link"):
        p = sim.body_pos(b); print(f"   {b:28s} world={np.round(p,3)} palm_frame={np.round(R.T@(p-palm),3)}")
    gc = sim.grasp_center(); print("   grasp_center", np.round(gc,3), "palm z axes", np.round(R,2).tolist())
    if cup is not None: sim.place_cup(cup)
    r = mujoco.Renderer(m, 600, 800)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = gc; cam.distance = 0.45
    for az, el in ((0,-20),(90,-20),(180,-20),(0,-70)):
        cam.azimuth, cam.elevation = az, el; r.update_scene(d, camera=cam)
        cv2.imwrite(os.path.join(ROOT,"results",f"view-{tag}-az{az}-el{el}.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    return gc
gc = pose(hand[0], "open")
pose(hand[-1], "seed-closed", cup=gc)
pose(CLOSE_TGT, "close-tgt", cup=gc)
print("copo geom aabb center offset", np.round(m.geom_aabb[m.geom('copo_geom').id][:3],4), "half", np.round(m.geom_aabb[m.geom('copo_geom').id][3:],4))
