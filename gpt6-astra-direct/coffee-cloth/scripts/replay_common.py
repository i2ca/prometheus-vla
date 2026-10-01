"""Diagnostic setup from preserved G1 attempt31. Model: gpt-6-astra."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
from pathlib import Path
import json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
G1=ROOT.parent/'g1-cup-grasp'
sys.path.insert(0,str(G1/'scripts'))
from g1_sim import G1Sim, HAND_JOINTS, ARM_JOINTS
from dataset_recorder import joint_for,NAMES
from kinematics import ArmIK,PALM


def replay_grasp(scene=None, observer=None, frame_observer=None, hand_kp=None):
    source=G1/'results/attempt-31'
    p=json.loads((source/'parameters.json').read_text())
    action=np.load(source/'dataset/action.npy')
    sim=G1Sim(str(scene) if scene else 'scene_grasp.xml')
    m,d=sim.m,sim.d
    joints=[joint_for(n) for n in NAMES]
    qa=np.array([m.jnt_qposadr[m.joint(j).id] for j in joints])
    ai=np.array([sim.act_joint[j] for j in joints])
    d.qpos[qa]=action[0]
    sim.q_des[:]=d.qpos[sim.qadr]
    sim.set_targets(HAND_JOINTS,action[0][[NAMES.index(j+'.q') for j in HAND_JOINTS]],kp=p['kp'] if hand_kp is None else hand_kp,kd=1)
    yaw=np.radians(p['cup_yaw_deg'])
    sim.place_cup([*p['cup_xy'],.752],quat=(np.cos(yaw/2),0,0,np.sin(yaw/2)))
    for i,a in enumerate(action):
        sim.q_des[ai]=a
        advance_to(sim,(i+1)/30,observer)
        if frame_observer: frame_observer(sim)
    return sim


def advance_to(sim,until,observer=None):
    m,d=sim.m,sim.d
    while d.time < until-m.opt.timestep/2:
        tau=sim.kp*(sim.q_des-d.qpos[sim.qadr])-sim.kd*d.qvel[sim.vadr]+d.qfrc_bias[sim.vadr]
        d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        mujoco.mj_step(m,d)
        if observer: observer(sim)
    mujoco.mj_forward(m,d)
