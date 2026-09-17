#!/usr/bin/env bash
# Recoloca a cena do unitree_sim_isaaclab (dominio DDS 1), sem tocar em nada mais.
~/miniforge3/envs/prometheus-vla/bin/python - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
ChannelFactoryInitialize(1)
p = ChannelPublisher("rt/reset_pose/cmd", String_); p.Init()
time.sleep(0.5)
for _ in range(3):
    p.Write(String_(data="1")); time.sleep(0.3)
print("cena recolocada")
PY
