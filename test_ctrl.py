import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("car_scene.xml")
d = mujoco.MjData(m)

mujoco.mj_step(m, d)
print("Initial qpos:", d.qpos)
print("Initial qvel:", d.qvel)

d.ctrl[0] = 10.0
mujoco.mj_step(m, d)
print("After step with ctrl[0]=10.0, qvel:", d.qvel)
