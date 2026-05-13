import sys; sys.path.insert(0, '.')
from src.simulation.env import SimEnv, SimEnvConfig
import mujoco
import numpy as np

obs = {"cuboid": {"near_box": {"dims": [0.5,0.5,0.5], "pose": [0,0,0.5,1,0,0,0]}}}
cfg = SimEnvConfig(obstacles=obs)
env = SimEnv(cfg)
m = env._model
print(f"nbody={m.nbody}, ngeom={m.ngeom}")
print("Bodies:")
for i in range(m.nbody):
    print(f"  [{i}] name='{m.body(i).name}'")
print("Geoms:")
for i in range(m.ngeom):
    g = m.geom(i)
    bid = int(m.geom_bodyid[i])
    ct = int(m.geom_contype[i])
    ca = int(m.geom_conaffinity[i])
    print(f"  [{i}] name='{g.name}' body={bid}('{m.body(bid).name}') type={m.geom_type[i]} contype={ct} conaffinity={ca}")
