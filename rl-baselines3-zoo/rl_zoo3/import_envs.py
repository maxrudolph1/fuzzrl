import gym
from gym.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None
    
try:
    import jpeg_encode  # pytype: disable=import-error
except ImportError:
    jpeg_encode = None
    print("jpeg_encode not found")

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    rocket_lander_gym = None


# Register no vel envs
def create_no_vel_env(env_id: str):
    def make_env():
        env = gym.make(env_id)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )


register(
     id="JPEGEncode-v0",
     entry_point="jpeg_encode.envs.jpeg_encode:JPEGEncode",
     kwargs={"continuous": True, "state_mode": "img"},
     max_episode_steps=10,
     reward_threshold=200,
)

register(
     id="FuzzGym-v0",
     entry_point="fuzzgym.envs.fuzzgym:FuzzGym",
     kwargs={"continuous": True, "state_mode": "img"},
     max_episode_steps=10,
     reward_threshold=200,
)