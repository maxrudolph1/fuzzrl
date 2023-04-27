from gymnasium.envs.registration import register

register(
     id="FuzzGym-v0",
     entry_point="fuzzgym.envs.fuzzgym:FuzzGym",
     kwargs={"continuous": True},
     max_episode_steps=10,
     reward_threshold=200,
)

# register(
#     id="LunarLanderContinuous-v2",
#     entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
#     kwargs={"continuous": True},
#     max_episode_steps=1000,
#     reward_threshold=200,
# )