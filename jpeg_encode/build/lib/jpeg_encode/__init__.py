from gymnasium.envs.registration import register

register(
     id="JPEGEncode-v0",
     entry_point="jpeg_encode.envs.jpeg_encode:JPEGEncode",
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