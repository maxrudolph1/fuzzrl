from gymnasium.envs.registration import register

register(
     id="FuzzGym-v0",
     entry_point="fuzzgym.envs.fuzzgym:FuzzGym",
     kwargs={"continuous": True},
     max_episode_steps=10,
     reward_threshold=200,
)

