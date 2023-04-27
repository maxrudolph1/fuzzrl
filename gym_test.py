import gymnasium as gym
import jpeg_encode
import fuzzgym
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
env = gym.make("FuzzGym-v0")



seed=0
total_reward = 0
steps = 0
s, info = env.reset(seed=seed)
plt.figure(1)
a = np.array([1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1])/16
a -= a.mean()
print(a.sum())
for i in range(9):
    k = time.time()
    s, r, done, truncated, info = env.step(np.random.random((64 * 64)))
    total_reward += r

    steps += 1
    # print(i)

    if done:
        break
    
plt.show()



