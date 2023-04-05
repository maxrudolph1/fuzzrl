import gymnasium as gym
import jpeg_encode
import numpy as np
import torch
from matplotlib import pyplot as plt
env = gym.make("JPEGEncode-v0")



seed=0
total_reward = 0
steps = 0
s, info = env.reset(seed=seed)
plt.figure(1)
a = np.array([1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1])/16
a -= a.mean()
print(a.sum())
for i in range(9):
    # a = torch.rand((18,))
    s, r, done, truncated, info = env.step(a)
    total_reward += r
    
    steps += 1
    plt.subplot(3,3,i+1)
    plt.imshow(s)
    if done:
        break
    
plt.show()



