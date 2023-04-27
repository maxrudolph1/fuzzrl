__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
import torch
import torch.nn.functional as F
import time
import simplejpeg as sj
import scipy

if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well


VIEWPORT_W = 600
VIEWPORT_H = 400




class JPEGEncode(gym.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        continuous: bool = True,
        state_mode: str = "img",
        render_mode: Optional[str] = None,
    ):

        self.continuous = continuous
        
        self.state_mode = state_mode
        if self.state_mode == 'img':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        else:    
            self.observation_space = spaces.Box(
                    low=0, high=255, shape=(64 * 64, ), dtype=np.uint8)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (18,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)


        self.render_mode = render_mode
        self.max_convs = 10

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed=seed)
        self.convs = 0
        
        self.img = np.random.randint(low=0, high=255, size=(64, 64))

        
        if self.state_mode == 'img':
            self.stacked_img = np.stack([self.img, self.img, self.img], axis=2)
            return self.stacked_img, {}
        else:
            return self.img.flatten(), {}
        



    def step(self, action):
        # print(action)
        
        kernel_len = len(action)
        kernel_width = int(math.sqrt(kernel_len//2))
        kernel1 = action[:kernel_len//2].reshape(kernel_width, kernel_width)
        kernel2 = action[kernel_len//2:].reshape(kernel_width, kernel_width)
        # self.img = torch.stack([self.img, self.img, self.img], dim=0)

        self.img_prev = self.img
        
        self.img = scipy.signal.convolve2d(self.img, kernel1, mode='same')
        self.img[self.img < 0] = 0
        
        self.img = scipy.signal.convolve2d(self.img, kernel2, mode='same')
        
        self.img = self.img.clip(0, 255).astype(np.uint8)
    
        enc_start = (time.time())
        
        # write a timing function to time the encoding and decoding
        # jpeg_enc = torch.stack([self.img, self.img, self.img], dim=0)
        jpeg_enc = np.stack([self.img, self.img, self.img], axis=2)
        encoded = sj.encode_jpeg(jpeg_enc, 85, colorsubsampling='422')
        enc_end = (time.time())
        # print(enc_end)
        
        if self.render_mode == "human":
            self.render()
        
        diff = (self.img- self.img_prev).flatten()
        
        diff_hist, _ = np.histogram(diff, bins=256)
        diff_hist = diff_hist.astype(np.float32)
        diff_hist /= float(diff_hist.sum())
        
        
        
        
        
        reward = enc_end - enc_start
        # reward = scipy.stats.entropy(diff_hist, base=2)
        if self.convs > self.max_convs:
            done = True
        else:
            done = False
            self.convs += 1
            
            
        if self.state_mode == 'img':
            self.stacked_img = np.stack([self.img, self.img, self.img], axis=2)
            return self.stacked_img, reward, done, False, {}
        else:
            return self.img.flatten(), reward, done, False, {}
            

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((608, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            
        # display image IMG with pygame
        

    def close(self):
        # if self.screen is not None:
        #     import pygame

        #     pygame.display.quit()
        #     pygame.quit()
        #     self.isopen = False
        pass






if __name__ == "__main__":
    env = gym.make("jpeg_encode/JPEGEncode-v0")
    demo_heuristic_fuzzer(env, render=True)