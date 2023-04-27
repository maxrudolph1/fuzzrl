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
import subprocess
import coverage
import re
import jxlpy
from zarr_jpeg2k import jpeg2k
# import numpy as np
import zarr
import contextlib
import io

if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well


VIEWPORT_W = 600
VIEWPORT_H = 400




class FuzzGym(gym.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        continuous: bool = True,
        state_mode: str = "img",
        action_mode: str = "bitmap",
        render_mode: Optional[str] = None,
    ):
        print(state_mode)

        self.continuous = continuous
        
        self.state_mode = state_mode
        if self.state_mode == 'img':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        else:    
            self.observation_space = spaces.Box(
                    low=0, high=255, shape=(64 * 64, ), dtype=np.uint8)
        self.action_mode = action_mode
        if action_mode == "continuous":
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (18,), dtype=np.float32)
        elif action_mode == "bitmap":
            self.action_space = spaces.Box(-1,1, (9, ), dtype=np.float32)
        elif action_mode == "parameters":
            self.action_space = spaces.Box(1,100, (4, ), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)


        self.render_mode = render_mode
        self.max_convs = 10
        self.imgstart = np.random.randint(low=0, high=255, size=(64, 64))
        self.size = (64,64)
        self.cov = coverage.Coverage()
        self.enc = jxlpy.JXLPyEncoder(quality=100, colorspace='RGB', size=self.size, effort=7)
        self.codec = jpeg2k(level=50)
        self.output = io.StringIO()
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        print('\n\n\n\n\n')
        super().reset(seed=seed)
        self.convs = 0
        
        

        self.img = self.imgstart
        
        if self.state_mode == 'img':
            self.stacked_img = np.stack([self.img, self.img, self.img], axis=2)
            return self.stacked_img, {}
        else:
            return self.img.flatten(), {}
        

    def fuzzing_target(self, a):
        
        # self.img += a.reshape(np.array(self.img).shape).astype(int)
        print(a)
        if self.action_mode == 'bitmap':
            a = a.reshape((3,3))
            self.img = scipy.signal.convolve2d(self.img, a, mode='same')
            self.img = np.clip(self.img, 0, 255)
            toencode = bytes([int(x) for x in self.img.flatten().tolist()])
            enc = jxlpy.JXLPyEncoder(quality=100, colorspace='RGB', size=(64,64), effort=7)
            enc.add_frame(toencode)
            enc.get_output()
        elif self.action_mode == 'parameters':
            qual = int(a[0])
            h = int(np.clip(a[1], 1, 100))
            w = int(np.clip(a[1], 1, 100))
            eff = int(np.clip(a[3], 3, 9))
            
            # print(a)
            fuzz_img = np.random.randint(low=0, high=255, size=(h, w))
            toencode = bytes([int(x) for x in fuzz_img.flatten().tolist()])
            enc = jxlpy.JXLPyEncoder(quality=qual, colorspace='RGB', size=(h,w), effort=eff)
            enc.add_frame(toencode)
            enc.get_output()
            
        return self.img

    def step(self, action):
        
        # cmd = 'coverage run --branch simplejpeg_test.py'
        # subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)  
        # cmd = 'coverage report -m'
        # results = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        # total_rest = results.stdout
        # post_total = total_rest.split("TOTAL")[1].split(" ")
        # post_total = ([x for x in post_total if x != ''][-1])
        # reward = int(post_total[:-2])
        
        str = time.time()
        self.cov.start()
        action = action.astype(np.float32)
        
        self.fuzzing_target(action)
        self.cov.stop()
        with contextlib.redirect_stdout(self.output):
            out = self.cov.report()
        reward = out
        done = False
        end = time.time()
        reward=(end-str + reward)
        if self.convs >= self.max_convs:
            done = True
        else:
            self.convs += 1
            
        # print(np.sum(self.img))
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
    env = gym.make("fuzzgym/FuzzGym-v0")
    demo_heuristic_fuzzer(env, render=True)