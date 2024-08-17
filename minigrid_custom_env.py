from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key, Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ObservationWrapper

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import random
import pygame
import os
import numpy as np



class CustomEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        change_reward: bool = False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.agent_pos = agent_start_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 5 * size
        
        self.step_count = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Define rewards for each color
        if change_reward: # 2 option for reward ranking
            self.color_rewards = {
                'blue': 2,
                'green': 1.5,
                'red': 1
            }
        else:
            self.color_rewards = {
                'red': 2,
                'green': 1.5,
                'blue': 1
            }

    @staticmethod
    def _gen_mission():
        return ""
        # return "go to the red box" # just for the it to run, TODO: remove this
    
        # color_rank = ' red > green > blue' if not self.change_reward else ' blue > green > red'
        # return "Collect as many balls as possible, colors rank: " + color_rank

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        

        # Place a ball in some cells
        num_objects = 3

        for _ in range(num_objects):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            if (x_loc, y_loc) == (width - 2, height - 2) or (x_loc, y_loc) == (1, 1):
                continue
            color = random.choice(list(self.color_rewards.keys()))
            self.put_obj(Ball(color), x_loc, y_loc)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if the agent picked up a ball
        if self.carrying:
            ball_color = self.carrying.color
            reward += self.color_rewards.get(ball_color, 0) 
            self.carrying = None
            self.on_baord_objects -= 1
            # if self.on_baord_objects == 0: # if all balls are collected end the episode
            #     terminated = True

        reward -= 1 / self.max_steps 
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.on_baord_objects = 0
        return super().reset()


def main():
    env = CustomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()
