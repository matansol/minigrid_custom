from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key, Goal, Door, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ObservationWrapper

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch
# import torch.nn as nn


# import matplotlib.pyplot as plt
import random
# import pygame
# import os
# import numpy as np



class CustomEnv(MiniGridEnv):
    def __init__(
        self,
        size=14,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        change_reward: bool = False,
        num_objects: int = 6,
        difficult_grid: bool = False,
        train_env: bool = False,
        lava_cells: int = 1,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.agent_pos = agent_start_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = size * size 
        self.train_env = train_env
        # if self.train_env:
        #     max_steps = size * size * 5
        self.num_objects = num_objects
        self.difficult_grid = difficult_grid
        self.step_count = 0
        self.num_laval_cells = lava_cells
        self.lava_reward = -1
        self.current_state = None
        self.took_key = None
        self.step_count = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            highlight=False,
            agent_view_size=size - (1-size%2),
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

    def reset(self, **kwargs):
        self.on_baord_objects = 0
        self.step_count = 0
        self.took_key = False
        self.current_state = {}
        state , info = super().reset()
        self.current_state['image'] = state['image']
        return state, info
    
    @staticmethod
    def _gen_mission():
        return ""
        # return "go to the red box" # just for the it to run, TODO: remove this
    
        # color_rank = ' red > green > blue' if not self.change_reward else ' blue > green > red'
        # return "Collect as many balls as possible, colors rank: " + color_rank

    def _gen_grid(self, width, height):
        if self.difficult_grid and width >= 8 and height >= 8:
            self._gen_difficult_grid(width, height)
            return
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        

        # Place a ball in some cells

        for _ in range(self.num_objects):
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

    def _gen_difficult_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        self.grid.wall_rect(0, 0, width, height)
        
        for i in range(self.num_laval_cells):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            if (x_loc == width - 2 and y_loc == height - 2) or x_loc == 1 and y_loc == 1:
                continue
            self.put_obj(Lava(), x_loc, y_loc)
            
        #put a wall in the middle of the grid
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)
        
        # Place a door in the wall
        doorIdx = self._rand_int(1, height - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # Ensure a yellow key is placed on the left side
        while True:
            (x_loc, y_loc) = self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))
            placed_obj = self.grid.get(x_loc, y_loc)

            if placed_obj is not None and placed_obj.type == "key":
                break
  
            
        # Place a ball in some cells
        for _ in range(self.num_objects):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            if (x_loc, y_loc) == (width - 2, height - 2) or (x_loc, y_loc) == (1, 1):
                continue
            if x_loc == splitIdx or self.grid.get(x_loc, y_loc) is not None:
                continue
            color = random.choice(list(self.color_rewards.keys()))
            self.put_obj(Ball(color), x_loc, y_loc)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        
        self.mission = self._gen_mission()
    
    def grid_objects(self):
        grid = self.grid
        objects = {"balls": [], "wall": (False, None, None), "key" : (False, None), "lava": []} # wall: (is_wall, splitIdx, doorIdx), key: (is_key, key_pos)
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell is not None:
                    if cell.type == "ball":
                        objects["balls"].append((i, j, cell.color))
                    elif cell.type == "door":
                        objects["wall"] = (True, i, j)
                    elif cell.type == "key":
                        objects["key"] = (True, (i, j))
                    elif cell.type == "lava":
                        objects["lava"].append((i, j))
        return objects
    
    def _remove_objects(self, obj_to_remove):
        # obj_to_remove is a list of tuples (x, y)
        for (x,y) in obj_to_remove:
            self.grid.set(x, y, None)
            self.on_baord_objects -= 1
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_state = obs
        self.step_count += 1
        
        if self.train_env and action == self.actions.pickup and self.carrying and self.carrying.type == "key" and not self.took_key:
            self.took_key = True
            reward += 10

        if action == self.actions.toggle:
            # You can handle the toggle logic here, such as unlocking doors
            fwd_pos = self.front_pos  # Position in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # If the agent is in front of a door, check if it's locked and use the key
            if fwd_cell is not None and fwd_cell.type == "door" and self.carrying and self.carrying.type == "key":
                fwd_cell.is_locked = False  # Unlock the door
                self.carrying = None  # Drop the key after using it
                if self.train_env: # we want to train the agent to use the key to open the door
                    reward += 10
                
        # Check if the agent picked up a ball
        if self.carrying:
            hold_obj = self.carrying
            if hold_obj.type == "ball":
                ball_color = self.carrying.color
                reward += self.color_rewards.get(ball_color, 0) 
                self.carrying = None
            self.on_baord_objects -= 1
            # if self.on_baord_objects == 0: # if all balls are collected end the episode
            #     terminated = True
        if self.step_count >= self.max_steps:
            terminated = True
            print(f"reached max steps={self.max_steps}")
            reward -= 10
        reward -= 0.1
        return obs, reward, terminated, truncated, info



def main():
    env = CustomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()
