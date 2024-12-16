from __future__ import annotations

from minigrid.core.constants import *
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Goal, Door, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.world_object import Point, WorldObj


import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ObservationWrapper
from gymnasium import spaces


# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch
# import torch.nn as nn


# import matplotlib.pyplot as plt
import random
# import pygame
# import os
import numpy as np

basic_colors_rewards  = {
                'red': 2,
                'green': 2,
                'blue': 2,
            }


class CustomEnv(MiniGridEnv):
    def __init__(
        self,
        grid_size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int = 100, 
        change_reward: bool = False,
        num_objects: int = 6,
        difficult_grid: bool = False,
        train_env: bool = False,
        unique_env: int = 0,
        lava_cells: int = 2,
        image_full_view: bool = False,
        width: int | None = None,
        height: int | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = None,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        colors_rewards: dict = basic_colors_rewards, # all colors have the same reward = 2
        partial_obs: bool = False,
        step_count_observation: bool = False,
        # lava_reward: int = 0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.agent_pos = agent_start_pos
        self.image_full_view = image_full_view
        self.partial_obs = partial_obs
        self.unique_env = unique_env
        self.step_count_observation = step_count_observation
        
        if not highlight:
            self.highlight = not image_full_view

        if max_steps is None:
            max_steps = size * size 
        self.train_env = train_env
        # if self.train_env:
        #     max_steps = size * size * 5
        self.num_objects = num_objects
        self.difficult_grid = difficult_grid
        self.num_lava_cells = lava_cells
        # self.lava_reward = lava_reward
        self.current_state = None
        self.took_key = None
        self.step_count = 0
        self.initial_objects = [] # list of objects on the grid in the format (x, y, reward when picked)

        # super().__init__(
        #     mission_space=mission_space,
        #     grid_size=size,
        #     # see_through_walls=True,
        #     max_steps=max_steps,
        #     highlight= not image_full_view,
        #     agent_view_size=7, # size - (1-size%2),
        #     **kwargs,
        # )
        
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        if self.image_full_view:
            self.agent_view_size = max(width, height)
            image_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(width, height, 3),
                dtype="uint8",
            )
        else:
            assert agent_view_size % 2 == 1
            assert agent_view_size >= 3
            self.agent_view_size = agent_view_size
            image_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(agent_view_size, agent_view_size, 3),
                dtype="uint8",
            )

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                # "step_count": spaces.Box(low=0, high=max_steps+1, shape=(1,), dtype="int"),
                # "mission": None,
            }
        )
        if self.step_count_observation:
            print("add step count")
            self.observation_space["step_count"] = spaces.Box(low=0, high=max_steps+1, shape=(1,), dtype="int")

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos: np.ndarray | tuple[int, int] = None
        self.agent_dir: int = None

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        # Define rewards for each color
        self.color_rewards = colors_rewards

    def reset(self, **kwargs):
        self.on_baord_objects = 0
        self.step_count = 0
        self.took_key = False
        self.current_state = {}
        self.initial_objects = []
        state , info = super().reset()
        if self.image_full_view:
            state['image'] = self.grid.encode()
            self.put_agent_in_obs(state)
        self.current_state['image'] = state['image']
        if self.step_count_observation:
            self.current_state['step_count'] = self.step_count
            state['step_count'] = self.step_count
        return state, info
    
    @staticmethod
    def _gen_mission():
        return ""
        # return "go to the red box" # just for the it to run, TODO: remove this
    
        # color_rank = ' red > green > blue' if not self.change_reward else ' blue > green > red'
        # return "Collect as many balls as possible, colors rank: " + color_rank

    def _gen_grid(self, width, height):
        self.mission = self._gen_mission()

        # Create an empty grid
        self.grid = Grid(width, height)
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.unique_env > 0:
            return self._gen_unique_grid(width, height)
        if self.difficult_grid and width >= 8 and height >= 8:
            self._gen_difficult_grid(width, height)
            return
        
        # place lava cells
        for _ in range(self.num_lava_cells):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            if (x_loc == width - 2 and y_loc == height - 2) or x_loc == 1 and y_loc == 1:
                continue
            self.put_obj(Lava(), x_loc, y_loc)
            
        # Place a ball in some cells
        for _ in range(self.num_objects):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            if (x_loc, y_loc) == (width - 2, height - 2) or (x_loc, y_loc) == (1, 1):
                continue
            color = random.choice(list(self.color_rewards.keys()))
            self.put_obj(Ball(color), x_loc, y_loc)
            self.initial_objects.append((x_loc, y_loc, self.color_rewards[color]))
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)


    def _gen_unique_grid(self, width, height):
        if self.unique_env == 1:
            # put a red ball in the right top corner and lava cells around it
            self.put_obj(Ball('red'), width-2, 1)
            self.initial_objects.append((width-2, 1, self.color_rewards['red']))
            self.put_obj(Lava(), width-3, 2)
            self.put_obj(Lava(), width-3, 1)
            self.put_obj(Lava(), width-2, 2)

            self.put_obj(Ball('blue'), width-4, 3)
            self.initial_objects.append((width-4, 3, self.color_rewards['blue']))
        if self.unique_env == 2:
            self.put_obj(Ball('blue'), width-2, 1)
            self.initial_objects.append((width-2, 1, self.color_rewards['blue']))
            self.put_obj(Ball('green'), 1, height-2)
            self.initial_objects.append((1, height-2, self.color_rewards['green']))
            self.put_obj(Ball('green'), 2, height-2)
            self.initial_objects.append((2, height-2, self.color_rewards['green']))
            self.put_obj(Ball('green'), 3, height-2)
            self.initial_objects.append((3, height-2, self.color_rewards['green']))


        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

    def _gen_difficult_grid(self, width, height):    
        # place lava cells
        for i in range(self.num_lava_cells):
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
            self.initial_objects.append((x_loc, y_loc, self.color_rewards[color]))
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
            
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
    
    def put_agent_in_obs(self, obs):
        i, j = self.agent_pos
        obs['image'][i][j] = (OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], 0)
        
    
    def _remove_objects(self, obj_to_remove):
        # obj_to_remove is a list of tuples (x, y)
        for (x,y) in obj_to_remove:
            self.grid.set(x, y, None)
            self.on_baord_objects -= 1
    
    def step(self, action):
        # print(f"step {self.step_count}, action={action}")
        self.step_count += 1
        obs, reward, terminated, truncated, info = super().step(action)
        if self.step_count_observation:
            obs['step_count'] = self.step_count
        if self.image_full_view:
            obs['image'] = self.grid.encode() # get the full grid image
            self.put_agent_in_obs(obs)
        self.current_state = obs
        
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

        # if self.train_env and terminated: # reached the goal
            # reward += 10
        
        # got to the right bottom corner
        if self.agent_pos == (self.grid.width - 2, self.grid.height - 2):
            reward += 5
        
        if truncated:
            terminated = True
            print(f"reached max steps={self.max_steps}")
            # reward -= 10
        reward -= 0.1
        # if terminated:
        #     print(f"terminated, reward={reward}")
        return obs, reward, terminated, truncated, info

    def find_optimal_path(self):
        points = [(1,1,0)] + self.initial_objects
        points = {i: p for i, p in enumerate(points) if p[2] >= 0}
        print("points", points) 
        matrix = np.zeros((len(points.keys()), len(points.keys())))
        for i in range(len(points)):
            p1 = points[i]
            for j in range(len(points)):
                if i == j:
                    matrix[i][j] = np.inf
                    continue
                matrix[i][j] = np.abs(p1[0] - points[j][0]) + np.abs(p1[1] - points[j][1])
        total_reward = 0
        total_steps = 0
        curent_pos = 0
        # print("maxtix", matrix)
        for s in range(len(points)-1):
            min_steps = np.min(matrix[curent_pos])
            min_arg = np.argmin(matrix[curent_pos])
            # print(f"curent_pos {curent_pos}, min_steps {min_steps}, min_arg {min_arg}")
            total_steps += min_steps
            total_reward += points[min_arg][2]
            matrix[:, curent_pos] = np.inf
            curent_pos = min_arg

        p1 = points[curent_pos]
        # print(f"reward earned {total_reward}, total steps {total_steps}, curent_pos {p1}")
        total_steps += np.abs(p1[0] - (self.grid.width - 2)) + np.abs(p1[1] - (self.grid.height - 2)) # go to the goal
        total_reward -= (total_steps + len(points)*2)*0.1 # the aproximation of number of steps
        return total_reward, total_steps

        


    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        Plus - hide all cells that are not highlighted
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        # black all cells that are not highlighted
        if self.partial_obs:
            for j in range(0, self.height):
                for i in range(0, self.width):
                    if highlight_mask[i, j]:
                        continue
                    tile_img = 0

                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img
        
        return img

    def get_full_obs(self):
        tmp = self.partial_obs
        self.partial_obs = False
        img = self.render()
        self.partial_obs = tmp
        return img

def main():
    env = CustomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()
