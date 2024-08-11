from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
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

class SimpleEnv(MiniGridEnv):
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

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 3 * size**2

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
    def _gen_mission(): #change_reward : bool = False):
        # colors_rank = "red > green > blue"
        # if change_reward:
        #     colors_rank = "blue > green > red"
        return "Collect as many balls as possible, colors rank: red > green > blue"# + colors_rank

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        

        # Place a ball in some cells
        self.num_objects = 3

        for _ in range(self.num_objects):
            x_loc = self._rand_int(1, width - 2)
            y_loc = self._rand_int(1, height - 2)
            # skip in the object is in the goal or agent position
            if (x_loc, y_loc) == (width - 2, height - 2) or (x_loc, y_loc) == (1, 1):
                continue
            color = random.choice(list(self.color_rewards.keys()))
            self.put_obj(Ball(color), x_loc, y_loc)
            self.on_baord_objects += 1
        
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
            # if self.on_baord_objects == 0:
            #     terminated = True

        self.steps += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        self.on_baord_objects = 0
        return super().reset()



class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
    

class ObjEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                cnn = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with torch.no_grad():
                    n_flatten = cnn(
                        torch.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

            elif key == "mission":
                pass
                # extractors["mission"] = nn.Linear(subspace.shape[0], 32)
                # total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
    

class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
            }
        )

        # self.color_one_hot_dict = {
        #     "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        #     "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        #     "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        #     "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        #     "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        #     "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        # }

        # self.obj_one_hot_dict = {
        #     "ball": np.array([1.0, 0.0, 0.0]),
        #     "box": np.array([0.0, 1.0, 0.0]),
        #     "key": np.array([0.0, 0.0, 1.0]),
        # }

    def observation(self, obs):
        # mission_array = np.concatenate(
        #     [
        #         self.color_one_hot_dict[self.target_color],
        #         self.obj_one_hot_dict[self.target_obj],
        #     ]
        # )

        wrapped_obs = {
            "image": obs["image"],
            # "mission": mission_array,
        }
        

        return wrapped_obs
    

def main():
    # Initialize Pygame
    pygame.init()

    # For headless environments, set the SDL_VIDEODRIVER environment variable to 'dummy'.
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    env = SimpleEnv(render_mode="human")
    env = ObjObsWrapper(env)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()