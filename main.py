from __future__ import annotations
import asyncio
import pygame
import random

from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Box, Key, Goal
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace

from minigrid.minigrid_env import MiniGridEnv
# from minigrid_custom_env import CustomEnv

# import argparse
# from datetime import datetime
# from pdb import set_trace
# from time import time
# import os

# import torch as th
# import torch.nn as nn

# import gymnasium as gym
# import minigrid
# import numpy as np

# from gymnasium.core import ObservationWrapper
# from gymnasium.spaces import Box, Dict

# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# from minigrid.manual_control import ManualControl


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

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2
        
        self.steps = 0

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
            # if self.on_baord_objects == 0:
            #     terminated = True

        self.steps += 1
        reward -= self.steps / self.max_steps
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        self.on_baord_objects = 0
        return super().reset()
    




# class ObjObsWrapper(ObservationWrapper):
#     def __init__(self, env):
#         """A wrapper that makes image the only observation.
#         Args:
#             env: The environment to apply the wrapper
#         """
#         super().__init__(env)

#         self.observation_space = Dict(
#             {
#                 "image": env.observation_space.spaces["image"],
#                 #"mission": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
#             }
#         )

#         # self.color_one_hot_dict = {
#         #     "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#         #     "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
#         #     "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#         #     "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#         #     "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#         #     "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
#         # }

#         # self.obj_one_hot_dict = {
#         #     "ball": np.array([1.0, 0.0, 0.0]),
#         #     "box": np.array([0.0, 1.0, 0.0]),
#         #     "key": np.array([0.0, 0.0, 1.0]),
#         # }

#     def observation(self, obs):
#         # mission_array = np.concatenate(
#         #     [
#         #         self.color_one_hot_dict["red"],
#         #         self.obj_one_hot_dict["ball"],
#         #     ]
#         # )

#         wrapped_obs = {
#             "image": obs["image"],
#             # "mission": mission_array,
#         }

#         return wrapped_obs


# class ObjEnvExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super().__init__(observation_space, features_dim=1)

#         extractors = {}
#         total_concat_size = 0

#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 cnn = nn.Sequential(
#                     nn.Conv2d(3, 16, (2, 2)),
#                     nn.ReLU(),
#                     nn.Conv2d(16, 32, (2, 2)),
#                     nn.ReLU(),
#                     nn.Conv2d(32, 64, (2, 2)),
#                     nn.ReLU(),
#                     nn.Flatten(),
#                 )

#                 # Compute shape by doing one forward pass
#                 with th.no_grad():
#                     n_flatten = cnn(
#                         th.as_tensor(subspace.sample()[None]).float()
#                     ).shape[1]

#                 linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
#                 extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
#                 total_concat_size += 64

#             elif key == "mission":
#                 extractors["mission"] = nn.Linear(subspace.shape[0], 32)
#                 total_concat_size += 32

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))

#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)



class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        init_observation = self.env.render()

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)    

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)
            
def main():
    env = CustomEnv(render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
    env.close()
    
def main_old():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", action="store_true", help="train the model")
    # parser.add_argument(
    #     "--load_model",
    #     default="minigrid_custom_20240723/iter_1000_steps",
    # )
    # parser.add_argument("--render", action="store_true", help="render trained models")
    # args = parser.parse_args()

    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)


    env = CustomEnv(render_mode="human")

    # env = ObjObsWrapper(env)
    manual_control = ManualControl(env)
    manual_control.start()
    # asyncio.run(manual_control.start())

    # ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # # add the experiment time stamp
    # ppo = ppo.load(f"models/ppo/{args.load_model}", env=env)

    

    # number_of_episodes = 5
    # for i in range(number_of_episodes):
    #     obs, info = env.reset()
    #     score = 0
    #     done = False
    #     while(not done):
    #         action, _state = ppo.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         score += reward
    #         print(f'Action: {action}, Reward: {reward}, Score: {score}, Terminated: {terminated}')

    #         if terminated or truncated:
    #             print(f"Test score: {score}")
    #             done = True
            # await asyncio.sleep(0)

    env.close()


# impliments the manual control class without the class
def step(env, action: Actions):
    seed = None
    _, reward, terminated, truncated, _ = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if terminated:
        print("terminated!")
        reset(env, seed)
    elif truncated:
        print("truncated!")
        reset(env, seed)
    else:
        env.render()

def reset(env, seed=None):
    env.reset(seed=seed)
    env.render()

def key_handler(env, event):
    key: str = event.key
    print("pressed", key)

    if key == "escape":
        env.close()
        return
    if key == "backspace":
        reset(env)
        return

    key_to_action = {
        "left": Actions.left,
        "right": Actions.right,
        "up": Actions.forward,
        "space": Actions.toggle,
        "pageup": Actions.pickup,
        "pagedown": Actions.drop,
        "tab": Actions.pickup,
        "left shift": Actions.drop,
        "enter": Actions.done,
    }
    if key in key_to_action.keys():
        action = key_to_action[key]
        step(env, action)
    else:
        print(key)
            
            
# async def main():
#     # Initialize Pygame
#     pygame.init()

#     env = CustomEnv(render_mode="human")
#     reset(env)
    
#     closed = False
#     while not closed:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 env.close()
#                 break
#             if event.type == pygame.KEYDOWN:
#                 event.key = pygame.key.name(int(event.key))
#                 key_handler(env, event)
#         await asyncio.sleep(0)
    
#     env.close()


if __name__ == "__main__":
    # asyncio.run(main())
    main()