import argparse
from datetime import datetime
from time import time

import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from gymnasium.core import ObservationWrapper
# from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
# from gym.wrappers import TimeLimit
# from gymnasium import spaces
# from torch import optim
from typing import Dict


# import wandb
# from wandb.integration.sb3 import WandbCallback
import random

from minigrid_custom_env import CustomEnv, ObjObsWrapper, ObjEnvExtractor
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    pass

# class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        size = env.observation_space['image'].shape[0]
        print("observation size:", size)
        self.observation_space = Dict(
            {
                "image": Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
                "step_count": Box(low=0, high=env.max_steps+1, shape=(1,), dtype=np.float32),
                #"mission": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
            }
        )
        # if env.step_count_observation:
        #     print("add the step count variable to the observation")
        #     self.observation_space['step_count'] = spaces.Box(low=0, high=env.max_steps+1, shape=(1,), dtype="int")

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
        #         self.color_one_hot_dict["red"],
        #         self.obj_one_hot_dict["ball"],
        #     ]
        # )
        if self.env.step_count_observation:
            wrapped_obs = {
                "image": obs["image"],
                # "mission": mission_array,
                "step_count": np.array([obs["step_count"]]),
            }
        else:
            wrapped_obs = {
                "image": obs["image"],
            }

        return wrapped_obs


# class ObjEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        print("Observation space:", observation_space)
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
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

            elif key == "mission":
                extractors["mission"] = nn.Linear(subspace.shape[0], 32)
                total_concat_size += 32
            elif key == "step_count": 
                # Add a linear layer to process the scalar `step_count`
                extractors["step_count"] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16),  # Convert 1D input to 16 features
                    nn.ReLU(),
                    )
                total_concat_size += 16  # Update the total feature size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

# class WandbEvalCallback(BaseCallback):
    # """
    # Custom callback for logging evaluation results to wandb and saving the best model.
    # """
    # def __init__(self, eval_env, eval_freq, n_eval_episodes, wandb_run, best_model_save_path, verbose=0):
    #     super(WandbEvalCallback, self).__init__(verbose)
    #     self.eval_env = eval_env
    #     self.eval_freq = eval_freq
    #     self.n_eval_episodes = n_eval_episodes
    #     self.wandb_run = wandb_run
    #     self.best_model_save_path = best_model_save_path
    #     self.best_mean_reward = -float("inf")  # Initialize with a very low value

    # def _on_step(self) -> bool:
    #     # Perform evaluation every `eval_freq` steps
    #     if self.n_calls % self.eval_freq == 0:
    #         episode_rewards = []
    #         episode_lengths = []
    #         for _ in range(self.n_eval_episodes):
    #             obs = self.eval_env.reset()
    #             done = False
    #             step_count = 0
    #             episode_reward = 0
    #             while not done:
    #                 action, _ = self.model.predict(obs, deterministic=True)
    #                 obs, reward, terminated, truncated = self.eval_env.step(action)
    #                 episode_reward += reward
    #                 done = terminated or truncated
    #                 step_count += 1
    #             episode_lengths.append(step_count)
    #             episode_rewards.append(episode_reward)

    #         mean_reward = sum(episode_rewards) / len(episode_rewards)
    #         mean_length = sum(episode_lengths) / len(episode_lengths)
    #         print(f"Step {self.n_calls}: Mean reward: {mean_reward}, Mean length: {mean_length}")

    #         # Log to wandb
    #         self.wandb_run.log({"mean_reward": mean_reward, "mean_lenth": mean_length, "step": self.n_calls})   

    #         # Save the best model if the mean reward improves
    #         if mean_reward > self.best_mean_reward:
    #             self.best_mean_reward = mean_reward
    #             print(f"New best mean reward: {mean_reward}. Saving model...")
    #             self.model.save(f"{self.best_model_save_path}/best_model")

    #     return True


class UpgradedObjEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        self.extractors = nn.ModuleDict()
        total_feature_dim = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Get channel count correctly (usually 3 for RGB in MiniGrid)
                c, h, w = subspace.shape[::-1]  # From (H, W, C) to (C, H, W)

                self.extractors["image"] = nn.Sequential(
                            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
                            # nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            # nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            # nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Flatten()
                        )

                with th.no_grad():
                    sample = th.as_tensor(subspace.sample()[None]).float() / 255.0
                    sample = sample.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    n_flatten = self.extractors["image"](sample).shape[1]

                self.image_linear = nn.Sequential(
                    nn.Linear(n_flatten, 64),
                    nn.ReLU()
                )
                total_feature_dim += 64

            elif key == "step_count":
                self.extractors["step_count"] = nn.Sequential(
                    nn.LayerNorm(subspace.shape),
                    nn.Linear(subspace.shape[0], 16),
                    nn.ReLU()
                )
                total_feature_dim += 16

            elif key == "mission":
                self.extractors["mission"] = nn.Sequential(
                    nn.LayerNorm(subspace.shape),
                    nn.Linear(subspace.shape[0], 32),
                    nn.ReLU()
                )
                total_feature_dim += 32

        self._features_dim = total_feature_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        outputs = []
        for key, module in self.extractors.items():
            if key == "image":
                x = observations["image"].float() / 255.0
                x = x.permute(0, 3, 1, 2)  # BCHW, very important
                x = self.extractors["image"](x)
                x = self.image_linear(x)
                outputs.append(x)
            else:
                outputs.append(module(observations[key].float()))
        return th.cat(outputs, dim=1)


def create_env(grid_size, agent_view_size, max_steps, highlight, step_cost, num_objects, lava_cells, color_rewards, train_env=True, image_full_view=False, step_count_observation=False):
    env = CustomEnv(
        grid_size=grid_size,
        render_mode='rgb_array',
        max_steps=max_steps,
        highlight=highlight,
        step_cost=step_cost,
        num_objects=num_objects,
        lava_cells=lava_cells,
        train_env=train_env,
        image_full_view=image_full_view,
        agent_view_size=agent_view_size,
        color_rewards=color_rewards,
        step_count_observation=step_count_observation
    )
    env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-1.0)
    # env = TimeLimit(env, max_episode_steps=max_steps)
    env = Monitor(env)  # Add Monitor for logging
    return env

import os
def main(**kwargs):
    os.makedirs("./logs/eval_logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument(
        "--load_model", type=str, default=None, help="load a trained model"
    )

    args = parser.parse_args()
    # parser.add_argument("--render", action="store_true", help="render trained models")
    # parser.add_argument("--seed", type=int, default=42, help="random seed")
    # parser.add_argument("--model", type=str, default="ppo", help="what model to train")
    # args = parser.parse_args()

    policy_kwargs = dict(features_extractor_class=ObjEnvExtractor,) # )UpgradedObjEnvExtractor
    # set_random_seed(args.seed)

    # def linear_schedule(initial_value):
    #     def schedule(progress_remaining):
    #         return progress_remaining * initial_value
    #     return schedule

    # if args.load_model:
    #     print(f"Loading model from {args.load_model}")
    #     model_name = args.load_model
    #     env_info = model_name.split('/')[1].split('S')[0].split(',')
    #     lava_cost = -3 #float(env_info[3])
    #     step_cost = float(env_info[4])
    #     step_cost = 0.1 
    #     colors_rewards = {'red':float(env_info[0]), 'green': float(env_info[1]), 'blue': float(env_info[2])}

    # Models that need to be:
    # AllColors LL - 2,2,4,0.2,0.1
    # AllColors LH - 2,2,4,-4,0.1
    # OnlyBlue LH - -0.5, -0.5,4,-4,0.1
    # OnlyBlue LL - -0.5, -0.5,4, 0.2, 0.1
    # NoRed LL - -0.5, 3, 4, -0.5, 0.1
    # NoRed LH - -0.5, 3,4, -3, 0.1
    # NoGreen LL - 3,-0.5,4,0, 0.1
    # NoGreen LH - 3, -0.5, 4, -3, 0.1
    
    colors_options = [
        # AllColors LL  - 0
        {'balls': {'red': 2, 'green': 2, 'blue': 4}, 'lava': 0.2, 'step': 0.2},
        # AllColors LH  - 1
        {'balls': {'red': 2, 'green': 2, 'blue': 4}, 'lava': -4, 'step': 0.1},
        # OnlyBlue LH  - 2
        {'balls': {'red': -0.5, 'green': -0.5, 'blue': 4}, 'lava': -4, 'step': 0.1},
        # OnlyBlue LL  - 3
        {'balls': {'red': -0.5, 'green': -0.5, 'blue': 4}, 'lava': 0.2, 'step': 0.1},
        # NoRed LL  - 4
        {'balls': {'red': -0.5, 'green': 3, 'blue': 4}, 'lava': -0.5, 'step': 0.1},
        # NoRed LH  - 5
        {'balls': {'red': -0.5, 'green': 3, 'blue': 4}, 'lava': -3, 'step': 0.1},
        # NoGreen LL  - 6   
        {'balls': {'red': 2, 'green': -0.5, 'blue': 4}, 'lava': 0.2, 'step': 0.1},
        # NoGreen LH - 7
        {'balls': {'red': 3, 'green': -0.5, 'blue': 4}, 'lava': -3, 'step': 0.1},
        # OnlyGreen LH - 8
        {'balls': {'red': -0.5, 'green': 4, 'blue': -0.5}, 'lava': -3, 'step': 0.1},
        # OnlyGreen LL - 9
        {'balls': {'red': -0.5, 'green': 4, 'blue': -0.5}, 'lava': 0.2, 'step': 0.1},
    ] # size = 10
                        

    option = 7
    # else:
    lava_cost = colors_options[option]['lava']  
    step_cost = colors_options[option]['step']
    colors_rewards = colors_options[option]['balls']

    step_count_observation = False
    # lava_cost = -3 # override to make the agents avoid lava -------------------------------------

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d")
    # stamp = "20240717" # the date of the last model training
    env_type = 'easy' # 'hard'
    hard_env = True if env_type == 'hard' else False
    max_steps = 50
    grid_size = 8
    agent_view_size = 7
    num_lava_cell = 4
    num_balls = 6

    # if args.train:
    device = "cuda" if th.cuda.is_available() else "cpu"

    train_env = CustomEnv(
            grid_size=grid_size,
            render_mode='rgb_array',
            max_steps=max_steps,
            highlight=True,
            step_cost=step_cost,
            num_objects=num_balls,
            lava_cells=num_lava_cell,
            train_env=True,
            image_full_view=False,
            agent_view_size=agent_view_size,
            color_rewards=colors_rewards,
            step_count_observation=step_count_observation, # Add step count to observation
            lava_panishment=lava_cost,
        )
    train_env = NoDeath(ObjObsWrapper(train_env), no_death_types=('lava',), death_cost=lava_cost)
    train_env = Monitor(train_env)  # Add Monitor for logging

    # Wrap training env
    train_env = DummyVecEnv([lambda: train_env])

    # Access attributes from the underlying environment
    preference_vector = [colors_rewards['red'], colors_rewards['green'], colors_rewards['blue'], lava_cost, step_cost]
    pref_str = ",".join([str(i) for i in preference_vector])
    save_name = pref_str + f"Steps{max_steps}Grid{grid_size}_{stamp}"

    if step_count_observation:
        save_name = save_name[:-9] + "_Step_Count" + save_name[-9:]

    print("start training model with name:", save_name)


    # Set up callbacks
    # eval_env = DummyVecEnv([lambda: create_env(grid_size=grid_size,
    #                  agent_view_size=agent_view_size,
    #                  max_steps=max_steps,
    #                  highlight=True,
    #                  step_cost=step_cost,
    #                  num_objects=num_balls,
    #                  lava_cells=num_lava_cell,
    #                  train_env=True,
    #                  image_full_view=False,
    #                  color_rewards=colors_rewards,
    #                  step_count_observation=step_count_observation, # add step count to observation
    #                  )])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = CustomEnv(
            grid_size=grid_size,
            render_mode='rgb_array',
            max_steps=max_steps,
            highlight=True,
            step_cost=step_cost,
            num_objects=num_balls,
            lava_cells=num_lava_cell,
            train_env=True,
            image_full_view=False,
            agent_view_size=agent_view_size,
            color_rewards=colors_rewards,
            step_count_observation=step_count_observation, # Add step count to observation
            lava_panishment=lava_cost,
        )
    eval_env = NoDeath(ObjObsWrapper(eval_env), no_death_types=('lava',), death_cost=lava_cost)

    # eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{save_name}/',
        log_path=None,  # Do not write evaluation logs
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # wandb.init(
    #     project="minigrid_custom",
    #     config={
    #         "algorithm": "PPO",
    #         "max_steps": max_steps,
    #         "preference_vector": preference_vector,
    #     },
    #     name=f"grid{grid_size}_view{agent_view_size}_{preference_vector}",
    #     sync_tensorboard=True,  # Sync tensorboard logs
    #     settings=wandb.Settings(symlink=False),
    # )

    # wandb_eval_callback = WandbEvalCallback(
    #     eval_env=eval_env,
    #     eval_freq=10000,
    #     n_eval_episodes=5,
    #     wandb_run=wandb,
    #     best_model_save_path=f'./models/{save_name}/',
    # )

    if args.load_model:
        # Load the old model to get the parameters
        old_model = PPO.load(args.load_model, env=train_env, device=device)
        # Create a new model with the new n_steps
        model = PPO(
            "MultiInputPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            learning_rate=1e-4,
            ent_coef=1e-2,
            n_steps=64,
            batch_size=32,
            gamma=0.98,
            # gae_lambda=0.95,
            n_epochs=10,
            clip_range=0.2,
            device=device
        )
        # Copy parameters from the old model
        model.policy.load_state_dict(old_model.policy.state_dict())
    else:
        print("No model specified, creating a new PPO agent")
        model = PPO(
            "MultiInputPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,      # Lower for stability with multiple objectives
            ent_coef=0.01,           # Increase to encourage exploration
            n_steps=64,              # Increase to capture fuller episode contexts
            batch_size=32,           # Increase for better gradient estimates
            gamma=0.98,              # Higher to value future rewards more
            gae_lambda=0.95,         # Keep as is, good for advantage estimation
            n_epochs=10,             # More epochs for better learning
            clip_range=0.2,          # Add clipping to prevent too large updates
            device=device
        )

    print(f"observation state: {train_env.observation_space}")

    # Start training
    print(next(model.policy.parameters()).device)  # Ensure using GPU, should print cuda:0
    model.learn(
        2e5,
        tb_log_name=f"{stamp}",
        callback=[eval_callback]#, wandb_eval_callback]
    )

    print(f"finished training, saving the model to {save_name}")
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()


        # Save the model and VecNormalize statistics
        # model.save(f"./models/{save_name}_ppo_model")
        # env.save(f"./models/{save_name}_vecnormalize.pkl")
    # else:
    #     if args.render:
    #         env = CustomEnv(grid_size=grid_size, agent_view_size=agent_view_size, difficult_grid=hard_env, render_mode='human', image_full_view=False, lava_cells=4,
    #                         num_objects=3, train_env=False, max_steps=100, colors_rewards=colors_rewards, highlight=True, partial_obs=True)
    #     else:
    #         env = CustomEnv(grid_size=grid_size, difficult_grid=hard_env, agent_view_size=agent_view_size, image_full_view=False, lava_cells=1, num_objects=3, 
    #                         train_env=False, max_steps=100, highlight=True)
            
    #     env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-1.0)
    #     # env = ObjObsWrapper(env)

    #     if args.model == "ppo":
    #         ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    #     # add the experiment time stamp
    #         model = ppo.load(f"models/{args.load_model}", env=env)

    #     number_of_episodes = 5
    #     for i in range(number_of_episodes):
    #         obs, info = env.reset()
    #         score = 0
    #         done = False
    #         while(not done):
    #             action, _state = model.predict(obs, deterministic=True)
    #             obs, reward, terminated, truncated, info = env.step(action)
    #             score += reward
    #             # print(f'Action: {action}, Reward: {reward}, Score: {score}, Terminated: {terminated}')

    #             if terminated or truncated:
    #                 print(f"Test score: {score}")
    #                 done = True




