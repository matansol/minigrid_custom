import argparse
from datetime import datetime
from time import time

import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces

import wandb
from wandb.integration.sb3 import WandbCallback
import random

from minigrid_custom_env import CustomEnv
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

class ObjObsWrapper(ObservationWrapper):
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
                # "step_count": Box(low=0, high=env.max_steps+1, shape=(1,), dtype=np.float32),
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


class ObjEnvExtractor(BaseFeaturesExtractor):
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

class ObjEnvExtractorBig(BaseFeaturesExtractor):
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
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument(
        "--load_model",
        # default="minigrid_hard_20241010/iter_1000000_steps",
        # default="minigrid_easy7_20241030/iter_300000_steps",
    )
    parser.add_argument("--render", action="store_true", help="render trained models")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # parser.add_argument("--model", type=str, default="ppo", help="what model to train")
    args = parser.parse_args()

    policy_kwargs = dict(features_extractor_class=ObjEnvExtractor) # ObjEnvExtractorBig)
    set_random_seed(args.seed)

    # def linear_schedule(initial_value):
    #     def schedule(progress_remaining):
    #         return progress_remaining * initial_value
    #     return schedule

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d")
    # stamp = "20240717" # the date of the last model training
    env_type = 'easy' # 'hard'
    hard_env = True if env_type == 'hard' else False
    max_steps = 300
    colors_rewards = {'red': -0.1, 'green': 4, 'blue': -0.1}
    lava_cost = -4
    grid_size = 8
    agent_view_size = 7

    if args.train:
        device = "cuda" if th.cuda.is_available() else "cpu"

        env = CustomEnv(
                grid_size=grid_size,
                render_mode='rgb_array',
                max_steps=max_steps,
                highlight=True,
                step_cost=0.2,
                num_objects=4,
                lava_cells=3,
                train_env=True,
                image_full_view=False,
                agent_view_size=agent_view_size,
                colors_rewards=colors_rewards
            )
        env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=lava_cost)
        env = Monitor(env)  # Add Monitor for logging
        # Access attributes from the underlying environment
        step_cost = env.step_cost  # Access 'step_cost' from the first environment
        preference_vector = [colors_rewards['red'], colors_rewards['green'], colors_rewards['blue'], lava_cost, step_cost]
        pref_str = ",".join([str(i) for i in preference_vector])
        save_name = pref_str + f"Steps{max_steps}Grid{grid_size}_{stamp}"

        # Set up callbacks
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f'./models/{save_name}/',
            log_path='./logs/eval_logs/',
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        wandb.init(
            project="minigrid_custom",
            config={
                "algorithm": "PPO",
                "max_steps": max_steps,
                "preference_vector": preference_vector,
            },
            name=f"grid{grid_size}_view{agent_view_size}_{preference_vector}",
            sync_tensorboard=True,  # Sync tensorboard logs
            settings=wandb.Settings(symlink=False),
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=100000,
            model_save_path=f"./models/wandb_models/{pref_str}",  # Where to save models
            verbose=2,
        )

        # Define the PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.001,
            ent_coef=0.02,
            n_steps=2048,
            batch_size=32,
            clip_range=0.2,
            gamma = 0.8,
            # epochs=3,
            device=device
        )

        # Start training
        print(next(model.policy.parameters()).device)  # Ensure using GPU, should print cuda:0
        model.learn(
            5e5,
            tb_log_name=f"{stamp}",
            callback=[eval_callback]#, wandb_callback],
        )

        # Save the model and VecNormalize statistics
        # model.save(f"./models/{save_name}_ppo_model")
        # env.save(f"./models/{save_name}_vecnormalize.pkl")
    else:
        if args.render:
            env = CustomEnv(grid_size=grid_size, agent_view_size=agent_view_size, difficult_grid=hard_env, render_mode='human', image_full_view=False, lava_cells=4,
                            num_objects=3, train_env=False, max_steps=100, colors_rewards=colors_rewards, highlight=True, partial_obs=True)
        else:
            env = CustomEnv(grid_size=grid_size, difficult_grid=hard_env, agent_view_size=agent_view_size, image_full_view=False, lava_cells=1, num_objects=3, 
                            train_env=False, max_steps=100, highlight=True)
            
        env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-1.0)
        # env = ObjObsWrapper(env)

        if args.model == "ppo":
            ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

        # add the experiment time stamp
            model = ppo.load(f"models/{args.load_model}", env=env)
        else:
            dqn = DQN("multiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
            model = dqn.load(f"models/{args.load_model}", env=env)

        number_of_episodes = 5
        for i in range(number_of_episodes):
            obs, info = env.reset()
            score = 0
            done = False
            while(not done):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                score += reward
                # print(f'Action: {action}, Reward: {reward}, Score: {score}, Terminated: {terminated}')

                if terminated or truncated:
                    print(f"Test score: {score}")
                    done = True

        env.close()


if __name__ == "__main__":
    main()
