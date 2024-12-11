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
        print(size)
        self.observation_space = Dict(
            {
                "image": Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
                #"mission": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
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
        #         self.color_one_hot_dict["red"],
        #         self.obj_one_hot_dict["ball"],
        #     ]
        # )

        wrapped_obs = {
            "image": obs["image"],
            # "mission": mission_array,
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

    def linear_schedule(initial_value):
        def schedule(progress_remaining):
            return progress_remaining * initial_value
        return schedule

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d")
    # stamp = "20240717" # the date of the last model training
    env_type = 'easy' # 'hard'
    hard_env = True if env_type == 'hard' else False
    max_steps = 200
    colors_rewards = {'red': -2.0, 'green': 2, 'blue': 2}
    grid_size = 8
    agent_view_size = 7
    if args.train:
        env = CustomEnv(grid_size=grid_size, render_mode='rgb_array', difficult_grid=hard_env, max_steps=max_steps, highlight=True,
                        num_objects=4, lava_cells=4, train_env=True, image_full_view=False, agent_view_size=agent_view_size, colors_rewards=colors_rewards)
        
        env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-3.0)
        # env = DummyVecEnv([lambda: env])
        # env = VecNormalize(env, norm_obs=False, norm_reward=True)
        save_name = f"R{max_steps}N_LavaHate{agent_view_size}_{grid_size}_{stamp}"
        checkpoint_callback = CheckpointCallback(
            save_freq=250e3,
            save_path=f"./models/{save_name}/",
            name_prefix="iter",
        )

        # eval_callback = EvalCallback(
        #     env,
        #     best_model_save_path=f"./models/basic_L2_{env_type}{grid_size}_{stamp}/best_model/",
        #     log_path=f"./logs/minigrid_{env_type}{grid_size}_eval_logs/",
        #     eval_freq=10000,   # Evaluate every 10,000 steps
        #     deterministic=True,
        #     render=False,
        # )

        # model = PPO(
        #     "MultiInputPolicy",
        #     env,
        #     policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     tensorboard_log=f"./logs/minigrid_{env_type}_tensorboard/",
        #     learning_rate=0.01,
        #     ent_coef = 0.05,
        # )
        if args.load_model:
            model = PPO.load(f"models/{args.load_model}", env=env)
            print(f"Loaded model from {args.load_model}. Continuing training.")
        else:
            # Create a new model if no load path is provided
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                # seed=42,
                tensorboard_log=f"./logs/{save_name}/",
                learning_rate=0.01,
                ent_coef=0.3,
                n_steps=128,
                # batch_size=32,
                # clip_range=0.3,
                # vf_coef=0.7,
                # gradient_clip=0.6,
                # linear_schedule=linear_schedule(0.001),   
            )
        
        model.learn(
            1e6,
            tb_log_name=f"{stamp}",
            callback=checkpoint_callback,
        )
    else:
        if args.render:
            env = CustomEnv(grid_size=grid_size, agent_view_size=agent_view_size, difficult_grid=hard_env, render_mode='human', image_full_view=False, lava_cells=4,
                            num_objects=3, train_env=False, max_steps=100, colors_rewards=colors_rewards, highlight=True, partial_obs=True)
        else:
            env = CustomEnv(grid_size=grid_size, difficult_grid=hard_env, agent_view_size=agent_view_size, image_full_view=False, lava_cells=1, num_objects=3, 
                            train_env=False, max_steps=100, highlight=True)
            
        env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-1.0)
        env = ObjObsWrapper(env)

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
