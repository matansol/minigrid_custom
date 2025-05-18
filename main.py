import matplotlib
matplotlib.use('TkAgg')  # Use interactive TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import tkinter

# from IPython.display import HTML
# from IPython import display
# from IPython.display import clear_output

# from gym.wrappers.record_video import RecordVideo
from minigrid.wrappers import *

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath
from minigrid.core.actions import Actions
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from minigrid_custom_train import UpgradedObjEnvExtractor


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam

from minigrid_custom_env import *
from minigrid_custom_train import *
from dpu_clf import *


plt.rcParams['figure.figsize'] = (6.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def create_env_from_model(model_name, grid_size, agent_view_size, max_steps, highlight, num_objects, lava_cells, train_env=True, image_full_view=False):
    env_info = model_name.split('S')[0].split(',')
    lava_cost = float(env_info[3])
    step_cost = float(env_info[4])
    colors_rewards = {'red':float(env_info[0]), 'green': float(env_info[1]), 'blue': float(env_info[2])}
    step_count = True if 'Step_Count' in model_name else False
    base_env =  CustomEnv(
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
            color_rewards=colors_rewards,
            step_count_observation=step_count,
            # small_actions_space=True, 
        )
    base_env = NoDeath(ObjObsWrapper(base_env), no_death_types=('lava',), death_cost=lava_cost)
    return base_env

import matplotlib.pyplot as plt
from PIL import Image
import imageio
import base64
from io import BytesIO
import numpy as np
import copy

#test the environment

def plot_state(env):
    img = env.render()
    print(type(img), img.shape)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
# env = CustomEnv(size = 8, render_mode='rgb_array', difficult_grid=False, agent_pov=True, step_count_observation=False)
# env = ImgObsWrapper(ObjObsWrapper(env))
grid_size = 8
max_steps = 300
agent_view_size = 7
lava_cost = -5
colors_rewards = {'red':3, 'green': 1, 'blue': 0}
kwargs = {
    # "initial_balls": initial_balls,
    # "other_lava_cells": other_lava_cells,
    "grid_size": grid_size,
    "render_mode": 'rgb_array',
    "max_steps": max_steps,
    "highlight": True,
    "unique_env": 0,
    "step_count_observation": False,
    "num_objects": 4,
    "train_env": True,
    "image_full_view": False,
    "agent_view_size": agent_view_size,
    "colors_rewards": colors_rewards,
    "simillarity_level": 3,
    "num_lava_cells": 4,
    "unique_env": 7
}

env = CustomEnv(**kwargs)
        
# env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=lava_cost)
# env = ObjObsWrapper(env)


current_obs = env.reset()
current_obs = current_obs[0]
plot_state(env)

print("Tkinter is available")
plt.savefig("output.png")

