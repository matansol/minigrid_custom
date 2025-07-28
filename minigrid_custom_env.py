from __future__ import annotations

from minigrid.core.constants import *
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Goal, Door, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS, IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Point, WorldObj


import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ObservationWrapper
from gymnasium import spaces
import torch as th
import torch.nn as nn


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# import matplotlib.pyplot as plt
import random
# import pygame
import time
import numpy as np

basic_color_rewards  = {
                'red': -1,
                'green': 2,
                'blue': 4,
            }

MAX_STEPS = 1000
actions_translation = {0: 'turn left', 1: 'turn right', 2: 'move forward', 3: 'pickup', 'turn left': 0, 'turn right': 1, 'move forward': 2, 'forward': 2, 'pickup':3}

class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        size = env.observation_space['image'].shape[0]
        if False: #self.env.step_count_observation:
            self.observation_space = Dict(
                {
                    "image": Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
                    "step_count": Box(low=0, high=MAX_STEPS, shape=(1,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = Dict(
                {
                    "image": Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
                }
            )

    def observation(self, obs):
        if self.env.step_count_observation:
            wrapped_obs = {
                "image": obs["image"],
                "step_count": np.array([obs["step_count"]]),
            }
        else:
            wrapped_obs = {
                "image": obs["image"],
            }

        return wrapped_obs

class ObjObsWrapperImageOnly(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space =  Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs,
        


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
    

class CustomEnv(MiniGridEnv):
    def __init__(
        self,
        grid_size=8,
        agent_start_pos=(1, 1),
        agent_start_dir: int = 0, # 0: right, 1: down, 2: left, 3: up
        max_steps: int = 100, 
        change_reward: bool = False,
        num_objects: int = 6,
        difficult_grid: bool = False,
        train_env: bool = False,
        # unique_env: int = 0,
        num_lava_cells: int = 2,
        step_cost: float = 0.1,
        image_full_view: bool = False,
        width: int | None = None,
        height: int | None = None,
        see_through_walls: bool = True,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        color_rewards: dict = basic_color_rewards, # all colors have the same reward = 2
        partial_obs: bool = False,
        step_count_observation: bool = False,
        lava_panishment: float = -3,
        small_actions_space: bool = False,
        # from_unique_env: bool = True, # is the env should be one of the unique envs
        # simillar_env_from_near_objects: bool = True,
        # lava_reward: int = 0,
        **kwargs,
    ):
        # self.simillar_env_from_near_objects = simillar_env_from_near_objects
        self.agent_start_pos = agent_start_pos
        self.goal_pos = (width - 2, height -2) if (width is not None and height is not None) else (grid_size - 2, grid_size - 2)
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.agent_pos = agent_start_pos
        self.image_full_view = image_full_view
        self.partial_obs = partial_obs
        self.unique_env = 0
        self.step_count_observation = step_count_observation
        self.step_cost = step_cost
        self.lava_panishment = lava_panishment
        self.from_unique_env = True  # from_unique_env
        if not highlight:
            self.highlight = not image_full_view

        if max_steps is None:
            max_steps = size * size 
        self.train_env = train_env
        # if self.train_env:
        #     max_steps = size * size * 5
        self.num_objects = num_objects
        self.difficult_grid = difficult_grid
        self.num_lava_cells = num_lava_cells
        # self.lava_reward = lava_reward
        self.current_state = None
        self.took_key = None
        self.step_count = 0
        self.initial_balls = [] # list of objects on the grid in the format (x, y, color, reward when picked)
        self.lava_cells = [] # list of lava cells on the grid in the format (x, y)

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        if small_actions_space:
            self.action_space = spaces.Discrete(4)
        else:
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
                # "direction": spaces.Discrete(4),
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

        # # Current position and direction of the agent
        # self.agent_pos: np.ndarray | tuple[int, int] = None
        # self.agent_dir: int = 0

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        # Define rewards for each color
        self.color_rewards = color_rewards

        
    @staticmethod
    def is_illegal_move(action, last_obs, obs, agent_pos_befor, agent_pos):
        if action <= 1: # turn is always legal
            return False
        if action == 2 and agent_pos_befor == agent_pos:
            return True
        if action > 2 and np.array_equal(obs['image'], last_obs['image']):
            return True
        return False

    def reset(self, **kwargs):
        self.unique_env = 0
        self.ep_score = 0
        similarity_level = kwargs.get('simillarity_level', 1)
        if not 'optional_unique_env' in kwargs:
            kwargs['optional_unique_env'] = list(range(1,19))
        self.optional_unique_env = kwargs['optional_unique_env']
        if 'from_unique_env' in kwargs:
            self.from_unique_env = kwargs['from_unique_env']
        if 'unique_env' in kwargs:
            if self.train_env:
                self.unique_env = kwargs['unique_env']
            else:
                self.unique_env = kwargs['unique_env'] if kwargs['unique_env'] in self.optional_unique_env else random.choice(self.optional_unique_env)

        self.on_board_objects = 0
        self.step_count = 0
        self.took_key = False
        self.current_state = {}
        self.initial_balls = []
        self.lava_cells = []
        
        self._place_initial_objects(similarity_level, kwargs)
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
        # return "go to the red box" # just for the it to run
    
        # color_rank = ' red > green > blue' if not self.change_reward else ' blue > green > red'
        # return "Collect as many balls as possible, colors rank: " + color_rank
        
    def create_lava_locations(self, other_lava_cells, noise_factor=0):
            for l_cell in other_lava_cells:
                dx = random.randint(-noise_factor, noise_factor)
                dy = random.randint(-noise_factor, noise_factor)
                new_x = max(min(l_cell[0] + dx, self.width - 2), 1)
                new_y = max(min(l_cell[1] + dy, self.height - 2), 1)
                if (new_x, new_y) == self.goal_pos or (new_x, new_y) == self.agent_start_pos:
                    continue
                self.lava_cells.append((new_x, new_y))

    def generate_init_ball(self):
        x = random.randint(1, self.width - 2)
        y = random.randint(1, self.height - 2)
        if (x, y) == self.goal_pos or (x, y) == self.agent_start_pos:
            return self.generate_init_ball()
        color = random.choice(['red', 'blue', 'green'])
        return (x, y, color, self.color_rewards[color])
    
    def _place_initial_objects(self, simillarity_level, kwargs):
        """
        Create the initial balls list based on the simillarity level and the other initial_balls passed in the kwargs.
        simillarity_level (group): 0  - no demonstration
        simillarity_level (group): 1 - same baord as last one
        simillarity_level (group): 2 - lava the same, balls close to old locations   
        simillarity_level (group): 3 - infront objects (creating a board with the objects that the agent saw in the feedback actions 
        simillarity_level (group): 4+ - new random board from the optinal env boards
        """
        
        unique_env = kwargs.get('unique_env', 0)
        board_seen = kwargs.get('board_seen', [])

        added_lava = False

        # if (not self.train_env) or (random.random() < 0.5):
        if self.from_unique_env: # create a random unique env
            if unique_env == 0 or (unique_env not in self.optional_unique_env):
                unique_env = random.choice(self.optional_unique_env)
                c = 0
                while unique_env in board_seen and c < 10:
                    unique_env = random.choice(self.optional_unique_env)
                    c += 1
            board_seen.append(unique_env)
            
            self._gen_unique_grid(self.width, self.height, unique_env)
            added_lava = True          

        elif simillarity_level == 1 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list):
            if "other_lava_cells" in kwargs: 
                self.create_lava_locations(kwargs['other_lava_cells'], 0)
                added_lava = True
            self.initial_balls = kwargs['initial_balls'] # all balls in the same locations
        
            
        elif simillarity_level == 2 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list):
            if "other_lava_cells" in kwargs: 
                self.create_lava_locations(kwargs['other_lava_cells'], 0)
                added_lava = True

            base_balls = kwargs['initial_balls']
            # all noise in the balls placed on the grid
            for ball in base_balls:
                noise_factor = 1
                c = 0
                while c < 10:
                    c += 1
                    dx = random.randint(-noise_factor, noise_factor)
                    dy = random.randint(-noise_factor, noise_factor)
                    x = max(min(ball[0] + dx, self.width - 2), 1)
                    y = max(min(ball[1] + dy, self.height - 2), 1)
                    if (x, y) == self.goal_pos or (x, y) == self.agent_start_pos:
                        continue
                    if (x, y) not in self.lava_cells:
                        break
                    
                b_color = ball[2]
                self.initial_balls.append((x, y, b_color, self.color_rewards[b_color])) # (x, y, color, reward when picked)
            
        # elif simillarity_level == 3 and 'infront_objects' in kwargs:  
            # infront_objects = kwargs['infront_objects'][1]
            # infront_base_objects = kwargs['infront_objects'][0]
            # infront_feedback_objects = kwargs['infront_objects'][2]

            # combine = [obj for obj in infront_objects + infront_base_objects + infront_feedback_objects if IDX_TO_OBJECT[obj[0]] in ('lava', 'ball')]
            # if combine:
            #     self._place_infront_objects(infront_objects+infront_base_objects+infront_feedback_objects, 3)
            # else:
            #     print("infront objects is empty, not placing any objects")
            #     self.from_unique_env = True
        
        elif simillarity_level == 3:
            added_lava = True
            self._gen_unique_grid(self.width, self.height, unique_env)
            board_seen.append(unique_env)
        
        elif simillarity_level == 4:
            old_optinal_envs = kwargs.get('old_optional_envs', list(range(1,19)))
            new_optinal_envs = self.optional_unique_env
            optinal_envs = [x for x in old_optinal_envs if (x in new_optinal_envs and x not in board_seen)]
            unique_env = random.choice(optinal_envs)
            added_lava = True
            self._gen_unique_grid(self.width, self.height, unique_env)
            board_seen.append(unique_env)

        else: # random balls with random colors
            for i in range(self.num_objects):
                color = random.choice(list(self.color_rewards.keys()))
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == self.goal_pos or (x, y) == self.agent_start_pos:
                    continue
                self.initial_balls.append((x, y, color, self.color_rewards[color]))
            
        if not added_lava: # create random laval cells
            for i in range(self.num_lava_cells):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == self.goal_pos or (x, y) == self.agent_start_pos:
                    continue
                self.lava_cells.append((x,y))

    def _place_infront_objects(self, infront_objects, number_of_objects=5):
        """
        Make an env with the same objects as the ones in the infront_objects list
        """
        infront_objects = list(set(infront_objects)) # remove duplicates
        for obj in infront_objects:
            for i in range(number_of_objects):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                if IDX_TO_OBJECT[obj[0]] == 'lava':
                    self.lava_cells.append((x,y))

                elif IDX_TO_OBJECT[obj[0]] == 'ball':
                    color = IDX_TO_COLOR[obj[1]]
                    self.initial_balls.append((x, y, color, self.color_rewards[color]))
        
                    
    def _gen_grid(self, width, height, **kwargs):
        self.mission = "" #self._gen_mission()

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
        self.goal_position = self.goal_pos        
            
        # Place a ball in some cells
        for ball_info in self.initial_balls:
            x_loc = ball_info[0]
            y_loc = ball_info[1]
            color = ball_info[2]
            self.put_obj(Ball(color), x_loc, y_loc)
        
        # place lava cells
        for l_cell in self.lava_cells:
            self.put_obj(Lava(), l_cell[0], l_cell[1])
            
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)


    def _gen_unique_grid(self, width, height, unique_env=1):
        # Define balls and lava lists for each unique_env
        print(f"generate unique env number-{unique_env}")
        balls_list = []
        lava_list = []

        if unique_env == 1:
            balls_list = [(width - 2, 1, 'blue'), (width - 4, 3, 'green')]
            lava_list = [(width - 3, 2), (width - 3, 1)]

        if unique_env == 2:
            balls_list = [(width - 2, 1, 'green'), (1, 5, 'green'), (2, 6, 'green'), (2, 5, 'red'), (3, 4, 'blue')]
            lava_list = []

        if unique_env == 3:
            balls_list = [(5, 1, 'red'), (1, 3, 'blue'), (6, 3, 'blue'), (1, 4, 'green'), (6, 4, 'green')]
            lava_list = [(1, 2), (2, 2), (6, 2), (5, 2)]

        if unique_env == 4:
            balls_list = [(1, 5, 'green'), (1, 4, 'red'), (4, 4, 'blue'), (6, 5, 'green')]
            lava_list = [(1, 3), (2, 3), (3, 3)]

        if unique_env == 5:
            balls_list = [(1, 3, 'blue'), (6, 2, 'blue'), (6, 4, 'green'), (5, 1, 'green')]
            lava_list = [(6, 1), (5, 2)]

        if unique_env == 6:
            balls_list = [(4, 3, 'red'), (5, 2, 'red'), (2, 6, 'red'), (3, 6, 'blue'), (1, 4, 'green')]
            lava_list = [(3, 3), (1, 3)]

        if unique_env == 7:
            balls_list = [(4, 4, 'blue'), (5, 2, 'green')]
            lava_list = [(3,3), (3,4), (3,5), (5,5), (4,5), (5,3)]
            # for i in range(3, 6):
            #     for j in range(3, 6):
            #         if (i == 4 and j == 4) or (i == 5 and j == 4) or (i == 4 and j == 3):
            #             continue
            #     lava_list.append((i, j))

        if unique_env == 8:
            balls_list = [(5, 5, 'red'), (6, 4, 'green'), (3, 6, 'blue'), (2, 4, 'blue')]
            lava_list = []
            self.grid.vert_wall(4, 1, 3)

        if unique_env == 9:
            balls_list = [(4, 1, 'blue'), (6, 1, 'blue'), (2, 4, 'green'), (4, 4, 'green')]
            lava_list = [(5, 1), (3, 1)]

        if unique_env == 10:
            balls_list = [(2, 2, 'red'), (5, 6, 'red'), (3, 5, 'blue'), (5, 2, 'blue')]
            lava_list = [(4, 4), (3, 4), (5, 4), (3, 2)]

        if unique_env == 11:
            balls_list = [(2, 6, 'green'), (5, 2, 'red'), (6, 4, 'blue'), (3, 5, 'green'), (4, 6, 'red')]
            lava_list = [(3, 3), (4, 3), (5, 3)]

        if unique_env == 12:
            balls_list = [(2, 2, 'green'), (6, 5, 'blue'), (2, 5, 'green'), (6, 2, 'red')]
            lava_list = [(5, 2), (2, 4)]

        if unique_env == 13:
            balls_list = [(2, 2, 'green'), (4, 4, 'blue'), (5, 2, 'green'), (6, 3, 'blue'), (5, 6, 'red')]
            lava_list = [(3, 3), (4, 3), (2, 4), (6, 5)]

        if unique_env == 14:
            balls_list = [(3, 3, 'red'), (6, 4, 'green'), (3, 5, 'green'), (6, 2, 'blue')]
            lava_list = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)]:
                x, y = 6 + dx, 2 + dy
                if 1 <= x <= width - 2 and 1 <= y <= height - 2:
                    lava_list.append((x, y))
            lava_list.extend([(5, 3), (4, 4)])

        if unique_env == 15:
            balls_list = [(3, 1, 'red'), (3, 3, 'blue'), (2, 5, 'green'), (5, 5, 'red')]
            lava_list = [(1, 2), (2, 2), (3, 2)]

        if unique_env == 16:
            balls_list = [(4, 1, 'blue'), (2, 3, 'green'), (6, 3, 'green'), (5, 5, 'green')]
            lava_list = [(4, 3), (5,3), (3, 2), (3, 1)]
        
        if unique_env == 17:
            balls_list = [(4, 1, 'red'), (6, 1, 'blue'), (3, 2, 'green'), (6, 4, 'blue')]
            lava_list = [(6, 2), (5, 2)]

        if unique_env == 18:
            balls_list = [(1, 4, 'red'), (1, 6, 'blue'), (2, 3, 'green'), (4, 6, 'blue')]
            lava_list = [(2, 6), (2, 5)]

        # Place balls and lava on the grid
        for ball_info in balls_list:
            x_loc = ball_info[0]
            y_loc = ball_info[1]
            color = ball_info[2]
            # self.put_obj(Ball(color), x_loc, y_loc)
            self.initial_balls.append((x_loc, y_loc, color, self.color_rewards[color]))
        for lava_cell in lava_list:
            x_loc = lava_cell[0]
            y_loc = lava_cell[1]
            # self.put_obj(Lava(), x_loc, y_loc)
            self.lava_cells.append((x_loc, y_loc))

        # Place a wall in the middle of the grid
        # self.put_obj(Goal(), width - 2, height - 2)

        # self.board_seen.append(unique_env)


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
        # for _ in range(self.num_objects):
        #     x_loc = self._rand_int(1, width - 2)
        #     y_loc = self._rand_int(1, height - 2)
        #     if (x_loc, y_loc) == (width - 2, height - 2) or (x_loc, y_loc) == (1, 1):
        #         continue
        #     if x_loc == splitIdx or self.grid.get(x_loc, y_loc) is not None:
        #         continue
        #     color = random.choice(list(self.color_rewards.keys()))
        #     self.put_obj(Ball(color), x_loc, y_loc)
        #     self.initial_balls.append((x_loc, y_loc, color, self.color_rewards[color]))
        for ball_info in self.initial_balls:
            x_loc = ball_info[0]
            y_loc = ball_info[1]
            color = ball_info[2]
            self.put_obj(Ball(color), x_loc, y_loc)
        
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
            self.on_board_objects -= 1
    
    def step(self, action):
        if self.current_state:
            last_obs = self.current_state.copy() # save the last observation
            agent_pos_befor = self.agent_pos
        obs, reward, terminated, truncated, info = super().step(action)
        if self.step_count_observation:
            obs['step_count'] = self.step_count
        if self.image_full_view:
            obs['image'] = self.grid.encode() # get the full grid image
            self.put_agent_in_obs(obs)
        self.current_state = obs

        if last_obs:
            if self.is_illegal_move(action, last_obs, obs, agent_pos_befor, self.agent_pos):
                reward -= 1

        # Check if the agent picked up a ball
        if self.carrying:
            hold_obj = self.carrying
            if hold_obj.type == "ball":
                ball_color = self.carrying.color
                reward += self.color_rewards.get(ball_color, 0)
                self.carrying = None
                self.on_board_objects -= 1
        
        # got to the right bottom corner - the goal
        if self.agent_pos == (self.grid.width - 2, self.grid.height - 2) and self.train_env:
            reward += 5

        # hit a lava cell
        # if self.agent_pos in self.lava_cells: 
        #     # terminated = False
        #     reward += self.lava_panishment

        if truncated:
            terminated = True
            print(f"reached max steps={self.max_steps}")
            if self.train_env:
                reward -= 5

        dis_from_goal = np.abs(self.agent_pos[0] - (self.grid.width - 2)) + np.abs(self.agent_pos[1] - (self.grid.height - 2))
        reward -=  self.step_cost # * dis_from_goal #the aproximation of number of steps to the goal
        self.ep_score += reward
        return obs, round(reward,1), terminated, truncated, info

    def find_optimal_path(self):
        points = [(1,1,0)] + self.initial_balls
        points = {i: p for i, p in enumerate(points) if p[2] >= 0}
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
        for s in range(len(points)-1):
            min_steps = np.min(matrix[curent_pos])
            min_arg = np.argmin(matrix[curent_pos])
            total_steps += min_steps
            total_reward += points[min_arg][2]
            matrix[:, curent_pos] = np.inf
            curent_pos = min_arg

        p1 = points[curent_pos]
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

    def get_full_image(self):
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
