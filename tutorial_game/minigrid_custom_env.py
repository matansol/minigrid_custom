from __future__ import annotations

from minigrid.core.constants import *
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Goal, Door, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS, IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX, COLOR_TO_IDX


import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ObservationWrapper
from gymnasium import spaces
# import torch as th
# import torch.nn as nn


import random
import numpy as np

basic_color_rewards  = {
                'red': -1,
                'green': 2,
                'blue': 4,
            }

MAX_STEPS = 1000
class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        size = env.observation_space['image'].shape[0]
        print("observation size:", size)
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


class CustomEnv(MiniGridEnv):
    def __init__(
        self,
        grid_size=8,
        agent_start_pos=(1, 1),
        agent_start_dir: int = 0, # 0: right, 1: down, 2: left, 3: up
        max_steps: int = 70, 
        change_reward: bool = False,
        num_objects: int = 6,
        difficult_grid: bool = False,
        train_env: bool = False,
        unique_env: int = 0,
        num_lava_cells: int = 4,
        step_cost: float = 0.1,
        image_full_view: bool = False,
        width: int | None = None,
        height: int | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        color_rewards: dict = basic_color_rewards, # all colors have the same reward = 2
        partial_obs: bool = False,
        step_count_observation: bool = False,
        lava_panishment: int = -3,
        small_actions_space: bool = False,
        # set_env: bool = False, # is the env should be one of the unique envs
        simillarity_level: int = 0,
        simillar_env_from_near_objects: bool = True,
        # lava_reward: int = 0,
        **kwargs,
    ):
        self.simillar_env_from_near_objects = simillar_env_from_near_objects
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.agent_pos = agent_start_pos
        self.image_full_view = image_full_view
        self.partial_obs = partial_obs
        self.unique_env = unique_env
        self.from_unique_env = False
        self.step_count_observation = step_count_observation
        self.step_cost = step_cost
        self.lava_panishment = lava_panishment  
        self.set_env = False  
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

        

    def reset(self, **kwargs):
        if 'simillarity_level' in kwargs:
            simillarity_level = kwargs['simillarity_level']
        else:
            simillarity_level = 5
        if 'set_env' in kwargs:
            self.set_env = kwargs['set_env']
        if 'unique_env' in kwargs:
            self.unique_env = kwargs['unique_env']
        if 'from_unique_env' in kwargs:
            self.from_unique_env = kwargs['from_unique_env']
        self.on_board_objects = 0
        self.step_count = 0
        self.took_key = False
        self.current_state = {}
        self.initial_balls = []
        self.lava_cells = []
        if 'infront_objects' in kwargs:
            self._place_infront_objects(kwargs['infront_objects'], 5)
        else:
            self._place_initial_objects(simillarity_level, kwargs)

        state , info = super().reset()
        if self.image_full_view:
            state['image'] = self.grid.encode()
            self.put_agent_in_obs(state)
        self.current_state['image'] = state['image']
        if self.step_count_observation:
            self.current_state['step_count'] = self.step_count
            state['step_count'] = self.step_count
        
        return state, info
    
    def update_set_env(self, set_env: bool):
        self.set_env = set_env

    @staticmethod
    def _gen_mission():
        return ""
        # return "go to the red box" # just for the it to run, TODO: remove this
    
        # color_rank = ' red > green > blue' if not self.change_reward else ' blue > green > red'
        # return "Collect as many balls as possible, colors rank: " + color_rank

    def create_lava_locations(self, other_lava_cells, noise_factor=0):
            for l_cell in other_lava_cells:
                dx = random.randint(-noise_factor, noise_factor)
                dy = random.randint(-noise_factor, noise_factor)
                new_x = max(min(l_cell[0] + dx, self.width - 2), 1)
                new_y = max(min(l_cell[1] + dy, self.height - 2), 1)
                if (new_x, new_y) == (self.width - 2, self.height - 2) or (new_x, new_y) == (1, 1):
                    continue
                self.lava_cells.append((new_x, new_y))


    def _place_initial_objects(self, simillarity_level, kwargs):
    
        """
        Create the initial balls list based on the simillarity level and the other initial_balls passed in the kwargs.
        simillarity_level: 0  - all balls and lava in the same locations
        simillarity_level: 1 - all balls and lava in maximum 2 distance from the old locations
        simillarity_level: 2 - same balls locations but different colors, lava close to old locations 
        simillarity_level: 3 - same colors but different balls locations, lava same location 
        simillarity_level: 4+ - random balls with random colors
        """

        added_lava = False
        if simillarity_level == 0 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list):
            if "other_lava_cells" in kwargs: 
                self.create_lava_locations(kwargs['other_lava_cells'], 0)
                added_lava = True
            self.initial_balls = kwargs['initial_balls'] # all balls in the same locations
            
        elif simillarity_level == 1 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list):
            if "other_lava_cells" in kwargs: 
                self.create_lava_locations(kwargs['other_lava_cells'], 0)
                added_lava = True

            base_balls = kwargs['initial_balls']
            # all noise in the balls placed on the grid
            for ball in base_balls:
                noise_factor = 1
                dx = random.randint(-noise_factor, noise_factor)
                dy = random.randint(-noise_factor, noise_factor)
                x = max(min(ball[0] + dx, self.width - 2), 1)
                y = max(min(ball[1] + dy, self.height - 2), 1)

                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                b_color = ball[2]
                self.initial_balls.append((x, y, b_color, self.color_rewards[b_color])) # (x, y, color, reward when picked)
            
            
        elif simillarity_level == 2 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list): # 
            if "other_lava_cells" in kwargs:
                self.create_lava_locations(kwargs['other_lava_cells'], 1)
                added_lava = True

            initial_balls = kwargs["initial_balls"]
            for ball in initial_balls:
                x = ball[0]
                y = ball[1]
                color = random.choice(list(self.color_rewards.keys()))
                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                self.initial_balls.append((x, y, color, self.color_rewards[color])) # (x, y, color, reward when picked)

        
        elif simillarity_level == 3 and "initial_balls" in kwargs and isinstance(kwargs["initial_balls"], list):  # same place for the balls but different colors
            if "other_lava_cells" in kwargs:
                self.create_lava_locations(kwargs['other_lava_cells'], 0)
                added_lava = True
            
            initial_colors = [ball[2] for ball in kwargs["initial_balls"]]
            for color in initial_colors:
                # color = random.choice(initial_colors)
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                self.initial_balls.append((x, y, color, self.color_rewards[color]))


        else: # random balls with random colors
            for i in range(self.num_objects):
                color = random.choice(list(self.color_rewards.keys()))
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                self.initial_balls.append((x, y, color, self.color_rewards[color]))

        if not added_lava: # create random laval cells
            for i in range(self.num_lava_cells):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) == (self.width - 2, self.height - 2) or (x, y) == (1, 1):
                    continue
                self.lava_cells.append((x,y))



    def _place_infront_objects(self, infront_objects, number_of_objects=5):
        """
        Make an env with the same objects as the ones in the infront_objects list
        """
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

        if self.set_env: # create a random unique env
            self.unique_env = random.randint(1, 8)

                
        if self.unique_env > 0:
            return self._gen_unique_grid(width, height)
        if self.difficult_grid and width >= 8 and height >= 8:
            self._gen_difficult_grid(width, height)
            return
        
        # place lava cells
        for l_cell in self.lava_cells:
            self.put_obj(Lava(), l_cell[0], l_cell[1])
            
        # Place a ball in some cells
        # Check if kwargs include a list of initial_balls - create a new env sith the same balls but different locations
        
        for ball_info in self.initial_balls:
            x_loc = ball_info[0]
            y_loc = ball_info[1]
            color = ball_info[2]
            self.put_obj(Ball(color), x_loc, y_loc)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)


    def _gen_unique_grid(self, width, height):
        self.initial_balls = [] # reset the initial balls
        if self.unique_env == 1:
            # put a red ball in the right top corner and lava cells around it
            self.put_obj(Ball('red'), width-2, 1)
            self.initial_balls.append((width-2, 1, 'red', self.color_rewards['red']))
            self.put_obj(Lava(), width-3, 2)
            self.put_obj(Lava(), width-3, 1)
            # self.put_obj(Lava(), width-2, 2)
            self.put_obj(Ball('blue'), width-4, 3)
            self.initial_balls.append((width-4, 3, 'blue', self.color_rewards['blue']))

        if self.unique_env == 2:
            self.put_obj(Ball('blue'), width-3, 2)
            self.initial_balls.append((width-3, 2, 'blue', self.color_rewards['blue']))
            self.put_obj(Ball('green'), 1, 4)
            self.initial_balls.append((1, 4, 'green', self.color_rewards['green']))
            self.put_obj(Ball('green'), 1, 6)
            self.initial_balls.append((1, 6, 'green', self.color_rewards['green']))

        if self.unique_env == 3:
            self.put_obj(Lava(), 1, 2)
            self.put_obj(Lava(), 2, 2)
            self.put_obj(Lava(), 3, 2)
            self.put_obj(Lava(), 4, 2)
            self.put_obj(Ball('blue'), 1, 3)
            self.initial_balls.append((1, 3, 'blue',self.color_rewards['blue']))
            self.put_obj(Ball('green'), 1, 4)
            self.initial_balls.append((1, 4, 'green',self.color_rewards['green']))

        if self.unique_env == 4:
            self.put_obj(Lava(), 1, 3)
            self.put_obj(Lava(), 2, 3)
            self.put_obj(Lava(), 3, 3)
            self.put_obj(Ball('green'), 1, 5)
            self.initial_balls.append((1, 5,'green', self.color_rewards['green']))
            self.put_obj(Ball('red'), 1, 4)
            self.initial_balls.append((1, 4,'red', self.color_rewards['red']))
            self.put_obj(Ball('blue'), 6, 2)
            self.initial_balls.append((6, 2,'blue', self.color_rewards['blue']))
            self.put_obj(Ball('green'), 5, 5)
            self.initial_balls.append((5, 5,'green', self.color_rewards['green']))
        
        if self.unique_env == 5:
            self.put_obj(Lava(), 6, 1)
            self.put_obj(Lava(), 5, 2)
            self.put_obj(Ball('blue'), 1, 3)
            self.initial_balls.append((1, 3,'blue', self.color_rewards['blue']))
            self.put_obj(Ball('blue'), 6, 2)
            self.initial_balls.append((6, 2,'blue', self.color_rewards['blue']))
            self.put_obj(Ball('green'), 5, 4)
            self.initial_balls.append((1, 4,'green', self.color_rewards['green']))
            self.put_obj(Ball('green'), 6, 5)
            self.initial_balls.append((6, 5,'green', self.color_rewards['green']))
        

        if self.unique_env == 6:
            self.put_obj(Lava(), 3, 3)
            self.put_obj(Lava(), 1, 3)
            self.put_obj(Ball('red'), 4, 3)
            self.initial_balls.append((4, 3,'red', self.color_rewards['red']))
            self.put_obj(Ball('red'), 5, 2)
            self.initial_balls.append((5, 2,'red', self.color_rewards['red']))
            self.put_obj(Ball('red'), 2, 6)
            self.initial_balls.append((2, 6,'red', self.color_rewards['red']))
            self.put_obj(Ball('red'), 3, 6)
            self.initial_balls.append((3, 6,'red', self.color_rewards['red']))
            self.put_obj(Ball('green'), 1, 4)
            self.initial_balls.append((1, 4,'green', self.color_rewards['green']))
            

        if self.unique_env == 7:
            self.put_obj(Ball('blue'), 4, 4)
            self.initial_balls.append((4, 4, 'blue',self.color_rewards['blue']))
            self.put_obj(Ball('green'), 6, 2)
            self.initial_balls.append((6, 2, 'green',self.color_rewards['green']))
            for i in range(3,6):
                for j in range(3, 6):
                    if (i == 4 and j == 4) or (i == 5 and j == 4):
                        continue
                    self.put_obj(Lava(), i,j)
        
        if self.unique_env == 8:
            red_ball_positions = [(3, 2), (5, 6), (4, 3), (4, 6), (5, 1)]
            for pos in red_ball_positions:
                self.put_obj(Ball('red'), pos[0], pos[1])
                self.initial_balls.append((pos[0], pos[1], 'red', self.color_rewards['red']))

        if self.unique_env == 100:
            # Special final step environment - create a more challenging layout
            blue_ball_positions = [(2, 3), (5, 2)]
            green_ball_positions = [(3, 5), (4, 2)]
            red_ball_positions = [(2, 6)]
            lava_positions = [(3, 3), (4, 4), (5, 5), (3, 4)]
            
            for pos in blue_ball_positions:
                self.put_obj(Ball('blue'), pos[0], pos[1])
                self.initial_balls.append((pos[0], pos[1], 'blue', self.color_rewards['blue']))
            
            for pos in green_ball_positions:
                self.put_obj(Ball('green'), pos[0], pos[1])
                self.initial_balls.append((pos[0], pos[1], 'green', self.color_rewards['green']))
                
            for pos in red_ball_positions:
                self.put_obj(Ball('red'), pos[0], pos[1])
                self.initial_balls.append((pos[0], pos[1], 'red', self.color_rewards['red']))
                
            for pos in lava_positions:
                self.put_obj(Lava(), pos[0], pos[1])
                self.lava_cells.append(pos)

            
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
        # print(f"step {self.step_count}, action={action}")
        obs, reward, terminated, truncated, info = super().step(action)
        if self.step_count_observation:
            obs['step_count'] = self.step_count
        if self.image_full_view:
            obs['image'] = self.grid.encode() # get the full grid image
            self.put_agent_in_obs(obs)
        self.current_state = obs
        
        # Check if the agent picked up a ball
        if self.carrying:
            hold_obj = self.carrying
            if hold_obj.type == "ball":
                ball_color = self.carrying.color
                reward += self.color_rewards.get(ball_color, 0) 
                self.carrying = None
                self.on_board_objects -= 1

        # # hit a lava cell
        # if self.agent_pos in self.lava_cells: 
        #     terminated = False    
        #     reward += self.lava_panishment
        if self.agent_pos == (self.width - 2, self.height - 2):
            reward = 10 if self.train_env else 4.9
            terminated = True
            return obs, round(reward, 1), terminated, truncated, info

        if truncated:
            terminated = True
            print(f"reached max steps={self.max_steps}")
            reward -= 5

        # each step cost the agent some reward, in order to minimize the number of steps
        reward -=  self.step_cost 
        # if terminated:
        #     print(f"terminated, reward={reward}")
        return obs, round(reward, 1), terminated, truncated, info

    def find_optimal_path(self):
        points = [(1,1,0)] + self.initial_balls
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
