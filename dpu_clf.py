import numpy as np
import matplotlib
import time
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Rectangle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from stable_baselines3 import PPO
from minigrid_custom_env import *
from minigrid_custom_train import UpgradedObjEnvExtractor, ImageObjEnvExtractor, LargeObjEnvExtractor
import json
import copy
import heapq
from collections import defaultdict, deque
from itertools import combinations, permutations




def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

@timeit
def load_agent(env, model_path, update=False, image_obs=False, update_type="regular") -> PPO:
    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    model_path = model_path.split('.zip')[0]
    print(f'load new model path {model_path}')
    if update:
        if update_type == "updated":
            custom_objects = {
            "policy_kwargs": {"features_extractor_class": UpgradedObjEnvExtractor},  # Example kernel size
            "clip_range": 0.2,  # Example custom parameters
            "lr_schedule": 0.001  # Example learning rate schedule
            }
        elif update_type == "large":
            custom_objects = {
                "policy_kwargs": {"features_extractor_class": LargeObjEnvExtractor},  # Example kernel size
                "clip_range": 0.2,  # Example custom parameters
                "lr_schedule": 0.001  # Example learning rate schedule
            }
        else:
            custom_objects = {
                "policy_kwargs": {"features_extractor_class": ObjEnvExtractor},  # Example kernel size
                "clip_range": 0.2,  # Example custom parameters
                "lr_schedule": 0.001  # Example learning rate schedule
            }
    elif image_obs:
        custom_objects = {
        "policy_kwargs": {"features_extractor_class": ImageObjEnvExtractor},  # Example kernel size
        "clip_range": 0.2,  # Example custom parameters
        "lr_schedule": 0.001,  # Example learning rate schedule
        "image_obs": True  # Indicate that the model uses image observations
    }
    else:
        custom_objects = {
        "policy_kwargs": {"features_extractor_class": ObjEnvExtractor},  # Example kernel size
        "clip_range": 0.2,  # Example custom parameters
        "lr_schedule": 0.001  # Example learning rate schedule
    }
    # Load the model
    ppo = PPO.load(f"{model_path}", custom_objects=custom_objects, env=env)
    # Print environment observation and PPO model observation
    # print("Environment Observation Space:", env.observation_space)
    # print("PPO Model Observation Space:", ppo.observation_space)
    return ppo

def add_path_to_csv(model_path, preference_vector, name, eval_reward):
    new_data_dict = {
        "model_path": [model_path],
        "preference_vector": [preference_vector],
        "model_name": [name],
        "eval_reward": [eval_reward]
        }
    new_df = pd.DataFrame(new_data_dict)

    # Append to the existing CSV:
    new_df.to_csv("models/models_vectors.csv", mode="a", header=False, index=False)

def image_to_base64(image_array):
    """Convert NumPy array to a base64-encoded PNG."""
    img = Image.fromarray(np.uint8(image_array))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

def is_illegal_move(action, last_obs, obs, agent_pos_befor, agent_pos):
    if action <= 1: # turn is always legal
        return False
    if action == 2 and agent_pos_befor == agent_pos:
        # print("illigal_move ", action)
        return True
    if action > 2 and np.array_equal(obs['image'], last_obs['image']):
        # print("illigal_move ", action)
        return True
    return False


# resert the environment and run the agent on that environment to find his path

def interesting_objects(env, obs, agent_action, feedback_action):
    """
    Get the interesting objects from the observation based on the agent's and feedback's actions.
    """
    tmp_env = copy.deepcopy(env)
    interesting_objects = []
    interesting_objects += get_objects_from_image(obs['image'], number_of_cells=3)
    tmp_env.step(agent_action)
    interesting_objects += get_objects_from_image(tmp_env.get_wrapper_attr('current_state')['image'], number_of_cells=3)
    env.step(feedback_action)
    interesting_objects += get_objects_from_image(env.get_wrapper_attr('current_state')['image'], number_of_cells=3)
    return set(interesting_objects)

def get_objects_from_image(image, number_of_cells=5):
    """
    Get the objects from the image and return them as a list of dictionaries.
    Each dictionary contains the object type, color, and state.
    """
    around_objects = []
    mid = 3 # env.width //2
    for i in range(2,number_of_cells+2):
        # ASSERT number_of_objects <= min(env.width, env.height) - 1
        around_objects.append(image[mid][-i]) # for some reason the obs image is reversed
    around_objects.append(image[mid+1][-2])
    around_objects.append(image[mid-1][-2])
    return list(map(tuple, around_objects))

def get_infront_object(obs, to_print=False):
    """get the first object in front of the agent - in the main row"""
    image = obs['image']
    if to_print:
        print(f"(get_infront_object) image:\n {image}")
    mid = 3 # env.width //2
    i = 0
    for obj in image[mid][::-1]:
        if to_print:
            print(f"(get_infront_object) index:{i} obj: {obj}, {IDX_TO_OBJECT[obj[0]]}")
        if IDX_TO_OBJECT[obj[0]] == 'ball' or IDX_TO_OBJECT[obj[0]] == 'lava':
            return tuple(obj)
        i += 1
    return tuple(image[mid][0])


@timeit
def capture_agent_path(copy_env: CustomEnv, agent: PPO) -> (list, int, int, list): # -> list of moves (action and diraction, action), number of illegal moves, total reward, list of actions names
    illigal_moves = 0
    last_obs = copy_env.get_wrapper_attr('current_state')
    
    # last_obs = env.reset()
    # last_obs = last_obs[0]
    ligal_actions = []
    agent_actions = []
    state_record = [last_obs]
    total_reward = 0    
    derminated = False  
    truncated = False
    # copy_env = copy.deepcopy(env)
    # plt.imshow(copy_env.render())
    while not (derminated or truncated):
        agent_pos_before = copy_env.get_wrapper_attr('agent_pos')
        action, _states = agent.predict(last_obs, deterministic=True)
        agent_actions.append(action)
        obs, reward, derminated, truncated, info = copy_env.step(action)
        total_reward += reward
        
        if is_illegal_move(action, last_obs, obs, agent_pos_before, copy_env.get_wrapper_attr('agent_pos')):
            illigal_moves += 1
            continue

        ligal_actions.append(action)
        last_obs = obs
        state_record.append(obs)
        # plt.imshow(copy_env.render())
        # plt.show()
        
        
    # number_to_action = {0: 'turn right', 1: 'turn left', 3: 'pickup'}
    agent_dir = "right"
    move_sequence = []
    for action in ligal_actions:
        if action == 0: # turn left
            agent_dir = turn_agent(agent_dir, "left")
            move_sequence.append((agent_dir, 'turn left'))
        elif action == 1: # turn right
            agent_dir = turn_agent(agent_dir, "right")
            move_sequence.append((agent_dir, 'turn right'))
        elif action == 2: # move forward
            move_sequence.append((agent_dir, 'forward'))
        elif action == 3: # pickup
            move_sequence.append((agent_dir, 'pickup'))
    return move_sequence, illigal_moves, total_reward, ligal_actions

def actions_cells_locations(move_sequence: list) -> list:
    x, y = 1,1
    cells_dir = {"up" :   (0, -1),
                 "down":  (0, 1),
                 "right": (1, 0),
                 "left":  (-1, 0)}
    actions_cells = [(x,y)]
    for dir, action in move_sequence:
        if action == "forward":
            x += cells_dir[dir][0]
            y += cells_dir[dir][1]
        actions_cells.append((x,y))
    # print("move sequence=", move_sequence)
    # print("actions_cells", actions_cells)
    return actions_cells


def convert_move_sequence_to_jason(move_sequence: list):
    """
    Convert a move_sequence (list of strings) into a JSON-serializable list 
    where each element is a two-item list: [actionDir, action].
    
    Example conversion:
      "left"      -> ["left", "turn left"]
      "pickup right"   -> ["right", "pickup"]
      "right"          -> ["right", "forward"]
    """
    converted = []
    for direction, action in move_sequence:
        converted.append([direction, action])

    return json.dumps(converted)


def turn_agent(agent_dir, turn_dir) -> str:
    turnning_dict = {("up", "left"): "left", ("up", "right"): "right", 
                     ("down", "left"): "right", ("down", "right"): "left",
                     ("left", "left"): "down", ("left", "right"): "up",
                     ("right", "left"): "up", ("right", "right"): "down"}
    return turnning_dict[(agent_dir, turn_dir)]

def ax_to_feedback_image(ax):
    ax.axis('off')
    feedback_buf = BytesIO()
    plt.savefig(feedback_buf, format='png', bbox_inches='tight')
    feedback_buf.seek(0)
    img_base64 = base64.b64encode(feedback_buf.getvalue()).decode('ascii')
    return img_base64

@timeit
def plot_all_move_sequence(img, move_sequence, agent_true_actions, move_color='y', turn_color='cyan', pickup_color='pink', converge_action_location = -1): # -> State image with the path of the agent, actions marks locations    
    imgs_action_list = []
    feedback_action_color = 'cyan'
    start_point = (50, 50)
    arrow_size = 20
    arrow_head_size = 12
    small_shift = 9
    actions_with_location = []
    all_arrow_size = arrow_size + arrow_head_size
    move_arrow_sizes = {'up': (0, -20, 0, -all_arrow_size), 
                        'down': (0, 20, 0, all_arrow_size), 
                        'right': (20, 0, all_arrow_size, 0), 
                        'left': (-20, 0, -all_arrow_size, 0)}
    turn_arrow_sizes = {'up': (0, -5),
                        'down': (0, 5),
                        'right': (5, 0),
                        'left': (-5, 0)}
    pickup_direction = {'up': (0, -1),
                         'down': (0, 1),
                         'left': (-1, 0),
                         'right': (1, 0)}
    # arrows_list = ['right', 'right', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'up', 'right', 'down']

    mark_x, mark_y = start_point[0] + 80, start_point[1] + 40 # 300, 160
    mark_sizes = {'move_vertical': (25, 70), 'move_horizontal': (80, 20), 'turn': (20, 20), 'pickup': (20, 20)}
    # mark_sizes = {'move_vertical': (30, 30), 'move_horizontal': (30, 30), 'turn': (30, 30), 'pickup': (30, 30)}

    # mark_move_sizes = {'move_vertical': 20, 'move_horizontal': 20}
    fig, ax = plt.subplots()
    ax.imshow(img)
    current_point = start_point

    i= 0
    for action_dir, actual_action in move_sequence:
        if i == converge_action_location:
            # add a rectangle to mark the converging point:
            ax.add_patch(Rectangle((current_point[0] - 10, current_point[1]- 10), 15, 15, color='b', alpha=0.4))
        i += 1
        action_loc = {'action': actual_action}
        # moving action
        if actual_action == 'forward': # a big arrow that represents a move
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=feedback_action_color, ec=feedback_action_color)
            imgs_action_list.append(ax_to_feedback_image(ax))

            # overide the feedback arrow with the real action
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=move_color, ec=move_color)
            current_point = (current_point[0] + move_arrow_sizes[action_dir][2], current_point[1] + move_arrow_sizes[action_dir][3])
            mark_size = mark_sizes['move_vertical'] if action_dir == 'up' or action_dir == 'down' else mark_sizes['move_horizontal']
            
            if action_dir == 'up':
                mark_y -= 25 + all_arrow_size
            elif action_dir == 'left': # move left
                mark_x += - 43 - all_arrow_size
            action_loc['x'] = mark_x #current_point[0] + mark_x
            action_loc['y'] = mark_y #current_point[1] + mark_y
            action_loc['width'] = mark_size[0]
            action_loc['height'] = mark_size[1]
            # action_loc['width'] = max(move_arrow_sizes[action_name][2], min_width) * inlarge_factor
            # action_loc['height'] = max(move_arrow_sizes[action_name][3], min_hieght) * inlarge_factor
            
            if action_dir == 'down':
                mark_y += 25 + all_arrow_size
            elif action_dir == 'right':
                mark_x += 43 + all_arrow_size
            
        
        #turning action   
        elif actual_action in turn_arrow_sizes.keys(): # a small arrow that represents a turn or a pickup
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=feedback_action_color, ec=feedback_action_color)
            imgs_action_list.append(ax_to_feedback_image(ax))

            # overide the feedback arrow with the real action
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            shift_size = 17
            turnning_mark_shifts = {'up': (0, -shift_size), 'down': (0, shift_size), 'right': (shift_size, 0), 'left': (-shift_size, 0)}
            x_shift, y_shift = turnning_mark_shifts[action_dir]
            # print("turned to:", action_dir)
            
            if action_dir == 'up':
                mark_x -= 2
                # mark_y += -10
            elif action_dir == 'down':
                mark_x -= 2
                # mark_y += -5
            elif action_dir == 'right':
                mark_x -= 2
                # mark_y += -10
            else: # turn left
                mark_x += 5
                # mark_y += -10

            action_loc['x'] = mark_x + x_shift
            action_loc['y'] = mark_y + y_shift
            action_loc['width'] = mark_sizes['turn'][0]
            action_loc['height'] = mark_sizes['turn'][1]

            
        # pickup action
        elif actual_action == 'pickup':        
            pickup_position = pickup_direction[action_dir]

            ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=feedback_action_color)
            imgs_action_list.append(ax_to_feedback_image(ax))

            # overide the feedback arrow with the real action
            ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=pickup_color)
            action_loc['x'] = mark_x + 25 * pickup_position[0]
            action_loc['y'] = mark_y + 15 * pickup_position[1]
            action_loc['width'] = mark_sizes['pickup'][0]
            action_loc['height'] = mark_sizes['pickup'][1]
        actions_with_location.append(action_loc)

        # Capture the current state of the plot
        # Add a yellow half-transparent rectangle for the last action

        #, actions_translation[agent_true_actions[i-1]]))

    buf = ax_to_feedback_image(ax)
    return buf, actions_with_location, imgs_action_list

# each move with its own image
@timeit
# return -> buffer with the last image, actions :[{'action', 'x', 'y', 'width', 'height'},..], imgs_action_list: list of base64 encoded images
def plot_move_sequence_by_parts(imgs, move_sequence, agent_true_actions, move_color='yellow', turn_color='cyan', pickup_color='#e6007a', converge_action_location = -1): # -> State image with the path of the agent, actions marks locations    
    # print("move sequence:", move_sequence)
    imgs_action_list = []
    feedback_action_color = 'cyan'
    start_point = (50, 50)
    arrow_size = 20
    arrow_head_size = 12
    small_shift = 9
    actions_with_location = []
    all_arrow_size = arrow_size + arrow_head_size
    move_arrow_sizes = {'up': (0, -20, 0, -all_arrow_size), 
                        'down': (0, 20, 0, all_arrow_size), 
                        'right': (20, 0, all_arrow_size, 0), 
                        'left': (-20, 0, -all_arrow_size, 0)}
    turn_arrow_sizes = {'up': (0, -5),
                        'down': (0, 5),
                        'right': (5, 0),
                        'left': (-5, 0)}
    pickup_direction = {'up': (0, -1),
                         'down': (0, 1),
                         'left': (-1, 0),
                         'right': (1, 0)}
    # arrows_list = ['right', 'right', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'up', 'right', 'down']

    mark_x, mark_y = start_point[0] + 80, start_point[1] + 40 # 300, 160
    mark_sizes = {'move_vertical': (25, 70), 'move_horizontal': (80, 20), 'turn': (20, 20), 'pickup': (20, 20)}
    current_point = start_point

    i= 0
    print(f'len agent_true_actions: {len(agent_true_actions)} move_sequence len: {len(move_sequence)}, imgs len: {len(imgs)}')
    for action_dir, actual_action in move_sequence:
        # print(f"plot move sequence by parts: action_dir: {action_dir}, actual_action: {actual_action}, i: {i}")
        fig , ax = plt.subplots()
        ax.imshow(imgs[i])
        if i == converge_action_location:
            # add a rectangle to mark the converging point:
            ax.add_patch(Rectangle((current_point[0] - 10, current_point[1]- 10), 15, 15, color='b', alpha=0.4))
        i += 1

        action_loc = {'action': actual_action, 'action_dir': action_dir}
        # moving action
        if actual_action  == 'forward': # a big arrow that represents a move
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=move_color, ec=move_color) #fc=feedback_action_color, ec=feedback_action_color)
            buf = ax_to_feedback_image(ax)

            # overide the feedback arrow with the real action
            current_point = (current_point[0] + move_arrow_sizes[action_dir][2], current_point[1] + move_arrow_sizes[action_dir][3])
            mark_size = mark_sizes['move_vertical'] if action_dir == 'up' or action_dir == 'down' else mark_sizes['move_horizontal']
            
            if action_dir == 'up':
                mark_y -= 25 + all_arrow_size
            elif action_dir == 'left': # move left
                mark_x += - 43 - all_arrow_size
            action_loc['x'] = mark_x 
            action_loc['y'] = mark_y 
            action_loc['width'] = mark_size[0]
            action_loc['height'] = mark_size[1]
            
            if action_dir == 'down':
                mark_y += 25 + all_arrow_size
            elif action_dir == 'right':
                mark_x += 43 + all_arrow_size
            
        
        #turning action   
        elif 'turn' in actual_action: # a small arrow that represents a turn
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            buf = ax_to_feedback_image(ax)
            shift_size = 17
            turnning_mark_shifts = {'up': (0, -shift_size), 'down': (0, shift_size), 'right': (shift_size, 0), 'left': (-shift_size, 0)}
            x_shift, y_shift = turnning_mark_shifts[action_dir]
            # print("turned to:", action_dir)
            
            if action_dir == 'up':
                mark_x -= 2
            elif action_dir == 'down':
                mark_x -= 2
            elif action_dir == 'right':
                mark_x -= 2
            else: # turn left
                mark_x += 5

            action_loc['x'] = mark_x + x_shift
            action_loc['y'] = mark_y + y_shift
            action_loc['width'] = mark_sizes['turn'][0]
            action_loc['height'] = mark_sizes['turn'][1]
            
        # pickup action
        elif actual_action == 'pickup':    
            pickup_position = pickup_direction[action_dir]

            
            _ = ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=pickup_color)
            buf = ax_to_feedback_image(ax)
            
            action_loc['x'] = mark_x + 25 * pickup_position[0]
            action_loc['y'] = mark_y + 15 * pickup_position[1]
            action_loc['width'] = mark_sizes['pickup'][0]
            action_loc['height'] = mark_sizes['pickup'][1]
        
        #invalide move  #TODO:make the image with an invalide mode
        else:
            action_loc['x'] = mark_x
            action_loc['y'] = mark_y
            action_loc['width'] = 0
            action_loc['height'] = 0
            buf = ax_to_feedback_image(ax)

        actions_with_location.append(action_loc)
        imgs_action_list.append(buf)
        plt.close(fig)
    return buf, actions_with_location, imgs_action_list

def will_it_stuck(agent: PPO, env: CustomEnv) -> bool:
    truncated = False
    terminated = False
    cpy_env = copy.deepcopy(env)
    obs = cpy_env.get_wrapper_attr('current_state')
    while not truncated and not terminated:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = cpy_env.step(action)
    if truncated:
        return True
    return False

WALL_SHIFT_FACTOR = 1
WALL_FACTOR = 10
DOOR_FACTOR = 0.1
BALLS_FACTOR = 3
min_ball_distance = 3

def manhattan_distance(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
    
def balls_distance(balls):
    ball_dist = 0
    for i in range(len(balls)-1):
        for j in range(i+1, len(balls)):
            ball_dist += np.linalg.norm(np.array(balls[i][:-1]) - np.array(balls[j][:-1]))
    return ball_dist

def balls_groups(balls_list, to_print=False):
    print(f"balls_list: {balls_list}")
    groups = []
    in_any_group = set()
    for i in range(len(balls_list)):
        if i in in_any_group:
            continue
        group = [balls_list[i]]
        need_to_check = [i]
        in_any_group.add(i)
        while need_to_check:
            ball_index = need_to_check.pop()
            check_ball = balls_list[ball_index]
            for j in range(len(balls_list)):
                if j in in_any_group: # TODO: option to switch to a list with all the balls that are not in any group
                    continue
                if manhattan_distance(check_ball, balls_list[j]) <= min_ball_distance:
                    group.append(balls_list[j])
                    in_any_group.add(j)
                    need_to_check.append(j)
        groups.append(group)
    if to_print:
        print(f"groups: {groups}")
    
    res = []
    for group in groups:
        x_center = np.mean([ball[0] for ball in group])
        y_center = np.mean([ball[1] for ball in group])
        res.append((len(group), (x_center, y_center)))
    if to_print:
        print(f"res: {res}")
    return res

def biggest_group(balls_groups):
    max = 0
    max_group = None
    for group in balls_groups:
        if group[0] > max:
            max = group[0]
            max_group = group
    return max_group
    
def state_distance(objects1, objects2):
    distance = 0
    if objects1['wall'][0] or objects2['wall'][0]: # if one of the states has a wall
        if objects1['wall'][0] != objects2['wall'][0]:
            distance += WALL_FACTOR
        else:
            distance += np.abs((objects1['wall'][2]) - (objects2['wall'][2]))*DOOR_FACTOR
            distance += np.abs((objects1['wall'][1]) - (objects2['wall'][1]))*WALL_SHIFT_FACTOR
    
    ball_groups1 = balls_groups(objects1['balls'])
    ball_groups2 = balls_groups(objects2['balls'])
    max_group1 = biggest_group(ball_groups1)
    max_group2 = biggest_group(ball_groups2)
    distance += np.abs(max_group1[0] - max_group2[0])*BALLS_FACTOR # changes in the biggest group size
    distance += np.abs(len([group for group in ball_groups1 if group[0] > 1]) - len([group for group in ball_groups2 if group[0] > 1]))# change in number of real groups(more then 1 ball)
    # distance += np.abs(balls_distance(objects1['balls']) - balls_distance(objects2['balls']))* BALLS_FACTOR
    return distance


def find_best_path_simple(env):
    """
    Simplified function to find the best path for collecting balls and reaching the goal.
    Uses a greedy approach with dynamic programming for better performance.
    
    Args:
        env: Your CustomEnv instance
    
    Returns:
        dict: {
            'path': list of (x, y) positions,
            'actions': list of action indices,
            'expected_reward': float,
            'collection_order': list of balls to collect
        }
    """
    
    grid = env.grid
    agent_pos = env.agent_pos
    color_rewards = env.color_rewards
    step_cost = getattr(env, 'step_cost', -0.1)
    
    # Find all valuable balls (positive reward)
    valuable_balls = []
    goal_pos = (grid.width - 2, grid.height - 2)  # Assuming bottom-right is goal
    
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and cell.type == "ball":
                reward = color_rewards.get(cell.color, 0)
                if reward > 0:  # Only consider balls with positive rewards
                    valuable_balls.append({
                        'pos': (x, y),
                        'color': cell.color,
                        'reward': reward,
                        'distance': abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                    })
            elif cell and cell.type == "goal":
                goal_pos = (x, y)
    
    if not valuable_balls:
        # No valuable balls, just go to goal
        distance_to_goal = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
        return {
            'path': [agent_pos, goal_pos],
            'actions': ['forward'] * distance_to_goal,  # Simplified
            'expected_reward': distance_to_goal * step_cost,
            'collection_order': []
        }
    
    # Use dynamic programming to find best subset of balls to collect
    def calculate_path_reward(ball_subset):
        """Calculate total reward for collecting a specific subset of balls."""
        if not ball_subset:
            distance_to_goal = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            return distance_to_goal * step_cost, [agent_pos, goal_pos]
        
        # Find optimal order using nearest neighbor heuristic
        current_pos = agent_pos
        path = [current_pos]
        total_reward = 0
        total_steps = 0
        remaining_balls = ball_subset.copy()
        
        # Collect balls in optimal order
        while remaining_balls:
            # Find nearest valuable ball
            best_ball = min(remaining_balls, 
                          key=lambda b: abs(b['pos'][0] - current_pos[0]) + abs(b['pos'][1] - current_pos[1]))
            
            # Move to ball
            distance = abs(best_ball['pos'][0] - current_pos[0]) + abs(best_ball['pos'][1] - current_pos[1])
            total_steps += distance
            total_reward += best_ball['reward']
            
            path.append(best_ball['pos'])
            current_pos = best_ball['pos']
            remaining_balls.remove(best_ball)
        
        # Go to goal
        distance_to_goal = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
        total_steps += distance_to_goal
        path.append(goal_pos)
        
        # Apply step cost
        total_reward += total_steps * step_cost
        
        return total_reward, path
    
    # Try different subsets of balls (limited to avoid exponential explosion)
    best_reward = float('-inf')
    best_path = []
    best_collection = []
    
    # Sort balls by reward/distance ratio for better heuristic
    valuable_balls.sort(key=lambda b: b['reward'] / max(b['distance'], 1), reverse=True)
    
    # Try different combinations (limited to first 5 balls for performance)
    from itertools import combinations
    max_balls_to_consider = min(len(valuable_balls), 5)
    
    for r in range(len(valuable_balls) + 1):
        if r > max_balls_to_consider:
            break
            
        for ball_combo in combinations(valuable_balls[:max_balls_to_consider], r):
            reward, path = calculate_path_reward(list(ball_combo))
            
            if reward > best_reward:
                best_reward = reward
                best_path = path
                best_collection = list(ball_combo)
    
    # Generate simplified actions (this is a simplification - in reality you'd need proper path planning)
    actions = []
    for i in range(len(best_path) - 1):
        current = best_path[i]
        next_pos = best_path[i + 1]
        distance = abs(next_pos[0] - current[0]) + abs(next_pos[1] - current[1])
        actions.extend(['forward'] * distance)  # Simplified - assumes proper orientation
    
    return {
        'path': best_path,
        'actions': actions,
        'expected_reward': best_reward,
        'collection_order': best_collection
    }


def compare_strategies(env):
    """
    Compare different strategies: collect all balls vs. selective collection vs. direct to goal.
    """
    
    print("=== STRATEGY COMPARISON ===")
    
    # Strategy 1: Direct to goal
    goal_pos = (env.grid.width - 2, env.grid.height - 2)
    direct_distance = abs(goal_pos[0] - env.agent_pos[0]) + abs(goal_pos[1] - env.agent_pos[1])
    direct_reward = direct_distance * getattr(env, 'step_cost', -0.1)
    
    print(f"Strategy 1 - Direct to Goal:")
    print(f"  Steps: {direct_distance}")
    print(f"  Expected reward: {direct_reward:.2f}")
    
    # Strategy 2: Optimal selection
    optimal_result = find_best_path_simple(env)
    
    print(f"\\nStrategy 2 - Optimal Ball Collection:")
    print(f"  Steps: {len(optimal_result['actions'])}")
    print(f"  Expected reward: {optimal_result['expected_reward']:.2f}")
    print(f"  Balls to collect: {len(optimal_result['collection_order'])}")
    
    if optimal_result['collection_order']:
        print("  Collection order:")
        for i, ball in enumerate(optimal_result['collection_order']):
            print(f"    {i+1}. {ball['color']} ball at {ball['pos']} (reward: {ball['reward']:+.1f})")
    
    # Strategy 3: Collect all valuable balls
    valuable_balls = []
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            cell = env.grid.get(x, y)
            if cell and cell.type == "ball":
                reward = env.color_rewards.get(cell.color, 0)
                if reward > 0:
                    valuable_balls.append(reward)
    
    if valuable_balls:
        total_ball_reward = sum(valuable_balls)
        # Rough estimate: assume we need to visit each ball (simplified)
        estimated_steps = len(valuable_balls) * 3 + direct_distance  # Very rough estimate
        collect_all_reward = total_ball_reward + estimated_steps * getattr(env, 'step_cost', -0.1)
        
        print(f"\\nStrategy 3 - Collect All Valuable Balls (rough estimate):")
        print(f"  Estimated steps: {estimated_steps}")
        print(f"  Estimated reward: {collect_all_reward:.2f}")
        print(f"  Ball rewards sum: {total_ball_reward:.1f}")
    
    print(f"\\n=== RECOMMENDATION ===")
    if optimal_result['expected_reward'] > direct_reward:
        print("✅ Recommended: Collect selected balls then go to goal")
        print(f"   Advantage: {optimal_result['expected_reward'] - direct_reward:.2f} extra reward")
    else:
        print("✅ Recommended: Go directly to goal")
        print("   Reason: Ball collection is not worth the extra steps")
    
    return {
        'direct_reward': direct_reward,
        'optimal_reward': optimal_result['expected_reward'],
        'recommendation': 'collect_balls' if optimal_result['expected_reward'] > direct_reward else 'direct_to_goal'
    }


def use_optimal_path_with_custom_env(env):
    """
    Example of how to use the optimal path finder with your CustomEnv.
    
    Args:
        env: Your CustomEnv instance
    
    Returns:
        Dictionary with path analysis results
    """
    
    # Get current state information
    agent_pos = env.agent_pos
    grid = env.grid
    color_rewards = env.color_rewards
    
    print(f"Agent starting position: {agent_pos}")
    print(f"Color rewards: {color_rewards}")
    
    # Find the optimal path
    optimal_path, max_reward, actions = find_optimal_path_with_rewards(
        grid=grid, 
        agent_pos=agent_pos, 
        color_rewards=color_rewards,
        step_cost=getattr(env, 'step_cost', -0.1),
        lava_penalty=getattr(env, 'lava_panishment', -3.0)
    )
    
    # Analyze the results
    results = {
        'path': optimal_path,
        'expected_reward': max_reward,
        'num_actions': len(actions),
        'actions': actions,
        'path_length': len(optimal_path)
    }
    
    print(f"\\n=== OPTIMAL PATH ANALYSIS ===")
    print(f"Expected total reward: {max_reward:.2f}")
    print(f"Number of steps needed: {len(actions)}")
    print(f"Path efficiency: {max_reward/len(actions):.3f} reward per step")
    
    # Show the path step by step
    print(f"\\n=== PATH BREAKDOWN ===")
    for i, pos in enumerate(optimal_path):
        if i == 0:
            print(f"START: Position {pos}")
        elif i == len(optimal_path) - 1:
            print(f"GOAL:  Position {pos}")
        else:
            # Check what's at this position
            cell = grid.get(pos[0], pos[1])
            if cell and cell.type == 'ball':
                reward_value = color_rewards.get(cell.color, 0)
                print(f"Step {i:2d}: Position {pos} -> Collect {cell.color} ball (reward: {reward_value:+.1f})")
            else:
                print(f"Step {i:2d}: Position {pos} -> Move through")
    
    return results


def analyze_current_board_state(env):
    """
    Analyze the current board state and show all available options.
    
    Args:
        env: Your CustomEnv instance
    """
    
    grid = env.grid
    agent_pos = env.agent_pos
    color_rewards = env.color_rewards
    
    print(f"=== BOARD STATE ANALYSIS ===")
    print(f"Grid size: {grid.width} x {grid.height}")
    print(f"Agent position: {agent_pos}")
    
    # Find all balls
    balls = []
    lava_cells = []
    goal_pos = None
    
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "ball":
                    reward = color_rewards.get(cell.color, 0)
                    balls.append((x, y, cell.color, reward))
                elif cell.type == "lava":
                    lava_cells.append((x, y))
                elif cell.type == "goal":
                    goal_pos = (x, y)
    
    if goal_pos is None:
        goal_pos = (grid.width - 2, grid.height - 2)
    
    print(f"Goal position: {goal_pos}")
    print(f"\\nBALLS ON BOARD:")
    total_possible_reward = 0
    for i, (x, y, color, reward) in enumerate(balls):
        distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
        print(f"  {i+1}. {color} ball at ({x}, {y}) - Reward: {reward:+.1f} - Distance: {distance}")
        if reward > 0:
            total_possible_reward += reward
    
    print(f"\\nLAVA CELLS: {len(lava_cells)} cells")
    for x, y in lava_cells:
        distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
        print(f"  Lava at ({x}, {y}) - Distance: {distance}")
    
    print(f"\\nREWARD ANALYSIS:")
    print(f"  Total possible ball rewards: {total_possible_reward:+.1f}")
    print(f"  Distance to goal: {abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])}")
    
    step_cost = getattr(env, 'step_cost', -0.1)
    print(f"  Step cost: {step_cost:+.1f} per move")
    
    # Quick heuristic: is it worth collecting balls?
    min_steps_to_collect_all = sum(abs(x - agent_pos[0]) + abs(y - agent_pos[1]) for x, y, _, reward in balls if reward > 0)
    if min_steps_to_collect_all > 0:
        efficiency = total_possible_reward / min_steps_to_collect_all
        print(f"  Rough efficiency estimate: {efficiency:.3f} reward per step")
    
    return {
        'balls': balls,
        'lava_cells': lava_cells,
        'goal_pos': goal_pos,
        'total_possible_reward': total_possible_reward
    }


def find_optimal_path_with_rewards(grid, agent_pos, color_rewards, step_cost=-0.1, lava_penalty=-3.0):
    """
    Find the optimal path for the agent to collect balls and reach the goal while maximizing rewards.
    
    Args:
        grid: The MiniGrid grid object
        agent_pos: Current agent position (x, y)
        color_rewards: Dictionary mapping ball colors to their reward values
        step_cost: Cost per step (negative value)
        lava_penalty: Penalty for stepping on lava (negative value)
    
    Returns:
        tuple: (best_path, total_reward, path_actions)
            - best_path: List of positions [(x, y), ...]
            - total_reward: Maximum achievable reward
            - path_actions: List of actions to execute the path
    """
    
    # Extract game elements from the grid
    balls = []
    lava_cells = set()
    goal_pos = None
    
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "ball":
                    reward = color_rewards.get(cell.color, 0)
                    balls.append((x, y, cell.color, reward))
                elif cell.type == "lava":
                    lava_cells.add((x, y))
                elif cell.type == "goal":
                    goal_pos = (x, y)
    
    # If no goal found, assume bottom-right corner
    if goal_pos is None:
        goal_pos = (grid.width - 2, grid.height - 2)
    
    print(f"Found {len(balls)} balls, {len(lava_cells)} lava cells, goal at {goal_pos}")
    print(f"Balls: {balls}")
    
    # Filter balls with positive rewards
    valuable_balls = [(x, y, color, reward) for x, y, color, reward in balls if reward > 0]
    
    # Generate all possible ball collection sequences using dynamic programming
    def get_optimal_collection_sequence():
        if not valuable_balls:
            return [], 0
        
        # Create distance matrix between all points (agent start + balls + goal)
        points = [agent_pos] + [(x, y) for x, y, _, _ in valuable_balls] + [goal_pos]
        n_points = len(points)
        
        # Calculate Manhattan distances (avoiding walls but not considering lava for now)
        dist_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dist_matrix[i][j] = manhattan_distance_safe(points[i], points[j], grid, lava_cells)
                else:
                    dist_matrix[i][j] = 0
        
        # Dynamic programming to find best subset and order of balls to collect
        ball_indices = list(range(1, len(valuable_balls) + 1))  # indices 1 to len(valuable_balls)
        goal_index = len(valuable_balls) + 1  # goal is the last index
        
        best_reward = float('-inf')
        best_sequence = []
        
        # Try all possible subsets of balls
        for r in range(len(valuable_balls) + 1):  # r is the number of balls to collect
            for ball_subset in combinations(ball_indices, r):
                # Try all permutations of this subset
                for perm in permutations(ball_subset):
                    # Calculate reward for this sequence
                    sequence = [0] + list(perm) + [goal_index]  # start -> balls -> goal
                    total_steps = 0
                    total_reward = 0
                    
                    # Calculate total distance and reward
                    for i in range(len(sequence) - 1):
                        current_idx = sequence[i]
                        next_idx = sequence[i + 1]
                        total_steps += dist_matrix[current_idx][next_idx]
                    
                    # Add ball rewards
                    for ball_idx in perm:
                        ball_reward = valuable_balls[ball_idx - 1][3]  # -1 because ball indices start from 1
                        total_reward += ball_reward
                    
                    # Subtract step costs
                    total_reward += total_steps * step_cost
                    
                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_sequence = sequence
        
        return best_sequence, best_reward
    
    # Get optimal sequence
    sequence, max_reward = get_optimal_collection_sequence()
    
    if not sequence:
        # No balls to collect, just go to goal
        path = [agent_pos, goal_pos]
        actions = plan_path_between_points(agent_pos, goal_pos, grid, lava_cells)
        return path, step_cost * len(actions), actions
    
    # Convert sequence indices back to positions
    points = [agent_pos] + [(x, y) for x, y, _, _ in valuable_balls] + [goal_pos]
    path_positions = [points[i] for i in sequence]
    
    # Generate actions for the complete path
    all_actions = []
    full_path = []
    
    for i in range(len(path_positions) - 1):
        start_pos = path_positions[i]
        end_pos = path_positions[i + 1]
        segment_actions = plan_path_between_points(start_pos, end_pos, grid, lava_cells)
        all_actions.extend(segment_actions)
        
        if i == 0:
            full_path.append(start_pos)
        full_path.append(end_pos)
    
    return full_path, max_reward, all_actions


def manhattan_distance_safe(pos1, pos2, grid, lava_cells):
    """Calculate Manhattan distance while avoiding lava cells if possible."""
    x1, y1 = pos1
    x2, y2 = pos2
    base_distance = abs(x2 - x1) + abs(y2 - y1)
    
    # Add penalty if path goes through lava (simplified heuristic)
    lava_penalty_multiplier = 1.0
    path_through_lava = check_path_through_lava(pos1, pos2, lava_cells)
    if path_through_lava:
        lava_penalty_multiplier = 2.0  # Increase distance to discourage lava paths
    
    return base_distance * lava_penalty_multiplier


def check_path_through_lava(pos1, pos2, lava_cells):
    """Check if a straight-line path between two points goes through lava."""
    x1, y1 = pos1
    x2, y2 = pos2
    
    # Simple check: see if any lava cell is on the straight line path
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx == 0:  # Vertical line
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if (x1, y) in lava_cells:
                return True
    elif dy == 0:  # Horizontal line
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if (x, y1) in lava_cells:
                return True
    
    return False


def plan_path_between_points(start_pos, end_pos, grid, lava_cells):
    """
    Plan a path between two points using A* algorithm, avoiding walls and lava.
    
    Returns:
        List of actions to move from start_pos to end_pos
    """
    
    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        # Check all 4 directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            # Check bounds
            if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
                cell = grid.get(new_x, new_y)
                # Can move if cell is empty, has ball, or is goal (but not wall)
                if cell is None or cell.type in ['ball', 'goal']:
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def heuristic(pos):
        return abs(pos[0] - end_pos[0]) + abs(pos[1] - end_pos[1])
    
    def get_cost(pos):
        # Higher cost for lava cells to discourage but not prohibit
        return 10 if pos in lava_cells else 1
    
    # A* algorithm
    open_set = [(0, start_pos)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start_pos] = 0
    f_score = defaultdict(lambda: float('inf'))
    f_score[start_pos] = heuristic(start_pos)
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == end_pos:
            # Reconstruct path
            path = []
            while current in came_from:
                prev = came_from[current]
                path.append(current)
                current = prev
            path.append(start_pos)
            path.reverse()
            
            # Convert path to actions (simplified - assumes agent always faces right direction)
            actions = []
            for i in range(len(path) - 1):
                curr_pos = path[i]
                next_pos = path[i + 1]
                dx = next_pos[0] - curr_pos[0]
                dy = next_pos[1] - curr_pos[1]
                
                if dx == 1:  # Move right
                    actions.append('forward')  # Assuming correct orientation
                elif dx == -1:  # Move left
                    actions.append('forward')
                elif dy == 1:  # Move down
                    actions.append('forward')
                elif dy == -1:  # Move up
                    actions.append('forward')
            
            return actions
        
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + get_cost(neighbor)
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return []


def visualize_optimal_path(grid, path, balls_to_collect):
    """
    Visualize the optimal path on the grid.
    """
    print(f"\\nOptimal Path Visualization:")
    print(f"Path length: {len(path)} positions")
    print(f"Balls to collect: {len(balls_to_collect)}")
    
    for i, pos in enumerate(path):
        if i == 0:
            print(f"Start: {pos}")
        elif i == len(path) - 1:
            print(f"Goal: {pos}")
        else:
            # Check if this position has a ball to collect
            ball_here = None
            for ball in balls_to_collect:
                if ball[0] == pos[0] and ball[1] == pos[1]:
                    ball_here = ball
                    break
            
            if ball_here:
                print(f"Step {i}: {pos} - Collect {ball_here[2]} ball (reward: {ball_here[3]})")
            else:
                print(f"Step {i}: {pos}")


# Direction mappings for MiniGrid
DIRECTIONS = {
    0: (0, -1),  # Up
    1: (1, 0),   # Right  
    2: (0, 1),   # Down
    3: (-1, 0)   # Left
}

DIRECTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

# Actions in MiniGrid
MINIGRID_ACTIONS = {
    'turn_left': 0,
    'turn_right': 1, 
    'forward': 2,
    'pickup': 3
}

def get_direction_from_positions(from_pos, to_pos):
    """Get the direction needed to face from one position to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if dx == 0 and dy == -1:
        return 0  # Up
    elif dx == 1 and dy == 0:
        return 1  # Right
    elif dx == 0 and dy == 1:
        return 2  # Down
    elif dx == -1 and dy == 0:
        return 3  # Left
    else:
        return None  # Not adjacent or same position

def turn_to_direction(current_dir, target_dir):
    """Calculate the minimum turns needed to face target direction."""
    if current_dir == target_dir:
        return []
    
    # Calculate turns needed (clockwise)
    diff = (target_dir - current_dir) % 4
    
    if diff == 1:  # Turn right once
        return ['turn_right']
    elif diff == 2:  # Turn around (2 rights or 2 lefts, choose rights)
        return ['turn_right', 'turn_right']
    elif diff == 3:  # Turn left once (same as 3 rights)
        return ['turn_left']
    
    return []

def is_valid_position(pos, grid):
    """Check if a position is valid (within bounds and not a wall)."""
    x, y = pos
    if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
        return False
    
    cell = grid.get(x, y)
    if cell is not None and cell.type == 'wall':
        return False
    
    return True

def can_pickup_from_position(agent_pos, agent_dir, ball_pos):
    """Check if agent can pickup ball from current position and direction."""
    # Agent must be facing the ball (ball must be directly in front)
    dx, dy = DIRECTIONS[agent_dir]
    expected_ball_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
    return expected_ball_pos == ball_pos

def find_path_with_minigrid_rules(grid, start_pos, start_dir, goal_pos, color_rewards, step_cost=-0.1, lava_penalty=-3.0):
    """
    Find optimal path using proper MiniGrid movement rules.
    
    Args:
        grid: MiniGrid grid object
        start_pos: Starting position (x, y)
        start_dir: Starting direction (0=up, 1=right, 2=down, 3=left)
        goal_pos: Goal position (x, y)
        color_rewards: Dictionary of ball color rewards
        step_cost: Cost per action
        lava_penalty: Penalty for stepping on lava
    
    Returns:
        dict: {
            'actions': list of actions,
            'path': list of positions,
            'total_reward': expected reward,
            'balls_collected': list of collected balls
        }
    """
    
    # Find all balls and their rewards
    balls = []
    lava_cells = set()
    
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "ball":
                    reward = color_rewards.get(cell.color, 0)
                    if reward > 0:  # Only consider valuable balls
                        balls.append({
                            'pos': (x, y),
                            'color': cell.color,
                            'reward': reward
                        })
                elif cell.type == "lava":
                    lava_cells.add((x, y))
    
    print(f"Found {len(balls)} valuable balls: {balls}")
    
    # Use BFS to find optimal sequence considering all possible ball collection orders
    best_result = None
    best_reward = float('-inf')
    
    # Try different combinations of balls to collect
    # Include the option of collecting no balls
    for num_balls in range(len(balls) + 1):
        for ball_combination in combinations(balls, num_balls):
            # Try different orders of collecting these balls
            if len(ball_combination) == 0:
                # Direct path to goal
                result = find_direct_path(grid, start_pos, start_dir, goal_pos, lava_cells, step_cost, lava_penalty)
                if result and result['total_reward'] > best_reward:
                    best_reward = result['total_reward']
                    best_result = result
            else:
                for ball_order in permutations(ball_combination):
                    result = find_path_through_balls(grid, start_pos, start_dir, ball_order, goal_pos, lava_cells, step_cost, lava_penalty)
                    if result and result['total_reward'] > best_reward:
                        best_reward = result['total_reward']
                        best_result = result
    
    return best_result

def find_direct_path(grid, start_pos, start_dir, goal_pos, lava_cells, step_cost, lava_penalty):
    """Find direct path from start to goal."""
    path_result = bfs_pathfind(grid, start_pos, start_dir, goal_pos, lava_cells)
    
    if not path_result:
        return None
    
    actions, positions = path_result
    
    # Calculate total reward
    total_reward = len(actions) * step_cost
    
    # Add lava penalties
    for pos in positions:
        if pos in lava_cells:
            total_reward += lava_penalty
    
    return {
        'actions': actions,
        'path': positions,
        'total_reward': total_reward,
        'balls_collected': []
    }

def find_path_through_balls(grid, start_pos, start_dir, ball_order, goal_pos, lava_cells, step_cost, lava_penalty):
    """Find path that collects balls in specified order then goes to goal."""
    
    current_pos = start_pos
    current_dir = start_dir
    all_actions = []
    all_positions = [start_pos]
    total_reward = 0
    balls_collected = []
    
    # Visit each ball in order
    for ball in ball_order:
        ball_pos = ball['pos']
        
        # Find path to a position where we can pickup this ball
        pickup_result = find_path_to_pickup_ball(grid, current_pos, current_dir, ball_pos, lava_cells)
        
        if not pickup_result:
            return None  # Can't reach this ball
        
        path_actions, path_positions, final_dir = pickup_result
        
        # Add path actions and positions
        all_actions.extend(path_actions)
        all_positions.extend(path_positions[1:])  # Skip first position (current)
        
        # Add pickup action
        all_actions.append('pickup')
        
        # Update state
        current_pos = path_positions[-1]
        current_dir = final_dir
        total_reward += ball['reward']
        balls_collected.append(ball)
    
    # Finally, go to goal
    path_result = bfs_pathfind(grid, current_pos, current_dir, goal_pos, lava_cells)
    
    if not path_result:
        return None
    
    final_actions, final_positions = path_result
    all_actions.extend(final_actions)
    all_positions.extend(final_positions[1:])  # Skip first position
    
    # Calculate total cost
    total_reward += len(all_actions) * step_cost
    
    # Add lava penalties
    for pos in all_positions:
        if pos in lava_cells:
            total_reward += lava_penalty
    
    return {
        'actions': all_actions,
        'path': all_positions,
        'total_reward': total_reward,
        'balls_collected': balls_collected
    }

def find_path_to_pickup_ball(grid, start_pos, start_dir, ball_pos, lava_cells):
    """Find path to a position where agent can pickup the specified ball."""
    
    # Find all positions adjacent to the ball where agent could stand
    adjacent_positions = []
    for direction in range(4):
        dx, dy = DIRECTIONS[direction]
        adjacent_pos = (ball_pos[0] - dx, ball_pos[1] - dy)  # Position to stand to face ball
        
        if is_valid_position(adjacent_pos, grid):
            adjacent_positions.append((adjacent_pos, direction))
    
    if not adjacent_positions:
        return None  # Ball is not reachable
    
    # Try to find path to each adjacent position
    best_result = None
    best_cost = float('inf')
    
    for target_pos, required_dir in adjacent_positions:
        # Find path to this position with correct direction
        result = bfs_pathfind_with_direction(grid, start_pos, start_dir, target_pos, required_dir, lava_cells)
        
        if result:
            actions, positions = result
            cost = len(actions)
            
            if cost < best_cost:
                best_cost = cost
                best_result = (actions, positions, required_dir)
    
    return best_result

def bfs_pathfind(grid, start_pos, start_dir, goal_pos, lava_cells):
    """Basic BFS pathfinding that reaches goal position with any direction."""
    
    # State: (position, direction)
    queue = deque([(start_pos, start_dir, [], [start_pos])])
    visited = set()
    visited.add((start_pos, start_dir))
    
    while queue:
        (x, y), direction, actions, path = queue.popleft()
        
        # Check if we reached the goal
        if (x, y) == goal_pos:
            return actions, path
        
        # Try all possible actions
        for action_name, action_id in MINIGRID_ACTIONS.items():
            new_pos, new_dir, valid = simulate_action((x, y), direction, action_name, grid)
            
            if not valid:
                continue
            
            # Skip pickup actions when not at goal (we handle pickups separately)
            if action_name == 'pickup':
                continue
            
            state = (new_pos, new_dir)
            if state not in visited:
                visited.add(state)
                new_actions = actions + [action_name]
                new_path = path + [new_pos] if new_pos != (x, y) else path
                queue.append((new_pos, new_dir, new_actions, new_path))
    
    return None  # No path found

def bfs_pathfind_with_direction(grid, start_pos, start_dir, goal_pos, goal_dir, lava_cells):
    """BFS pathfinding that reaches goal position with specific direction."""
    
    queue = deque([(start_pos, start_dir, [], [start_pos])])
    visited = set()
    visited.add((start_pos, start_dir))
    
    while queue:
        (x, y), direction, actions, path = queue.popleft()
        
        # Check if we reached the goal with correct direction
        if (x, y) == goal_pos and direction == goal_dir:
            return actions, path
        
        # Try all possible actions
        for action_name in ['turn_left', 'turn_right', 'forward']:
            new_pos, new_dir, valid = simulate_action((x, y), direction, action_name, grid)
            
            if not valid:
                continue
            
            state = (new_pos, new_dir)
            if state not in visited:
                visited.add(state)
                new_actions = actions + [action_name]
                new_path = path + [new_pos] if new_pos != (x, y) else path
                queue.append((new_pos, new_dir, new_actions, new_path))
    
    return None  # No path found

def simulate_action(pos, direction, action, grid):
    """
    Simulate an action and return new position, direction, and validity.
    
    Returns:
        tuple: (new_pos, new_direction, is_valid)
    """
    x, y = pos
    
    if action == 'turn_left':
        new_dir = (direction - 1) % 4
        return pos, new_dir, True
    
    elif action == 'turn_right':
        new_dir = (direction + 1) % 4
        return pos, new_dir, True
    
    elif action == 'forward':
        dx, dy = DIRECTIONS[direction]
        new_pos = (x + dx, y + dy)
        
        # Check if new position is valid
        if is_valid_position(new_pos, grid):
            return new_pos, direction, True
        else:
            return pos, direction, False  # Invalid move
    
    elif action == 'pickup':
        # Pickup doesn't change position or direction
        return pos, direction, True
    
    return pos, direction, False

def find_optimal_path_proper(env):
    """
    Find the optimal path using proper MiniGrid movement rules.
    
    Args:
        env: Your CustomEnv instance
    
    Returns:
        dict: {
            'actions': list of action names ['turn_left', 'forward', 'pickup', ...],
            'path': list of positions [(x, y), ...],
            'total_reward': expected total reward,
            'balls_collected': list of ball info that will be collected,
            'action_sequence': list of action indices for env.step()
        }
    """
    
    # Get environment state
    grid = env.grid
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    color_rewards = env.color_rewards
    step_cost = getattr(env, 'step_cost', -0.1)
    lava_penalty = getattr(env, 'lava_panishment', -3.0)
    
    # Find goal position
    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and cell.type == "goal":
                goal_pos = (x, y)
                break
    
    if goal_pos is None:
        goal_pos = (grid.width - 2, grid.height - 2)  # Default to bottom-right
    
    print(f"=== PATHFINDING WITH PROPER MINIGRID RULES ===")
    print(f"Agent at: {agent_pos}, facing: {DIRECTION_NAMES[agent_dir]}")
    print(f"Goal at: {goal_pos}")
    print(f"Step cost: {step_cost}, Lava penalty: {lava_penalty}")
    
    # Find optimal path
    result = find_path_with_minigrid_rules(
        grid=grid,
        start_pos=agent_pos,
        start_dir=agent_dir,
        goal_pos=goal_pos,
        color_rewards=color_rewards,
        step_cost=step_cost,
        lava_penalty=lava_penalty
    )
    
    if not result:
        print("❌ No valid path found!")
        return None
    
    # Convert action names to action indices for env.step()
    action_name_to_id = {
        'turn_left': 0,
        'turn_right': 1,
        'forward': 2,
        'pickup': 3
    }
    
    action_sequence = []
    for action_name in result['actions']:
        if action_name in action_name_to_id:
            action_sequence.append(action_name_to_id[action_name])
    
    result['action_sequence'] = action_sequence
    
    # Print results
    print(f"\\n✅ OPTIMAL PATH FOUND!")
    print(f"Total actions: {len(result['actions'])}")
    print(f"Expected reward: {result['total_reward']:.2f}")
    print(f"Balls to collect: {len(result['balls_collected'])}")
    
    if result['balls_collected']:
        print("\\nBalls collection plan:")
        for i, ball in enumerate(result['balls_collected']):
            print(f"  {i+1}. {ball['color']} ball at {ball['pos']} (reward: {ball['reward']:+.1f})")
    
    print(f"\\nAction sequence:")
    for i, action in enumerate(result['actions'][:10]):  # Show first 10 actions
        print(f"  {i+1:2d}. {action}")
    if len(result['actions']) > 10:
        print(f"  ... and {len(result['actions']) - 10} more actions")
    
    return result

def simulate_optimal_path(env, result):
    """
    Simulate the optimal path in the environment (without actually executing it).
    
    Args:
        env: Your CustomEnv instance
        result: Result from find_optimal_path_proper()
    
    Returns:
        dict: Simulation results with step-by-step breakdown
    """
    
    if not result:
        return None
    
    print(f"\\n=== SIMULATING OPTIMAL PATH ===")
    
    # Create a copy of the environment for simulation
    sim_env = copy.deepcopy(env)
    
    total_reward = 0
    step_count = 0
    action_results = []
    
    for i, action_id in enumerate(result['action_sequence']):
        action_name = result['actions'][i]
        
        # Get state before action
        pos_before = sim_env.agent_pos
        dir_before = sim_env.agent_dir
        
        # Execute action
        obs, reward, terminated, truncated, info = sim_env.step(action_id)
        
        pos_after = sim_env.agent_pos
        dir_after = sim_env.agent_dir
        
        total_reward += reward
        step_count += 1
        
        action_results.append({
            'step': step_count,
            'action': action_name,
            'action_id': action_id,
            'pos_before': pos_before,
            'dir_before': DIRECTION_NAMES[dir_before],
            'pos_after': pos_after,
            'dir_after': DIRECTION_NAMES[dir_after],
            'reward': reward,
            'total_reward': total_reward
        })
        
        # Show progress for important actions
        if action_name == 'pickup':
            print(f"Step {step_count:2d}: {action_name:10s} at {pos_after} -> Reward: {reward:+.1f} (Total: {total_reward:+.2f})")
        elif i < 5 or i % 10 == 0:  # Show first few and every 10th action
            print(f"Step {step_count:2d}: {action_name:10s} {pos_before}->{pos_after} facing {DIRECTION_NAMES[dir_after]} -> {reward:+.1f}")
        
        if terminated or truncated:
            print(f"🎯 Reached goal! Total reward: {total_reward:.2f}")
            break
    
    return {
        'total_reward': total_reward,
        'steps_taken': step_count,
        'action_results': action_results,
        'success': terminated and not truncated
    }

def compare_path_strategies(env):
    """
    Compare optimal path strategy vs direct-to-goal strategy.
    """
    
    print("\\n" + "="*50)
    print("STRATEGY COMPARISON")
    print("="*50)
    
    # Strategy 1: Optimal path with ball collection
    optimal_result = find_optimal_path_proper(env)
    
    # Strategy 2: Direct to goal (simulate by setting all ball rewards to 0)
    original_rewards = env.color_rewards.copy()
    env.color_rewards = {color: 0 for color in original_rewards.keys()}
    
    direct_result = find_optimal_path_proper(env)
    
    # Restore original rewards
    env.color_rewards = original_rewards
    
    print(f"\\n📊 COMPARISON RESULTS:")
    
    if optimal_result:
        print(f"\\n🎯 Optimal Strategy (with ball collection):")
        print(f"   Actions needed: {len(optimal_result['actions'])}")
        print(f"   Expected reward: {optimal_result['total_reward']:.2f}")
        print(f"   Balls collected: {len(optimal_result['balls_collected'])}")
    
    if direct_result:
        print(f"\\n🏃 Direct-to-Goal Strategy:")
        print(f"   Actions needed: {len(direct_result['actions'])}")
        print(f"   Expected reward: {direct_result['total_reward']:.2f}")
        print(f"   Balls collected: 0")
    
    if optimal_result and direct_result:
        advantage = optimal_result['total_reward'] - direct_result['total_reward']
        extra_actions = len(optimal_result['actions']) - len(direct_result['actions'])
        
        print(f"\\n🎯 RECOMMENDATION:")
        if advantage > 0:
            print(f"✅ Collect balls! Extra reward: {advantage:+.2f}")
            print(f"   Cost: {extra_actions} additional actions")
            print(f"   Efficiency: {advantage/extra_actions:.3f} reward per extra action")
        else:
            print(f"✅ Go directly to goal!")
            print(f"   Ball collection would lose {abs(advantage):.2f} reward")
    
    return {
        'optimal': optimal_result,
        'direct': direct_result
    }



# run the agent and evaluate his prefermances
# return the average reward and the average number of illegal moves
def evaluate_agent(env, agent, num_episodes=100, from_unique_env=False) -> (float, float, float, int):
    """
    Evaluate the agent on the given environment, return the average reward, the average number of illegal moves, and the average number of moves, numer of reached max steps.
    """
    total_reward = 0
    total_illegal_moves = 0
    total_moves = 0
    reached_max_steps = 0
    for i in range(num_episodes):
        kwargs = {'from_unique_env': 3}
        state = env.unwrapped.reset(**kwargs)
        state = state[0]
        state = {'image': state['image']}
        done = False
        while not (done):
            last_obs = state
            agent_pos_before = env.unwrapped.get_wrapper_attr('agent_pos')
            # print(state)
            action, _ = agent.predict(state, deterministic=True)
            state, reward, done, truncle, _ = env.unwrapped.step(action)
            if truncle:
                reached_max_steps += 1
            state = {'image': state['image']}
            total_reward += reward
            total_moves += 1
            if is_illegal_move(action, last_obs, state, agent_pos_before, env.unwrapped.get_wrapper_attr('agent_pos')):
                total_illegal_moves += 1
        # print("Episode:", i, "Reward:", total_reward, "Illegal moves:", total_illegal_moves, "Total moves:", total_moves)
            
    return total_reward/num_episodes, total_illegal_moves//num_episodes, total_moves//num_episodes, reached_max_steps




