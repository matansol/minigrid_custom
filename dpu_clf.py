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
from minigrid_custom_train import UpgradedObjEnvExtractor


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
def load_agent(env, model_path, update=False) -> PPO:
    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    if update:
        custom_objects = {
        "policy_kwargs": {"features_extractor_class": UpgradedObjEnvExtractor},  # Example kernel size
        "clip_range": 0.2,  # Example custom parameters
        "lr_schedule": 0.001  # Example learning rate schedule
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
    print("Environment Observation Space:", env.observation_space)
    print("PPO Model Observation Space:", ppo.observation_space)
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
        return True
    if action > 2 and np.array_equal(obs['image'], last_obs['image']):
        return True
    return False

actions_translation = {0: 'turn left', 1: 'turn right', 2: 'move forward', 3: 'pickup'}
# resert the environment and run the agent on that environment to find his path

@timeit
def capture_agent_path(copy_env, agent) -> (list, int, int, list): # -> list of moves, number of illegal moves, total reward, list of legal actions
    illigal_moves = 0
    last_obs = copy_env.get_wrapper_attr('current_state')
    
    # last_obs = env.reset()
    # last_obs = last_obs[0]
    ligal_actions = []
    agent_actions = []
    state_record = [last_obs]
    total_reward = 0    
    done = False
    # copy_env = copy.deepcopy(env)
    # plt.imshow(copy_env.render())
    while not done:
        agent_pos_before = copy_env.get_wrapper_attr('agent_pos')
        action, _states = agent.predict(last_obs)
        agent_actions.append(action)
        obs, reward, done, _, info = copy_env.step(action)
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
    small_arrow = 'turn ' # small arrow is used to indicate the agent turning left or right
    agent_dir = "right"
    move_sequence = []
    for action in ligal_actions:
        if action == 0: # turn left
            agent_dir = turn_agent(agent_dir, "left")
            move_sequence.append((small_arrow + agent_dir, 'turn left'))
        elif action == 1: # turn right
            agent_dir = turn_agent(agent_dir, "right")
            move_sequence.append((small_arrow + agent_dir, 'turn right'))
        elif action == 2: # move forward
            move_sequence.append((agent_dir, 'forward'))
        elif action == 3: # pickup
            move_sequence.append(('pickup ' +  agent_dir, 'pickup'))
    return move_sequence, illigal_moves, total_reward, ligal_actions

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
def plot_all_move_sequence(img, move_sequence, agent_true_actions, move_color='y', turn_color='white', pickup_color='purple', converge_action_location = -1): # -> State image with the path of the agent, actions marks locations    
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
    turn_arrow_sizes = {'turn up': (0, -5),
                        'turn down': (0, 5),
                        'turn right': (5, 0),
                        'turn left': (-5, 0)}
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
        full_action = action_dir.split(' ')
        action = full_action[0]
        action_loc = {'action': actual_action}
        # moving action
        if action_dir in move_arrow_sizes.keys(): # a big arrow that represents a move
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
        elif action_dir in turn_arrow_sizes.keys(): # a small arrow that represents a turn or a pickup
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=feedback_action_color, ec=feedback_action_color)
            imgs_action_list.append(ax_to_feedback_image(ax))

            # overide the feedback arrow with the real action
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            shift_size = 17
            turnning_mark_shifts = {'turn up': (0, -shift_size), 'turn down': (0, shift_size), 'turn right': (shift_size, 0), 'turn left': (-shift_size, 0)}
            x_shift, y_shift = turnning_mark_shifts[action_dir]
            # print("turned to:", action_dir)
            
            if action_dir == 'turn up':
                mark_x -= 2
                # mark_y += -10
            elif action_dir == 'turn down':
                mark_x -= 2
                # mark_y += -5
            elif action_dir == 'turn right':
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
        elif action == 'pickup':        
            pickup_position = pickup_direction[full_action[1]]

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
def plot_move_sequence_by_parts(imgs, move_sequence, agent_true_actions, move_color='y', turn_color='white', pickup_color='purple', converge_action_location = -1): # -> State image with the path of the agent, actions marks locations    
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
    turn_arrow_sizes = {'turn up': (0, -5),
                        'turn down': (0, 5),
                        'turn right': (5, 0),
                        'turn left': (-5, 0)}
    pickup_direction = {'up': (0, -1),
                         'down': (0, 1),
                         'left': (-1, 0),
                         'right': (1, 0)}
    # arrows_list = ['right', 'right', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'up', 'right', 'down']

    mark_x, mark_y = start_point[0] + 80, start_point[1] + 40 # 300, 160
    mark_sizes = {'move_vertical': (25, 70), 'move_horizontal': (80, 20), 'turn': (20, 20), 'pickup': (20, 20)}
    # mark_sizes = {'move_vertical': (30, 30), 'move_horizontal': (30, 30), 'turn': (30, 30), 'pickup': (30, 30)}

    # mark_move_sizes = {'move_vertical': 20, 'move_horizontal': 20}
    
    
    current_point = start_point

    i= 0
    print(f'plot move sequence len: {len(move_sequence)}, imgs len: {len(imgs)}')
    for action_dir, actual_action in move_sequence:
        _ , ax = plt.subplots()
        ax.imshow(imgs[i])
        if i == converge_action_location:
            # add a rectangle to mark the converging point:
            ax.add_patch(Rectangle((current_point[0] - 10, current_point[1]- 10), 15, 15, color='b', alpha=0.4))
        i += 1
        full_action = action_dir.split(' ')
        action = full_action[0]
        action_loc = {'action': actual_action}
        # moving action
        if action_dir in move_arrow_sizes.keys(): # a big arrow that represents a move
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=feedback_action_color, ec=feedback_action_color)
            # imgs_action_list.append(ax_to_feedback_image(ax))
            buf = ax_to_feedback_image(ax)

            # overide the feedback arrow with the real action
            # ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=move_color, ec=move_color)
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
        elif action_dir in turn_arrow_sizes.keys(): # a small arrow that represents a turn or a pickup
            # add the action arrow to the feedback arrow and save the image
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=feedback_action_color, ec=feedback_action_color)
            # imgs_action_list.append(ax_to_feedback_image(ax))
            buf = ax_to_feedback_image(ax)
        
            # overide the feedback arrow with the real action
            # ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            shift_size = 17
            turnning_mark_shifts = {'turn up': (0, -shift_size), 'turn down': (0, shift_size), 'turn right': (shift_size, 0), 'turn left': (-shift_size, 0)}
            x_shift, y_shift = turnning_mark_shifts[action_dir]
            # print("turned to:", action_dir)
            
            if action_dir == 'turn up':
                mark_x -= 2
                # mark_y += -10
            elif action_dir == 'turn down':
                mark_x -= 2
                # mark_y += -5
            elif action_dir == 'turn right':
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
        elif action == 'pickup':        
            pickup_position = pickup_direction[full_action[1]]

            action_sign = ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=feedback_action_color)
            # imgs_action_list.append(ax_to_feedback_image(ax))
            buf = ax_to_feedback_image(ax)

            # overide the feedback arrow with the real action
            # ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=pickup_color)
            
            action_loc['x'] = mark_x + 25 * pickup_position[0]
            action_loc['y'] = mark_y + 15 * pickup_position[1]
            action_loc['width'] = mark_sizes['pickup'][0]
            action_loc['height'] = mark_sizes['pickup'][1]
        actions_with_location.append(action_loc)
        imgs_action_list.append(buf)
        plt.close()
        # Capture the current state of the plot
        # Add a yellow half-transparent rectangle for the last action
        

    
    return buf, actions_with_location, imgs_action_list

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

# def plot_buf_images(images_buf_list):
#     # Number of images
#     num_images = len(images_buf_list)
#     print('plot_buf_images, num_images:', num_images)
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
#     if num_images == 1:
#         axes = [axes]  # Ensure axes is iterable if there's only one image
    
#     for ax, img_base64 in zip(axes, images_buf_list):
#         # Decode the base64 image
#         img_data = base64.b64decode(img_base64)
#         img = plt.imread(BytesIO(img_data), format='png')
        
#         # Plot the image
#         ax.imshow(img)
#         ax.axis('off')  # Hide the axis
    
#     plt.show()


# run the agent and evaluate his prefermances
# return the average reward and the average number of illegal moves
def evaluate_agent(env, agent, num_episodes=100) -> (float, float, float):
    """
    Evaluate the agent on the given environment, return the average reward, the average number of illegal moves, and the average number of moves
    """
    total_reward = 0
    total_illegal_moves = 0
    total_moves = 0
    for i in range(num_episodes):
        # kwargs = {'simillarity_level': 3}
        state = env.unwrapped.reset()
        state = state[0]
        state = {'image': state['image']}
        done = False
        while not (done):
            last_obs = state
            agent_pos_before = env.unwrapped.agent_pos
            # print(state)
            action, _ = agent.predict(state)
            state, reward, done, truncle, _ = env.unwrapped.step(action)
            state = {'image': state['image']}
            total_reward += reward
            total_moves += 1
            if is_illegal_move(action, last_obs, state, agent_pos_before, env.unwrapped.agent_pos):
                total_illegal_moves += 1
        # print("Episode:", i, "Reward:", total_reward, "Illegal moves:", total_illegal_moves, "Total moves:", total_moves)
            
    return total_reward/num_episodes, total_illegal_moves//num_episodes, total_moves//num_episodes