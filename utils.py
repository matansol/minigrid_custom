import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib
import copy

# from IPython.display import HTML
# from IPython import display
# from IPython.display import clear_output

# from gym.wrappers.record_video import RecordVideo
from gym_minigrid.wrappers import *


from minigrid_custom_env import *
from minigrid_custom_train import *

def load_agent(env, model_path):
    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    custom_objects = {
        "policy_kwargs": {"features_extractor_class": ObjEnvExtractor},  # Example kernel size
        "clip_range": 0.2,  # Example custom parameters
        "lr_schedule": 0.001  # Example learning rate schedule
    }
    # Load the model
    ppo = PPO.load(f"models/{model_path}", custom_objects=custom_objects, env=env)
    return ppo

def is_illegal_move(action, last_obs, obs, agent_pos_befor, agent_pos):
    if action <= 1: # turn is always legal
        return False
    if action == 2 and agent_pos_befor == agent_pos:
        return True
    if action > 2 and np.array_equal(obs['image'], last_obs['image']):
        return True
    return False

# resert the environment and run the agent on that environment to find his path
def capture_agent_path(copy_env, agent):
    illigal_moves = 0
    last_obs = copy_env.unwrapped.current_state
    
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
        agent_pos_before = copy_env.unwrapped.agent_pos
        action, _states = agent.predict(last_obs)
        agent_actions.append(action)
        obs, reward, done, _, info = copy_env.step(action)
        total_reward += reward
        
        if is_illegal_move(action, last_obs, obs, agent_pos_before, copy_env.agent_pos):
            illigal_moves += 1
            continue

        ligal_actions.append(action)
        last_obs = obs
        state_record.append(obs)
        # plt.imshow(copy_env.render())
        # plt.show()
        
        
    number_to_action = {0: 'turn right', 1: 'turn left', 3: 'pickup'}
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
    return move_sequence, illigal_moves, total_reward, agent_actions


def turn_agent(agent_dir, turn_dir):
    turnning_dict = {("up", "left"): "left", ("up", "right"): "right", 
                     ("down", "left"): "right", ("down", "right"): "left",
                     ("left", "left"): "down", ("left", "right"): "up",
                     ("right", "left"): "up", ("right", "right"): "down"}
    return turnning_dict[(agent_dir, turn_dir)]

def plot_move_sequence(img, move_sequence, move_color='y', turn_color='orange', pickup_color='purple'):    
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
    min_width = 10
    min_hieght = 10
    container_x, container_y = 55, 40 # 300, 160
    mark_sizes = {'move_vertical': (25, 50), 'move_horizontal': (80, 20), 'turn': (20, 20), 'pickup': (20, 20)}
    # mark_sizes = {'move_vertical': (30, 30), 'move_horizontal': (30, 30), 'turn': (30, 30), 'pickup': (30, 30)}

    # mark_move_sizes = {'move_vertical': 20, 'move_horizontal': 20}
    inlarge_factor = 2.3
    fig, ax = plt.subplots()
    ax.imshow(img)
    current_point = start_point
    for action_dir, actual_action in move_sequence:
        full_action = action_dir.split(' ')
        action = full_action[0]
        action_loc = {'action': actual_action}
        if action_dir in move_arrow_sizes.keys(): # a big arrow that represents a move
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_dir][0], move_arrow_sizes[action_dir][1], head_width=10, head_length=10, fc=move_color, ec=move_color)
            current_point = (current_point[0] + move_arrow_sizes[action_dir][2], current_point[1] + move_arrow_sizes[action_dir][3])
            mark_size = mark_sizes['move_vertical'] if action_dir == 'up' or action_dir == 'down' else mark_sizes['move_horizontal']
            action_loc['x'] = current_point[0] + container_x
            action_loc['y'] = current_point[1] + container_y
            action_loc['width'] = mark_size[0]
            action_loc['height'] = mark_size[1]
            # action_loc['width'] = max(move_arrow_sizes[action_name][2], min_width) * inlarge_factor
            # action_loc['height'] = max(move_arrow_sizes[action_name][3], min_hieght) * inlarge_factor
            if action_dir == 'up':
                # container_y += action_loc['height'] / inlarge_factor * 0.7
                container_y -= 25
            elif action_dir == 'down':
                container_y += 25
            elif action_dir == 'right':
                # container_x += action_loc['width'] / inlarge_factor * 1.1
                container_x += 41
            else:
                container_x += -40
            
        elif action_dir in turn_arrow_sizes.keys(): # a small arrow that represents a turn or a pickup
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_dir][0], turn_arrow_sizes[action_dir][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            if action_dir == 'turn up' or action_dir == 'turn down':
                action_loc['x'] = current_point[0] + container_x + 30
                action_loc['y'] = current_point[1] + container_y + 10
                container_x += 25
                container_y += -10
            elif action_dir == 'turn right':
                # container_x += -5
                action_loc['x'] = current_point[0] + container_x - 10
                action_loc['y'] = current_point[1] + container_y + 10
                container_y += 10
            else: # turn left
                action_loc['x'] = current_point[0] + container_x - 20
                action_loc['y'] = current_point[1] + container_y + 10
                container_y += 10
                container_x += -40
            # action_loc['x'] = current_point[0] + container_x
            # action_loc['y'] = current_point[1] + container_y + 10
            action_loc['width'] = mark_sizes['turn'][0]
            action_loc['height'] = mark_sizes['turn'][1]
            # action_loc['width'] = max(turn_arrow_sizes[action_name][0], min_width) * inlarge_factor
            # action_loc['height'] = max(turn_arrow_sizes[action_name][1], min_hieght) * inlarge_factor
            
            
        elif action == 'pickup':        
            pickup_position = pickup_direction[full_action[1]]
            print('pickup_position:', pickup_position)
            print('action:', full_action)
            ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=pickup_color)
            action_loc['x'] = current_point[0] + container_x + 30 * pickup_position[0] + 10
            action_loc['y'] = current_point[1] + container_y + 10 * pickup_position[1]
            action_loc['width'] = mark_sizes['pickup'][0]
            action_loc['height'] = mark_sizes['pickup'][1]
            # action_loc['width'] = min_width * inlarge_factor
            # action_loc['height'] = min_hieght * inlarge_factor
        actions_with_location.append(action_loc)
        print(current_point, full_action)
    
    ax.axis('off')
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    
    # Convert the buffer to an image that can be used by Flask
    return buf, actions_with_location



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


def evaluate_agent(env, agent, num_episodes=100):
    total_reward = 0
    total_illegal_moves = 0
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            last_obs = state
            agent_pos_before = env.unwrapped.agent_pos
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if is_illegal_move(action, last_obs, state, agent_pos_before, env.unwrapped.unagent_pos):
                total_illegal_moves += 1
            
    return total_reward/num_episodes, total_illegal_moves//num_episodes