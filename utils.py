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
            move_sequence.append(small_arrow + agent_dir)
        elif action == 1: # turn right
            agent_dir = turn_agent(agent_dir, "right")
            move_sequence.append(small_arrow + agent_dir)
        elif action == 2: # move forward
            move_sequence.append(agent_dir)
        elif action == 3: # pickup
            move_sequence.append('pickup ' +  agent_dir)
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
    mark_sizes = {'move_vertical': (20, 50), 'move_horizontal': (80, 20), 'turn': (20, 20), 'pickup': (20, 20)}
    # mark_sizes = {'move_vertical': (30, 30), 'move_horizontal': (30, 30), 'turn': (30, 30), 'pickup': (30, 30)}

    # mark_move_sizes = {'move_vertical': 20, 'move_horizontal': 20}
    inlarge_factor = 2.3
    fig, ax = plt.subplots()
    ax.imshow(img)
    current_point = start_point
    for action_name in move_sequence:
        full_action = action_name.split(' ')
        action = full_action[0]
        action_loc = {'action': action_name}
        if action_name in move_arrow_sizes.keys(): # a big arrow that represents a move
            ax.arrow(current_point[0], current_point[1], move_arrow_sizes[action_name][0], move_arrow_sizes[action_name][1], head_width=10, head_length=10, fc=move_color, ec=move_color)
            current_point = (current_point[0] + move_arrow_sizes[action_name][2], current_point[1] + move_arrow_sizes[action_name][3])
            mark_size = mark_sizes['move_vertical'] if action_name == 'up' or action_name == 'down' else mark_sizes['move_horizontal']
            action_loc['x'] = current_point[0] + container_x
            action_loc['y'] = current_point[1] + container_y
            action_loc['width'] = mark_size[0]
            action_loc['height'] = mark_size[1]
            # action_loc['width'] = max(move_arrow_sizes[action_name][2], min_width) * inlarge_factor
            # action_loc['height'] = max(move_arrow_sizes[action_name][3], min_hieght) * inlarge_factor
            if action_name == 'up' or action_name == 'down':
                # container_y += action_loc['height'] / inlarge_factor * 0.7
                container_y += 25
            else:
                # container_x += action_loc['width'] / inlarge_factor * 1.1
                container_x += 41
            
        elif action_name in turn_arrow_sizes.keys(): # a small arrow that represents a turn or a pickup
            ax.arrow(current_point[0], current_point[1], turn_arrow_sizes[action_name][0], turn_arrow_sizes[action_name][1], head_width=7, head_length=6, fc=turn_color, ec=turn_color)
            if action_name == 'turn up' or action_name == 'turn down':
                # container_y += action_loc['height'] / inlarge_factor * 0.7
                container_x += 25
            action_loc['x'] = current_point[0] + container_x
            action_loc['y'] = current_point[1] + container_y + 10
            action_loc['width'] = mark_sizes['turn'][0]
            action_loc['height'] = mark_sizes['turn'][1]
            # action_loc['width'] = max(turn_arrow_sizes[action_name][0], min_width) * inlarge_factor
            # action_loc['height'] = max(turn_arrow_sizes[action_name][1], min_hieght) * inlarge_factor
            
            
        elif action == 'pickup':        
            pickup_position = pickup_direction[full_action[1]]
            ax.plot(current_point[0] + small_shift * pickup_position[0], current_point[1] + small_shift*pickup_position[1], marker='*', markersize=8, color=pickup_color)
            action_loc['x'] = current_point[0] + container_x  +small_shift * pickup_position[0]
            action_loc['y'] = current_point[1] + container_y  +small_shift * pickup_position[1]
            action_loc['width'] = mark_sizes['pickup'][0]
            action_loc['height'] = mark_sizes['pickup'][1]
            # action_loc['width'] = min_width * inlarge_factor
            # action_loc['height'] = min_hieght * inlarge_factor
        actions_with_location.append(action_loc)
    
    ax.axis('off')
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    
    # Convert the buffer to an image that can be used by Flask
    return buf, actions_with_location