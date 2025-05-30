from flask import Flask, render_template, request, g
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker

import pymysql
pymysql.install_as_MySQLdb()

from minigrid_custom_env import CustomEnv, ObjObsWrapper
from dpu_clf import *
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath
# from stable_baselines3 import PPO
from PIL import Image

import time
import os
import matplotlib.pyplot as plt
import time
import copy
from dotenv import load_dotenv
import os


# ---------------------- FLASK & SOCKETIO SETUP ----------------------
app = Flask(__name__)
socketio = SocketIO(app)

# -------------------- Database configuration ---------------------------
# Load environment variables from .env file
load_dotenv()

# Get the database URI from the environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('AZURE_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

def create_database():
    print("Creating database tables...")
    with app.app_context():
        db.drop_all()

        db.create_all()


# DB classes definition
# class Player(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(50), unique=True, nullable=False)
#     actions = db.relationship('Action', backref='player', lazy=True)

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(20))
    action_type = db.Column(db.String(50))
    agent_action = db.Column(db.Boolean)
    score = db.Column(db.Float)
    reward = db.Column(db.Float)
    done = db.Column(db.Boolean)
    episode = db.Column(db.Integer)
    timestamp = db.Column(db.Float)
    agent_index = db.Column(db.Integer)
    env_state = db.Column(db.String(1000))

class FeedbackAction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(20))
    env_state = db.Column(db.String(1000))
    agent_action = db.Column(db.String(50))
    feedback_action = db.Column(db.String(50))
    action_index = db.Column(db.Integer)
    timestamp = db.Column(db.Float)




SIMMILARITY_CONST = 500
class GameControl:
    def __init__(self, env, models_paths):
        self.env = env
        self.saved_env = None
        self.last_score = None
        self.agent_index = None
        self.ppo_agent = None
        self.prev_agent = None
        self.current_obs = None
        self.models_paths = models_paths
        self.episode_num = 0
        self.scores_lst = []
        self.last_obs = None
        self.episode_actions = []
        self.agent_last_pos = None
        self.episode_obs = []
        self.invalid_moves = 0
        self.user_feedback = None
        self.user_id = None
        self.current_session = None
    
    def reset(self):
        obs,_ = self.env.reset()
        self.saved_env = copy.deepcopy(self.env)
        self.update_agent(None)
        print("reset - saved the env")
        self.score = 0
        self.invalid_moves = 0
        return obs
    
    def actions_to_moves_sequence(self, episode_actions):
        small_arrow = 'turn ' # small arrow is used to indicate the agent turning left or right
        agent_dir = "right"
        move_sequence = []
        for action in episode_actions:
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
        return move_sequence
    

    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if not is_illegal_move(action, self.current_obs, observation, self.agent_last_pos, self.env.get_wrapper_attr('agent_pos')):
            self.episode_actions.append(action)
            self.episode_obs.append(self.env.get_full_obs()) # store the grid image for feedback page
        else:
            self.invalid_moves += 1
            
        self.score += reward
        self.score = round(self.score, 2)
        if done:
            self.scores_lst.append(self.score)
            self.last_score = self.score
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
        self.current_obs = observation
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        return {'image': image_base64, 'episode': self.episode_num, 'reward': reward, 'done': done, 'score': self.score, 'last_score': self.last_score, 'agent_action': agent_action, 'agent_index': self.agent_index}

    def handle_action(self, action_str):
        key_to_action = {
            "ArrowLeft": Actions.left,
            "ArrowRight": Actions.right,
            "ArrowUp": Actions.forward,
            "Space": Actions.toggle,
            "PageUp": Actions.pickup,
            "PageDown": Actions.drop,
            "1": Actions.pickup,
            "2": Actions.drop,
        }
        return self.step(key_to_action[action_str])
    
    # reset the environment and return the observation image. An option to update the agent
    def get_initial_observation(self):
        # print("entered get_initial_observation ", time.time())
        # print("get_initial_observation")
        self.current_obs = self.reset()
        # print(f"partial obs: {self.env.unwrapped.partial_obs}")
        self.episode_obs = [self.env.get_full_obs()] # for the overview image
        # print(f"partial obs: {self.env.unwrapped.partial_obs}")
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        self.episode_actions = []
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        self.episode_num += 1
        print(f"Episode {self.episode_num} started ________________________________________________________________________________________")
        return {'image': image_base64, 'last_score': self.last_score, 'action': None, 'reward': 0, 'done': False, 'score': 0, 'episode': self.episode_num, 'agent_action': False, 'agent_index': self.agent_index}

    def agent_action(self):
        action, _ = self.ppo_agent.predict(self.current_obs)
        action = action.item()
        return action, self.step(action, True)
    
    def update_env_to_action(self, action_index):
        tmp_env = copy.deepcopy(self.saved_env)
        obs = tmp_env.get_wrapper_attr('current_state')
        for action in self.episode_actions[:action_index]:
            obs, r, ter, tru, info = tmp_env.step(action)
        return tmp_env, obs

    # update the agent to the more trained one, if the current agent is the most trained one, do nothing
    def update_agent(self, data):
        if self.ppo_agent is None:
            self.ppo_agent = load_agent(self.env, self.models_paths[0][0])
            self.prev_agent = self.ppo_agent
            print('load the first model, return')
            return None
        # assert data is not None, "data is None"
        if data is None: # should never happend
            print("Data is None, return")
            return None
        if data['updateAgent'] == False:
            print("No need for update, return")
            return None
        self.user_feedback = data['userFeedback']
        if self.user_feedback is None or len(self.user_feedback) == 0:
            print("No user feedback, return")
            return None

        # DB code
        # Insert the action_feedback to the database
        
        if save_to_db:
            for action_feedback in self.user_feedback:
                try:
                    _, obs = self.update_env_to_action(action_index=action_feedback['index'])
                    agent_action = data['actions'][action_feedback['index']]['action']
                    # store the action in the database
                    feedback_action = FeedbackAction(
                        user_id=self.user_id,
                        env_state="some state", #str("obs['image']"),  # Ensure env_state is passed correctly
                        agent_action = actions_dict[agent_action],
                        feedback_action = actions_dict[action_feedback['feedback_action']],
                        action_index = action_feedback['index'],
                        timestamp = time.time(),
                    )
                    print("store feedback action into the database", feedback_action.agent_action, "->", feedback_action.feedback_action)
                    db.session.add(feedback_action)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    app.logger.error('Database operation failed: %s', e)
                    emit('error', {'error': 'Database operation failed'})
                finally:
                    db.session.remove()


        optional_models = []
        most_correct = 0
        tmp_agent = None
        for path in self.models_paths:
            agent = load_agent(self.env, path[0])
            print(f'checking model: {path[2]}')
            model_correctness = 0
            for action_feedback in self.user_feedback:
                tmp_env, obs = self.update_env_to_action(action_index=action_feedback['index'])
                model_action = agent.predict(obs)
                print(f"model_action[0].item()={model_action[0].item()},  action_feedback['action']={action_feedback['feedback_action']}")
                model_action = actions_dict[model_action[0].item()]
                print(f"feedback_action: {actions_dict[action_feedback['feedback_action']]}, model_predict_action: {model_action}")
                if model_action == actions_dict[action_feedback['feedback_action']]:
                    print("model is correct")
                    model_correctness += 1
            
            # if model_correctness > most_correct:
            #     most_correct = model_correctness
            #     tmp_agent = (agent, path[2])
                
            if len(self.user_feedback) - model_correctness <= 2: # number of mistakes allowed
                optional_models.append((agent, path[2]))
        
        print(f'optional_models: {optional_models}')
        if len(optional_models) == 0:
            print("No optional models, return")
            return None
        agent_tuple = random.choice(optional_models) # choose 1 agent from the optional models
        # agent_tuple = tmp_agent
        # if agent_tuple is None:
        #     print("No optional models, return")
        #     return None
        self.ppo_agent = agent_tuple[0]
        print(f'load new model: {agent_tuple[1]}')

        # model_path = self.models_paths[self.agent_index][0]
        # self.ppo_agent = load_agent(self.env, model_path)
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent

        # print(f'laod new model: {self.ppo_agent}')
        
        
    def agents_different_routs(self, count=0):
        if self.ppo_agent == None or self.prev_agent == None:
            print(f"No two agents to compare ppo_agent: {self.ppo_agent}, prev_agent: {self.prev_agent}")
            if self.ppo_agent == None:
                self.ppo_agent = self.prev_agent
            else:
                self.prev_agent = self.ppo_agent
        self.saved_env.reset()
        env = self.find_simillar_env(self.saved_env)
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_obs()
        move_sequence, _, _, agent_actions = capture_agent_path(copy_env, self.ppo_agent)
        
        
        # prev_agent_path
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_obs()
        prev_move_sequence, _, _, prev_agent_actions = capture_agent_path(copy_env, self.prev_agent)
        if prev_move_sequence == move_sequence and count < 5:
            count += 1
            return self.agents_different_routs(count)
        print(f"agents_different_routs {count} times")
        converge_action_index = -1
        for i in range(len(move_sequence)):
            if move_sequence[i] != prev_move_sequence[i]:
                converge_action_index = i
                break
        path_img_buffer, _, _ = plot_all_move_sequence(img, move_sequence, agent_actions, move_color='c', converge_action_location=converge_action_index)  # Generate the path image
        prev_path_img_buffer, _, _ = plot_all_move_sequence(img, prev_move_sequence, prev_agent_actions, converge_action_location=converge_action_index)  # Generate the path image
        
        return {'prev_path_image': prev_path_img_buffer, 'path_image': path_img_buffer}
        
  
    
    def end_of_episode_summary(self):
        # Generate the path image
        imgs = self.episode_obs
        path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(imgs, 
                                self.actions_to_moves_sequence(self.episode_actions), self.episode_actions)  # Generate the path image  

        return {'path_image': path_img_base64, 
                'actions': actions_locations, 
                'invalid_moves': self.invalid_moves, 
                'score': self.last_score, 
                'feedback_images': images_buf_list}
        

    def find_simillar_env(self, env, deploy=False):
        sim_env = copy.deepcopy(env)
        j = 0
        while True:
            sim_env.reset()
            if not deploy: # for testing we do not care what the new env is
                return sim_env
            env_objects = env.grid_objects()
            sim_objects = sim_env.grid_objects()
            if state_distance(env_objects, sim_objects) < SIMMILARITY_CONST or j > 10:
                if j > 10:
                    print("No simillar env found")
                break
            j += 1
        return sim_env
    
# functions that control the flow of the game
@app.route('/')
def index():
    return render_template('index.html')        

@socketio.on('send_action')
def handle_message(action):
    response = game_control.handle_action(action)
    response['action'] = action_dir[action]

    if save_to_db:
        try:
            # store the action in the database
            new_action = Action(
                    action_type=action,
                    agent_action=response['agent_action'],
                    score=response['score'],
                    reward=response['reward'],
                    done=response['done'],
                    user_id=game_control.user_id,
                    timestamp=time.time(),
                    episode=response['episode'],
                    agent_index=response['agent_index'],
                    env_state='some state',
                )
            db.session.add(new_action)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error('Database operation failed: %s', e)
            emit('error', {'error': 'Database operation failed'})
        finally:
            db.session.remove()

    finish_turn(response)
        

# Handle 'next_episode' event to start a new episode after user views the path
@socketio.on('next_episode')
def next_episode():
    response = game_control.get_initial_observation()
    emit('game_update', response)
    
    
@socketio.on('ppo_action')
def ppo_action():
    action, response = game_control.agent_action()
    response['action'] = action
    finish_turn(response)

@socketio.on('play_entire_episode')
def play_entire_episode():
     while True:
        action, response = game_control.agent_action()
        # print(f"Agent action: {action}")

        time.sleep(0.3)
        finish_turn(response)
        if response['done']:
            print("Agent Episode finished")
            break


@socketio.on('compare_agents')
def compare_agents(data):
    game_control.update_agent(data)
    res = game_control.agents_different_routs()
    emit('compare_agents', res)
    
    
def finish_turn(response):
    if response['done']:
        summary = game_control.end_of_episode_summary()  # Get the episode summary
        emit('episode_finished', summary)  # Send the path and actions to the frontend
    else:
        emit('game_update', response, broadcast=True)

        
@socketio.on('start_game')
def start_game(data):
    print("starting the game")
    player_name = data['playerName']
    game_control.user_id = player_name
    if data['updateAgent']:
        game_control.update_agent(data)
    response = game_control.get_initial_observation()
    emit('game_update', response)


@socketio.on('finish_game')
def finish_game():
    print("finish_game")
    scores = game_control.scores_lst
    print("Scores:", scores)  # Server-side console log for debugging
    emit('finish_game', {'scores': scores})


# creating env and game control objects

players_sessions = {}
# unique_env_id = 3
unique_env_id = 0
env = CustomEnv(grid_szie=8, render_mode="rgb_array", image_full_view=False, highlight=True, max_steps=100, lava_cells=3, partial_obs=True, unique_env=unique_env_id)
env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-3.0)
env.reset()


# Preference vector: (red ball, green ball, blue ball, lava, step penalty)
model_paths = [
            (os.path.join("models", "LavaLaver8_20241112", "iter_500000_steps.zip"), (2, 2 ,2, 0, -0.1), "LavaLaver8_20241112"),
            (os.path.join("models", "LavaHate8_20241112", "iter_500000_steps.zip"), (2, 2 ,2, -3, -0.1), "LavaHate8_20241112"),
            (os.path.join("models", "2,2,2,-3,0.2Steps100Grid8_20241230", "best_model.zip"), (2, 2 ,2, -3, -0.2), "LavaHate8_20241229"),
            (os.path.join("models", "0,5,0,-3,0.2Steps100Grid8_20241231", "best_model.zip"), (0, 5 ,0, -3, -0.2), "GreenOnly8_20241231"),
]
actions_dict = {0: Actions.left, 1: Actions.right, 2: Actions.forward, 3: Actions.pickup, 4: Actions.drop, 5: Actions.toggle, 6: Actions.done,
            'turn left': Actions.left, 'turn right': Actions.right, 'forward': Actions.forward, 'pickup': Actions.pickup}


game_control= GameControl(env, model_paths)
game_control.reset()

action_dir = {'ArrowLeft': 'Turn left', 'ArrowRight': 'Turn right', 'ArrowUp':
            'Move forward', 'Space': 'Toggle', 'PageUp': 'Pickup', 'PageDown': 'Drop', '1': 'Pickup', '2': 'Drop'}

save_to_db = False
if __name__ == '__main__':
    print("Starting the server")
    
    if save_to_db:
        create_database()
    
    # Read the port from the environment variable (use 8000 as a default for local testing)
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
