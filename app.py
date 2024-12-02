from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker

import pymysql
pymysql.install_as_MySQLdb()

from minigrid_custom_env import CustomEnv  
import utils
from minigrid.core.actions import Actions
from minigrid_custom_train import ObjEnvExtractor, ObjObsWrapper
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath
from stable_baselines3 import PPO
import numpy as np
import torch
from PIL import Image
import io
import base64
import time
import os
import matplotlib.pyplot as plt
import copy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testdb.db'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:GmGJtyAIzmnPuEjbUHFPBlTyxfFPvQOO@roundhouse.proxy.rlwy.net:22844/railway'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
socketio = SocketIO(app)

# Setup the engine, typically the same URI
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

# Scoped session that ensures different sessions for different threads
db = SQLAlchemy(app)

def create_database():
    with app.app_context():
        db.create_all()


# classes definition
class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    actions = db.relationship('Action', backref='player', lazy=True)

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50))
    agent_action = db.Column(db.Boolean)
    score = db.Column(db.Float)
    reward = db.Column(db.Float)
    done = db.Column(db.Boolean)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'))
    episode = db.Column(db.Integer)
    timestamp = db.Column(db.Float)
    agent_index = db.Column(db.Integer)

class PlayerSession:
    def __init__(self, player_name):
        self.player_name = player_name
        player = Player.query.filter_by(name=player_name).first()
        if not player:
            player = Player(name=player_name)
            db.session.add(player)
            db.session.commit()
        self.player = player

    def record_action(self, action, score, reward, done, agent_action=False, episode=None, agent_index=None):
        return None # Comment out this line to enable database recording
        new_action = Action(
            action_type=action,
            agent_action=agent_action,
            score=score,
            reward=reward,
            done=done,
            player_id=self.player.id,
            timestamp=time.time(),
            episode=episode,
            agent_index=agent_index
        )
        db.session.add(new_action)
        db.session.commit()


def image_to_base64(image_array):
    """Convert NumPy array to a base64-encoded PNG."""
    img = Image.fromarray(np.uint8(image_array))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

                
SIMMILARITY_CONST = 500
class ManualControl:
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
        self.episode_start = None
        self.invalid_moves = 0
    
    def reset(self):
        obs,_ = self.env.reset()
        self.saved_env = copy.deepcopy(self.env)
        self.score = 0
        self.invalid_moves = 0
        return obs
    
    def actions_to_moves_sequence(self, episode_actions):
        small_arrow = 'turn ' # small arrow is used to indicate the agent turning left or right
        agent_dir = "right"
        move_sequence = []
        for action in episode_actions:
            if action == 0: # turn left
                agent_dir = utils.turn_agent(agent_dir, "left")
                move_sequence.append((small_arrow + agent_dir, 'turn left'))
            elif action == 1: # turn right
                agent_dir = utils.turn_agent(agent_dir, "right")
                move_sequence.append((small_arrow + agent_dir, 'turn right'))
            elif action == 2: # move forward
                move_sequence.append((agent_dir, 'forward'))
            elif action == 3: # pickup
                move_sequence.append(('pickup ' +  agent_dir, 'pickup'))
        return move_sequence
    

    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if not utils.is_illegal_move(action, self.current_obs, observation, self.agent_last_pos, self.env.unwrapped.agent_pos):
            self.episode_actions.append(action)
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
        self.agent_last_pos = self.env.unwrapped.agent_pos
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
    def get_initial_observation(self, update_agent=False):
        print("get_initial_observation")
        if update_agent:
            self.update_agent()
        self.current_obs = self.reset()
        print(f"partial obs: {self.env.unwrapped.partial_obs}")
        self.episode_start = self.env.get_full_obs() # for the overview image
        print(f"partial obs: {self.env.unwrapped.partial_obs}")
        self.agent_last_pos = self.env.unwrapped.agent_pos
        self.episode_actions = []
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        self.episode_num += 1
        print(f"Episode {self.episode_num} started ________________________________________________________________________________________")
        return {'image': image_base64, 'last_score': self.last_score}

    def agent_action(self):
        action, _ = self.ppo_agent.predict(self.current_obs)
        return action, self.step(action, True)
    
    # update the agent to the more trained one, if the current agent is the most trained one, do nothing
    def update_agent(self):
        if self.agent_index is None:
            self.agent_index = 0
        else:
            self.agent_index += 1
        if self.agent_index not in self.models_paths.keys():
            return None
        self.prev_agent = self.ppo_agent
        model_path = self.models_paths[self.agent_index]
        self.ppo_agent = load_agent(self.env, model_path)
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent
        # print(f'laod new model: {self.ppo_agent}')
    
    def find_simillar_env(self, env):
        sim_env = copy.deepcopy(env)
        j = 0
        while True:
            sim_env.reset()
            env_objects = env.grid_objects()
            sim_objects = sim_env.grid_objects()
            if utils.state_distance(env_objects, sim_objects) < SIMMILARITY_CONST or j > 10:
                if j > 10:
                    print("No simillar env found")
                break
            j += 1
        return sim_env
        
        
    def agents_different_routs(self, count=0):
        self.saved_env.reset()
        env = self.find_simillar_env(self.saved_env)
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_obs()
        move_sequence, _, _, _ = utils.capture_agent_path(copy_env, self.ppo_agent)
        
        
        # prev_agent_path
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_obs()
        prev_move_sequence, _, _, _ = utils.capture_agent_path(copy_env, self.prev_agent)
        if prev_move_sequence == move_sequence and count < 5:
            count += 1
            return self.agents_different_routs(count)
        print(f"agents_different_routs {count} times")
        path_img_buffer, _ = utils.plot_move_sequence(img, move_sequence, move_color='c')  # Generate the path image
        prev_path_img_buffer, _ = utils.plot_move_sequence(img, prev_move_sequence)  # Generate the path image
        
        return {'prev_path_image': base64.b64encode(prev_path_img_buffer.getvalue()).decode('ascii'), 'path_image': base64.b64encode(path_img_buffer.getvalue()).decode('ascii')}
        
        
    
    def end_of_episode_summary(self):
        # Generate the path image
        img = self.episode_start
        path_img_buffer, actions_locations = utils.plot_move_sequence(img, self.actions_to_moves_sequence(self.episode_actions))  # Generate the path image

        # Convert the image buffer to base64 so it can be displayed in the frontend
        path_img_base64 = base64.b64encode(path_img_buffer.getvalue()).decode('ascii')
        

        print(f'action locations: {actions_locations}')
        print(f'end of episode, invalid moves: {self.invalid_moves}')
        return {'path_image': path_img_base64, 'actions': actions_locations, 'invalid_moves': self.invalid_moves, 'score': self.last_score}
        
        
# initialize the environment and the manual control object
players_sessions = {}
env = CustomEnv(grid_szie=8, render_mode="rgb_array", image_full_view=False, highlight=True, max_steps=100, lava_cells=3, partial_obs=True)
env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-3.0)

# env = CustomEnv(grid_size=8, difficult_grid=True, max_steps=50, num_objects=3, lava_cells=2, render_mode="rgb_array")
# env = NoDeath(ObjObsWrapper(env), no_death_types=('lava',), death_cost=-2.0)

env.reset()
model_dir1 = "LavaLaver8_20241112"
model_dir2 = "LavaHate8_20241112"
model_paths = {
            0 : model_dir1 + "/iter_250000_steps",
            1:  model_dir1 + "/iter_500000_steps",
            2:  model_dir2 + "/iter_250000_steps",
            3:  model_dir2 + "/iter_500000_steps",
}

# model_paths = {
#             # 0 : model_dir + "/iter_10^5_steps",
#             # 1:  model_dir + "/iter_20^5_steps",
#             # 2:  model_dir + "/iter_30^5_steps",
#             # 3:  model_dir + "/iter_40^5_steps",
#             0:  model_dir + "/iter_50^5_steps",
#             1:  model_dir + "/iter_60^5_steps",
#             2:  model_dir + "/iter_70^5_steps",
#             3:  model_dir + "/iter_80^5_steps",
#             4:  model_dir + "/iter_90^5_steps",
#             5:  model_dir + "/iter_10^6_steps"
# }

manual_control = ManualControl(env, model_paths)
# manual_control.reset()


# functions that control the flow of the game
@app.route('/')
def index():
    return render_template('index2.html')        

@socketio.on('send_action')
def handle_message(action):
    try:
        session = players_sessions.get(request.sid)
        response = manual_control.handle_action(action)
    except Exception as e:
        app.logger.error('Failed to handle action: %s', e)
        return
    
    # TODO: Uncomment the following block to enable database recording
    
    # #insert the action to the database
    # try:
    #     db.session.begin(nested=True)
    #     session.record_action(
    #         action=action,
    #         score=response['score'],
    #         reward=response['reward'],
    #         done=response['done'],
    #         agent_action=response['agent_action'],
    #         episode=response['episode'],
    #         agent_index=response['agent_index']
    #     )
    #     db.session.commit()
    # except Exception as e:
    #     db.session.rollback()
    #     app.logger.error('Database operation failed: %s', e)
    #     emit('error', {'error': 'Database operation failed'})
    # finally:
    #     db.session.remove()

    finish_turn(response)
    # if response['done']:
    #     response = manual_control.get_initial_observation(update_agent=True)
    #     emit('game_update', response)
    # else:
    #     emit('game_update', response, broadcast=True)
        

# Handle 'next_episode' event to start a new episode after user views the path
@socketio.on('next_episode')
def next_episode():
    response = manual_control.get_initial_observation()
    emit('game_update', response)
    
    
@socketio.on('ppo_action')
def ppo_action():
    action, response = manual_control.agent_action()
    session = players_sessions.get(request.sid)
    try:
        db.session.begin(nested=True)
        session.record_action(
            action=action,
            score=response['score'],
            reward=response['reward'],
            done=response['done'],
            agent_action=response['agent_action'],
            episode=response['episode'],
            agent_index=response['agent_index']
        )
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error('Database operation failed: %s', e)
    finally:
        db.session.remove()
    finish_turn(response)

@socketio.on('play_entire_episode')
def play_entire_episode():
    try:
        while True:
            action, response = manual_control.agent_action()
            session = players_sessions.get(request.sid)
            db.session.begin(nested=True)
            session.record_action(
                action=action,
                score=response['score'],
                reward=response['reward'],
                done=response['done'],
                agent_action=response['agent_action'],
                episode=response['episode'],
                agent_index=response['agent_index']
            )
            db.session.commit()
            time.sleep(0.3)
            finish_turn(response)
            if response['done']:
                break
            # if response['done']:  # Start a new episode
            #     response = manual_control.get_initial_observation()
            #     emit('game_update', response)
            #     break
            # else:
            #     emit('game_update', response, broadcast=True)
    except Exception as e:
        db.session.rollback()
        app.logger.error('Database operation failed: %s', e)
        emit('error', {'error': 'Database operation failed'})
    finally:
        pass
        # db.session.remove()


@socketio.on('compare_agents')
def compare_agents():
    res = manual_control.agents_different_routs()
    emit('compare_agents', res)
    
    
def finish_turn(response):
    if response['done']:
        summary = manual_control.end_of_episode_summary()  # Get the episode summary
        emit('episode_finished', summary)  # Send the path and actions to the frontend
    else:
        emit('game_update', response, broadcast=True)

        
@socketio.on('start_game')
def start_game(data):
    player_name = data['playerName']
    players_sessions[request.sid] = PlayerSession(player_name)
    response = manual_control.get_initial_observation(update_agent=True)
    emit('game_update', response)


@socketio.on('finish_game')
def finish_game():
    print("finish_game")
    scores = manual_control.scores_lst
    print("Scores:", scores)  # Server-side console log for debugging
    emit('finish_game', {'scores': scores})

    

def load_agent(env, model_path):
    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    custom_objects = {
        "policy_kwargs": {"features_extractor_class": ObjEnvExtractor},  # Example kernel size
        "clip_range": 0.2,  # Example custom parameters
        "lr_schedule": 0.001  # Example learning rate schedule
    }
    # ppo = PPO("MultiInputPolicy", env, verbose=1)

    # Load the model
    ppo = PPO.load(f"models/{model_path}", custom_objects=custom_objects, env=env)
    return ppo

if __name__ == '__main__':
    print("Starting the server")
    # print(torch.__version__)
    # print(torch.backends.cudnn.enabled) 
    # create_database()
    
    # socketio.run(app, debug=True)
    # Read the port from the environment variable (use 8000 as a default for local testing)
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
