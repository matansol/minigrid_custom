from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker

import pymysql
pymysql.install_as_MySQLdb()

from minigrid_custom_env import CustomEnv  
from minigrid.core.actions import Actions
from minigrid_custom_train import ObjEnvExtractor, ObjObsWrapper
from stable_baselines3 import PPO
import numpy as np
import torch
from PIL import Image
import io
import base64
import time
import os


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

                

class ManualControl:
    def __init__(self, env, agents_paths):
        self.env = env
        self.last_score = None
        self.agent_index = None
        self.ppo_agent = None
        self.current_obs = None
        self.agent_paths = agents_paths
        self.episode_num = 0
        self.scores_lst = []
        
    def reset(self):
        obs,_ = self.env.reset()
        self.score = 0
        return obs

    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.score += reward
        self.score = round(self.score, 2)
        if done:
            self.scores_lst.append(self.score)
            self.last_score = self.score
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
        self.current_obs = observation
        return {'image': image_base64, 'episode': self.episode_num, 'reward': reward, 'done': done, 'score': self.score, 'last_score': self.last_score, 'agent_action': agent_action, 'agent_index': self.agent_index}

    def handle_action(self, action_str):
        key_to_action = {
            "ArrowLeft": Actions.left,
            "ArrowRight": Actions.right,
            "ArrowUp": Actions.forward,
            "Space": Actions.pickup,
            "PageUp": Actions.pickup,
            "PageDown": Actions.drop,
            "1": Actions.pickup,
            "2": Actions.drop,
        }
        return self.step(key_to_action[action_str])
    
    # reset the environment and return the observation image. An option to update the agent
    def get_initial_observation(self, update_agent=False):
        if update_agent:
            self.update_agent()
        self.current_obs = self.reset()
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
        if self.agent_index not in self.agent_paths.keys():
            return None
        model_path = self.agent_paths[self.agent_index]
        self.ppo_agent = load_agent(self.env, model_path)
        
        
# initialize the environment and the manual control object
players_sessions = {}
env = CustomEnv(render_mode="rgb_array")
env = ObjObsWrapper(env)

env.reset()
model_dir = "minigrid_custom_20240907"
model_paths = {0 : model_dir + "/iter_10^5_steps",
            1:  model_dir + "/iter_20^5_steps",
            2:  model_dir + "/iter_30^5_steps",
            3:  model_dir + "/iter_40^5_steps",
            4:  model_dir + "/iter_50^5_steps",
            5:  model_dir + "/iter_60^5_steps",
            6:  model_dir + "/iter_70^5_steps",
            7:  model_dir + "/iter_80^5_steps",
            8:  model_dir + "/iter_90^5_steps",
            9:  model_dir + "/iter_10^6_steps"
}

manual_control = ManualControl(env, model_paths)



# functions that control the flow of the game
@app.route('/')
def index():
    return render_template('index.html')        

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

    if response['done']:
        response = manual_control.get_initial_observation(update_agent=True)
        emit('game_update', response)
    else:
        emit('game_update', response, broadcast=True)
        

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
        emit('error', {'error': 'Database operation failed'})
    finally:
        db.session.remove()
    if response['done']:  # start a new episode
        response = manual_control.get_initial_observation()
        emit('game_update', response)
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
    scores = manual_control.scores_lst
    print("Emitting scores:", scores)  # Server-side console log for debugging
    emit('game_finished', {'scores': scores})

    

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
    print(torch.__version__)
    print(torch.backends.cudnn.enabled) 
    # create_database()
    
    # socketio.run(app, debug=True)
    # Read the port from the environment variable (use 8000 as a default for local testing)
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
