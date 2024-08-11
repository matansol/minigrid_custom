from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

from minigrid_custom_env import CustomEnv  
from minigrid.core.actions import Actions
from minigrid_custom_train import ObjEnvExtractor, ObjObsWrapper
from stable_baselines3 import PPO
import numpy as np
from PIL import Image
import io
import base64
import time


app = Flask(__name__)
socketio = SocketIO(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testdb.db'  # Uses a local SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional: Disable modification tracking

# Create the SQLAlchemy db instance
db = SQLAlchemy(app)

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    actions = db.relationship('Action', backref='player', lazy=True)

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50))
    score = db.Column(db.Float)
    reward = db.Column(db.Float)
    done = db.Column(db.Boolean)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'))
    timestamp = db.Column(db.Float)



def image_to_base64(image_array):
    """Convert NumPy array to a base64-encoded PNG."""
    img = Image.fromarray(np.uint8(image_array))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

class PlayerSession:
    def __init__(self, player_name):
        self.player_name = player_name
        self.actions = []

    def record_action(self, action, score, reward, done, agent_action):
        self.actions.append({
            "action": action,
            "agent_action": agent_action,
            "score": score,
            "reward": reward,
            "done": done,
            "timestamp": time.time()
        })

    def save_to_file(self):
        with open(f"users_moves\{self.player_name}_session_data.txt", "w") as file:
            for action in self.actions:
                file.write(f"{action}\n")
                

class ManualControl:
    def __init__(self, env, ppo_agent):
        self.env = env
        self.last_score = None
        self.ppo_agent = ppo_agent
        self.current_obs = None
        
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
            self.last_score = self.score
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
        self.current_obs = observation
        return {'image': image_base64, 'reward': reward, 'done': done, 'score': self.score, 'last_score': self.last_score, 'agent_action': agent_action}

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
    
    def get_initial_observation(self):
        self.current_obs = self.reset()
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        return {'image': image_base64, 'last_score': self.last_score}

    def agent_action(self):
        action, _ = self.ppo_agent.predict(self.current_obs)
        return self.step(action, True)

@app.route('/')
def index():
    return render_template('index.html')
        
@socketio.on('send_action')
def handle_message(action):
    session = players_sessions.get(request.sid)
    response = manual_control.handle_action(action)
    session.record_action(action, response['score'], response['reward'], response['done'], response['agent_action'])
    if response['done']:  # start a new episode
        response = manual_control.get_initial_observation()
        session.save_to_file()  # Save at the end of each episode
        emit('game_update', response)
    else:
        emit('game_update', response, broadcast=True)
        
        

@socketio.on('ppo_action')
def ppo_action():
    response = manual_control.agent_action()
    if response['done']:  # start a new episode
        response = manual_control.get_initial_observation()
        emit('game_update', response)
    else:
        emit('game_update', response, broadcast=True)

@socketio.on('start_game')
def start_game(data):
    player_name = data['playerName']
    response = manual_control.get_initial_observation()
    players_sessions[request.sid] = PlayerSession(player_name)
    emit('game_update', response)
    

def main_agent_control(env, model_path):
    policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Load the model
    ppo = ppo.load(f"models/ppo/{model_path}", env=env)
    return ppo

if __name__ == '__main__':
    players_sessions = {}
    env = CustomEnv(render_mode="rgb_array")
    env = ObjObsWrapper(env)
    model_path = "minigrid_custom_20240804/iter_200000_steps"
    ppo_agent = main_agent_control(env, model_path)
    manual_control = ManualControl(env, ppo_agent)
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
