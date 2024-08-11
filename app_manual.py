from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from minigrid_custom_env import CustomEnv  
from minigrid.core.actions import Actions
from minigrid_custom_train import ObjEnvExtractor, ObjObsWrapper
from stable_baselines3 import PPO
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
socketio = SocketIO(app)

def image_to_base64(image_array):
    """Convert NumPy array to a base64-encoded PNG."""
    img = Image.fromarray(np.uint8(image_array))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

class ManualControl:
    def __init__(self, env):
        self.env = env
        self.last_score = None

    def reset(self):
        self.env.reset()
        self.score = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.score += reward
        self.score = round(self.score, 2)
        if done:
            self.last_score = self.score
            # print(f'done..... score: {self.score}, last_score: {self.last_score}')
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
        return {'image': image_base64, 'reward': reward, 'done': done, 'score': self.score, 'last_score': self.last_score}

    def handle_action(self, action_str):
        key_to_action = {
            "ArrowLeft": Actions.left,
            "ArrowRight": Actions.right,
            "ArrowUp": Actions.forward,
            "Space": Actions.pickup,
            "PageUp": Actions.pickup,
            "PageDown": Actions.drop,
            "Tab": Actions.pickup,
            "Shift": Actions.drop,
            "1": Actions.pickup,
            "2": Actions.drop,
            # "Enter": Actions.done
        }
        return self.step(key_to_action[action_str])
    
    def get_initial_observation(self):
        self.reset()
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        return {'image': image_base64, 'last_score': self.last_score}

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('send_action')
def handle_message(action):
    response = manual_control.handle_action(action)
    if response['done']: # start a new episode
        response = manual_control.get_initial_observation()
        emit('game_update', response)
    else:
        emit('game_update', response, broadcast=True)
    

@socketio.on('start_game')
def start_game():
    response = manual_control.get_initial_observation()
    emit('game_update', response)
    
    

def main_agent_control():
    env = CustomEnv()
    env = ObjObsWrapper(env)
    policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # load the model
    load_model = "minigrid_custom_20240723/iter_1000000_steps"
    ppo = ppo.load(f"models/ppo/{load_model}", env=env)

if __name__ == '__main__':
    env = CustomEnv(render_mode="rgb_array")
    manual_control = ManualControl(env)
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
    
    

