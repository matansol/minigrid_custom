import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import socketio
from minigrid_custom_env import CustomEnv, ObjObsWrapper
from minigrid.wrappers import NoDeath
from minigrid.core.actions import Actions
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()

# FastAPI application
app = FastAPI()

# Socket.IO server
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Wrap the FastAPI app with Socket.IO's ASGI application
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Templates
templates = Jinja2Templates(directory="templates")

# SQLAlchemy setup
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///tutorial.db")
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# Global variable to control database saving
save_to_db = False

class Tutorial_Action(Base):
    __tablename__ = "tutorial_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20))
    action_type = Column(String(50))
    score = Column(Float)
    reward = Column(Float)
    done = Column(Boolean)
    episode = Column(Integer)
    timestamp = Column(Float)
    env_state = Column(String(1000))

def create_database():
    """Creates the database tables if they do not already exist."""
    print("Ensuring database tables are created...")
    Base.metadata.create_all(bind=engine)

def encode_image(img_array):
    """Convert numpy array to base64 encoded image"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return None

class TutorialGameControl:
    def __init__(self, env):
        self.env = env
        self.episode_num = 0
        self.score = 0
        self.last_score = 0
        self.episode_actions = []
        self.episode_images = []
        self.current_obs = None
        self.agent_last_pos = None

    def reset(self):
        obs, _ = self.env.unwrapped.reset()
        if 'direction' in obs:
            obs = {'image': obs['image']}
        self.score = 0
        self.episode_actions = []
        self.episode_images = [self.env.get_full_image()]
        self.current_obs = obs
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        return obs

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episode_actions.append(action)
        self.episode_images.append(self.env.get_full_image())
        self.score += reward
        self.score = round(self.score, 2)
        if done:
            self.last_score = self.score
        img = self.env.render()
        self.current_obs = observation
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        return {
            'image': encode_image(img),
            'episode': self.episode_num,
            'reward': float(reward),
            'done': done,
            'score': float(self.score),
            'last_score': float(self.last_score),
            'step_count': int(self.env.get_wrapper_attr('step_count'))
        }

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

    def get_initial_observation(self):
        self.current_obs = self.reset()
        img = self.env.render()
        self.episode_num += 1
        return {
            'image': encode_image(img),
            'last_score': float(self.last_score),
            'action': None,
            'reward': 0.0,
            'done': False,
            'score': 0.0,
            'episode': self.episode_num,
            'step_count': int(self.env.get_wrapper_attr('step_count'))
        }

# Global variables for multi-user support
game_controls = {}
sid_to_user = {}

def create_new_env():
    env_instance = CustomEnv(grid_size=8, 
                           render_mode="rgb_array", 
                           image_full_view=False,
                           highlight=True, 
                           max_steps=50, 
                           num_objects=5, 
                           lava_cells=4, 
                           partial_obs=True)
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=-3.0)
    return env_instance

# FastAPI Routes
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("tutorial_index.html", {"request": request})

# Socket.IO Events
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    if sid in sid_to_user:
        del sid_to_user[sid]

@sio.on("start_game")
async def start_game(sid, data):
    user_id = data["playerName"]
    sid_to_user[sid] = user_id
    if user_id not in game_controls:
        env_instance = create_new_env()
        new_game = TutorialGameControl(env_instance)
        game_controls[user_id] = new_game
    else:
        new_game = game_controls[user_id]
    response = new_game.get_initial_observation()
    response['action'] = None
    await sio.emit("game_update", response, to=sid)

@sio.on("send_action")
async def handle_send_action(sid, action):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.handle_action(action)
    response["action"] = action

    if save_to_db:
        session = SessionLocal()
        try:
            new_action = Tutorial_Action(
                action_type=action,
                score=response["score"],
                reward=response["reward"],
                done=response["done"],
                user_id=user_id,
                timestamp=time.time(),
                episode=response["episode"],
                env_state=str(response.get("image", "")),  # Store the game state image
            )
            session.add(new_action)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database operation failed: {e}")
            await sio.emit("error", {"error": "Database operation failed"}, to=sid)
        finally:
            session.close()

    if response["done"]:
        await sio.emit("episode_finished", response, to=sid)
    else:
        await sio.emit("game_update", response, to=sid)

@sio.on("next_episode")
async def next_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.get_initial_observation()
    await sio.emit("game_update", response, to=sid)

if __name__ == "__main__":
    save_to_db = False  # Set to True to enable database saving
    if save_to_db:
        create_database()

    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001))
    ) 