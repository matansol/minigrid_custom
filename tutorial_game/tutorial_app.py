import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import datetime
import base64
import numpy as np
import asyncio
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
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

app = FastAPI()

# Socket.IO server configuration
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


class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    timestamp = Column(String(30))
    simillarity_level = Column(Integer)
    final_score = Column(Float, default=0.0)  # Default to 0.0 if not set



def create_database():
    """Creates the database tables if they do not already exist."""
    print("Ensuring database tables are created...")
    Base.metadata.create_all(bind=engine)

def clear_database():
    """Clears the database tables."""
    print("Clearing database tables...")
    Base.metadata.drop_all(bind=engine)

def encode_image(img_array):
    """Convert numpy array to base64 encoded image"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return None

class TutorialGameControl:
    def __init__(self, env, final_step=False):
        self.env = env
        self.episode_num = 0
        self.score = 0
        self.last_score = 0
        self.episode_actions = []
        self.episode_images = []
        self.current_obs = None
        self.agent_last_pos = None
        self.final_step = final_step

        
    def reset(self):
        # if self.final_step:
        #     # Special reset for final step with unique_env=100 and from_unique_env=True
        #     obs, _ = self.env.unwrapped.reset(unique_env=100, from_unique_env=True)
        # else:
        #     obs, _ = self.env.unwrapped.reset()
        obs, _ = self.env.unwrapped.reset()
        if 'direction' in obs:
            obs = {'image': obs['image']}
        self.score = 0
        self.episode_actions = []
        self.episode_images = [self.env.get_full_image()]
        self.current_obs = obs
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        self.episode_done = False  # Reset episode finished flag
        return obs

    def step(self, action):
        if self.episode_done:
            return
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episode_actions.append(action)
        self.episode_images.append(self.env.get_full_image())
        reward = round(float(reward), 1)
        self.score += reward
        self.score = round(self.score, 1)
        if done:
            self.last_score = self.score
            self.episode_done = True  # Mark episode as finished
        img = self.env.render()
        self.current_obs = observation
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        return {
            'image': encode_image(img),
            'episode': self.episode_num,
            'reward': reward,
            'done': done,
            'score': self.score,
            'last_score': self.last_score,
            'episode_finished': self.episode_done,
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
            'episode_finished': self.episode_done,
            'step_count': int(self.env.get_wrapper_attr('step_count'))
        }

# Global variables for multi-user support
import asyncio
game_controls = {}
sid_to_user = {}
# Add locks for thread safety
game_controls_lock = asyncio.Lock()
sid_to_user_lock = asyncio.Lock()

def create_new_env():
    env_instance = CustomEnv(grid_size=8, 
                           render_mode="rgb_array", 
                           image_full_view=False,
                           highlight=True, 
                           max_steps=70, 
                           num_objects=5, 
                           num_lava_cells=4, 
                           partial_obs=True)
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=-3.0)
    return env_instance

# FastAPI Routes
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("tutorial_index.html", {"request": request})

@app.get("/tutorial")
def tutorial(request: Request):
    return templates.TemplateResponse("tutorial_index.html", {"request": request})

# Serve a no-content favicon to avoid browser 404s during local dev
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# Socket.IO Events
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    
@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Clean up user mapping and game control with thread safety
    async with sid_to_user_lock:
        if sid in sid_to_user:
            user_id = sid_to_user[sid]
            # Remove the sid mapping first
            del sid_to_user[sid]

            # Only clean up the user's game if no other sockets are still mapped to this user
            other_sids_for_user = [s for s, uid in sid_to_user.items() if uid == user_id]
            if other_sids_for_user:
                print(f"Not cleaning game for user {user_id}; other active sockets: {len(other_sids_for_user)}")
                return

            # Critical: Clean up game instance to prevent memory leaks (last socket for this user disconnected)
            async with game_controls_lock:
                if user_id in game_controls:
                    game_instance = game_controls[user_id]
                    # Clear memory-intensive data
                    if hasattr(game_instance, 'episode_images'):
                        game_instance.episode_images.clear()
                    if hasattr(game_instance, 'episode_actions'):
                        game_instance.episode_actions.clear()
                    # Close environment if it has cleanup methods
                    if hasattr(game_instance.env, 'close'):
                        try:
                            game_instance.env.close()
                        except:
                            pass
                    del game_controls[user_id]
                    print(f"Cleaned up resources for user: {user_id}")


@sio.event
async def start_game(sid, data):
    user_id = data["playerName"]
    final_step = data.get("finalStep", 0) == 1
    
    async with sid_to_user_lock:
        sid_to_user[sid] = user_id
    
    async with game_controls_lock:
        if user_id not in game_controls:
            env_instance = create_new_env()
            new_game = TutorialGameControl(env_instance, final_step=final_step)
            game_controls[user_id] = new_game
        else:
            new_game = game_controls[user_id]
            # new_game.final_step = final_step
    
    response = new_game.get_initial_observation()
    response['action'] = None
    await sio.emit("game_update", response, to=sid)

@sio.event
async def send_action(sid, action):
    async with sid_to_user_lock:
        user_id = sid_to_user.get(sid)
    
    if not user_id:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
        
    async with game_controls_lock:
        if user_id not in game_controls:
            await sio.emit("error", {"error": "Game not found"}, to=sid)
            return
        user_game = game_controls[user_id]

    # Accept both string and dict payloads for action to be robust with different clients
    if isinstance(action, dict):
        action = action.get('action') or action.get('key') or action.get('code')

    if not isinstance(action, str):
        await sio.emit("error", {"error": "Invalid action payload"}, to=sid)
        return

    response = user_game.handle_action(action)
    if response is None:
        # await sio.emit("error", {"error": "Episode ended"}, to=sid)
        return
    response["action"] = action

    if response["done"]:
        await sio.emit("episode_finished", response, to=sid)
    else:
        await sio.emit("game_update", response, to=sid)

@sio.event
async def next_episode(sid):
    async with sid_to_user_lock:
        user_id = sid_to_user.get(sid)
    
    if not user_id:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
        
    async with game_controls_lock:
        if user_id not in game_controls:
            await sio.emit("error", {"error": "Game not found"}, to=sid)
            return
        user_game = game_controls[user_id]
    
    response = user_game.get_initial_observation()
    await sio.emit("game_update", response, to=sid)

save_to_db = True  # Set to True to enable database saving
if __name__ == "__main__":
    print("=== Starting Tutorial App ===", flush=True)
    # if save_to_db:
    #     print("Creating database tables...", flush=True)
    #     create_database()

    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001))
    ) 