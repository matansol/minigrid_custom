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

# Redis configuration for scaling (optional)
REDIS_URL = os.getenv("REDIS_URL", None)
if REDIS_URL:
    try:
        from socketio import AsyncRedisManager
        print(f"Using Redis manager at: {REDIS_URL}")
        mgr = AsyncRedisManager(REDIS_URL)
    except ImportError:
        print("Redis manager requested but python-socketio[asyncio_client] not installed")
        mgr = None
else:
    print("No Redis URL provided, using default manager")
    mgr = None

# FastAPI application for Final App
app = FastAPI(title="Final Game App")

# Socket.IO server configuration - allow polling for local testing like tutorial_app
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Wrap the FastAPI app with Socket.IO's ASGI application
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Templates
templates = Jinja2Templates(directory="templates")

# SQLAlchemy setup
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class Users(Base):
    __tablename__ = "users"  # Use the same table as the main app
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    timestamp = Column(String(30))
    similarity_level = Column(Integer)
    final_score = Column(Float, default=0.0)

def encode_image(img_array):
    """Convert numpy array to base64 encoded image"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return None

class FinalGameControl:
    def __init__(self, env):
        self.env = env
        self.episode_num = 0
        self.score = 0
        self.last_score = 0
        self.episode_actions = []
        self.episode_images = []
        self.current_obs = None
        self.agent_last_pos = None
        self.episode_finished = False  # Track if current episode is finished
        
    def reset(self):
        # Always use final step configuration with unique_env=100 and from_unique_env=True
        obs, _ = self.env.unwrapped.reset(unique_env=100, from_unique_env=True)
        if 'direction' in obs:
            obs = {'image': obs['image']}
        self.score = 0
        self.episode_actions = []
        self.episode_images = [self.env.get_full_image()]
        self.current_obs = obs
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        self.episode_finished = False  # Reset episode finished flag
        return obs

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episode_actions.append(action)
        self.episode_images.append(self.env.get_full_image())
        reward = round(float(reward), 1)
        self.score += reward
        self.score = round(self.score, 1)
        if done:
            self.last_score = self.score
            self.episode_finished = True  # Mark episode as finished
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
            'step_count': int(self.env.get_wrapper_attr('step_count'))
        }

    def handle_action(self, action_str):
        # Prevent actions if episode is already finished
        if self.episode_finished:
            # Return the current state without taking any action
            img = self.env.render()
            return {
                'image': encode_image(img),
                'episode': self.episode_num,
                'reward': 0.0,
                'done': True,
                'score': self.score,
                'last_score': self.last_score,
                'step_count': int(self.env.get_wrapper_attr('step_count'))
            }
        
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

# Global variables for Final App only
final_game_controls = {}
final_sid_to_user = {}

def create_new_env():
    env_instance = CustomEnv(grid_size=8, 
                           render_mode="rgb_array", 
                           image_full_view=False,
                           highlight=True, 
                           max_steps=70, 
                           num_objects=5, 
                           lava_cells=4, 
                           partial_obs=True)
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=-3.0)
    return env_instance

# FastAPI Routes for Final App
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("final_index.html", {"request": request})

@app.get("/final")
def final_route(request: Request):
    return templates.TemplateResponse("final_index.html", {"request": request})

# Serve a no-content favicon to avoid browser 404s during local dev
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health_check():
    """Health check endpoint for Azure Container Apps"""
    return {
        "status": "healthy",
        "app_type": "final",
        "active_connections": len(final_sid_to_user),
        "active_games": len(final_game_controls),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# Socket.IO Events with enhanced error handling for WebSocket-only mode
@sio.event
async def connect(sid, environ):
    print(f"Final App - WebSocket client connected: {sid}")
    # Store connection info for better debugging
    user_agent = environ.get('HTTP_USER_AGENT', 'Unknown')
    transport = environ.get('transport', 'Unknown')
    print(f"User agent: {user_agent[:100]}...")
    print(f"Transport: {transport}")
    # Send immediate acknowledgment to confirm connection
    await sio.emit("connection_confirmed", {"status": "connected", "transport": "websocket", "app": "final"}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"Final App - Client disconnected: {sid}")
    # Clean up user mapping and game control
    user_id = final_sid_to_user.get(sid)
    if sid in final_sid_to_user:
        del final_sid_to_user[sid]
    # Optionally clean up game control for this user to free memory
    # if user_id and user_id in final_game_controls:
    #     del final_game_controls[user_id]
    print(f"Final App - Cleaned up resources for user: {user_id}")

# Add error handling for Socket.IO server errors
@sio.event
async def connect_error(sid, data):
    print(f"Final App - Connection error for {sid}: {data}")

# Handle unknown events gracefully
@sio.event
async def default_handler(event, sid, data):
    print(f"Final App - Unknown event '{event}' from {sid}: {data}")
    await sio.emit("error", {"error": f"Unknown event: {event}"}, to=sid)

@sio.event
async def start_game(sid, data):
    try:
        # Validate data structure
        if not data or "playerName" not in data:
            print(f"Final App - Invalid start_game data from {sid}: {data}")
            await sio.emit("error", {"error": "Invalid game start request - missing playerName"}, to=sid)
            return
            
        user_id = data["playerName"]
        
        # Validate user_id
        if not user_id or len(str(user_id).strip()) == 0:
            print(f"Final App - Empty user_id from {sid}")
            await sio.emit("error", {"error": "Invalid player name"}, to=sid)
            return
        
        print(f"Final App - Start game request from {sid} for user {user_id}")
        
        # Clean up any existing mapping for this user
        old_sid = None
        for existing_sid, existing_user in list(final_sid_to_user.items()):
            if existing_user == user_id:
                old_sid = existing_sid
                break
        
        if old_sid and old_sid != sid:
            print(f"Final App - Replacing old connection {old_sid} with new connection {sid} for user {user_id}")
            del final_sid_to_user[old_sid]
            # Disconnect old session
            await sio.disconnect(old_sid)
        
        final_sid_to_user[sid] = user_id
        
        if user_id not in final_game_controls:
            env_instance = create_new_env()
            new_game = FinalGameControl(env_instance)
            final_game_controls[user_id] = new_game
            print(f"Final App - Created new game control for user {user_id}")
        else:
            new_game = final_game_controls[user_id]
            print(f"Final App - Reusing existing game control for user {user_id}")
        
        response = new_game.get_initial_observation()
        response['action'] = None
        await sio.emit("game_update", response, to=sid)
        print(f"Final App - Sent initial observation to {sid}")
        
    except KeyError as ke:
        print(f"Final App - Key error in start_game: {ke}")
        await sio.emit("error", {"error": f"Invalid request format: {str(ke)}"}, to=sid)
    except Exception as e:
        print(f"Final App - Error in start_game: {e}")
        await sio.emit("error", {"error": f"Failed to start game: {str(e)}"}, to=sid)

@sio.event
async def send_action(sid, action):
    try:
        # Validate action input
        if not action:
            print(f"Final App - Empty action from {sid}")
            await sio.emit("error", {"error": "Invalid action - empty"}, to=sid)
            return
        
        user_id = final_sid_to_user.get(sid)
        if not user_id:
            print(f"Final App - No user mapping for sid {sid}")
            await sio.emit("error", {"error": "Session not found - please refresh"}, to=sid)
            return
            
        if user_id not in final_game_controls:
            print(f"Final App - No game control for user {user_id}")
            await sio.emit("error", {"error": "Game not initialized - please start game"}, to=sid)
            return
        user_game = final_game_controls[user_id]
        
        # Validate the action is supported
        valid_actions = ["ArrowLeft", "ArrowRight", "ArrowUp", "Space", "PageUp", "PageDown", "1", "2"]
        if action not in valid_actions:
            # print(f"Final App - Invalid action {action} from user {user_id}")
            await sio.emit("error", {"error": f"Invalid action: {action}"}, to=sid)
            return
            
        response = user_game.handle_action(action)
        response["action"] = action

        # Database saving for individual actions is disabled
        # Only final scores are saved when episode finishes

        if response["done"]:
            if save_to_db:
                try:
                    session = SessionLocal()
                    new_user = Users(user_id=user_id,
                                         timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                         final_score=response["score"],)
                    session.add(new_user)
                    session.commit()
                    # print(f"Final App - Saved final score {response['score']} for user {user_id}")
                except Exception as e:
                    session.rollback()
                    print(f"Final App - Database operation failed: {e}")
                    await sio.emit("error", {"error": "Database operation failed to save final score"}, to=sid)
                finally:
                    session.close()
            # print(f"Final App - Episode finished for user {user_id}, data: {response}")
            await sio.emit("episode_finished", response, to=sid)
        else:
            await sio.emit("game_update", response, to=sid)
            
    except Exception as e:
        print(f"Final App - Error in send_action: {e}")
        await sio.emit("error", {"error": f"Action failed: {str(e)}"}, to=sid)

@sio.event
async def next_episode(sid):
    try:
        user_id = final_sid_to_user.get(sid)
        if not user_id:
            print(f"Final App - No user mapping for sid {sid} in next_episode")
            await sio.emit("error", {"error": "Session not found - please refresh"}, to=sid)
            return
            
        if user_id not in final_game_controls:
            print(f"Final App - No game control for user {user_id} in next_episode")
            await sio.emit("error", {"error": "Game not initialized - please start game"}, to=sid)
            return
            
        user_game = final_game_controls[user_id]
        response = user_game.get_initial_observation()
        await sio.emit("game_update", response, to=sid)
        print(f"Final App - Started next episode for user {user_id}")
        
    except Exception as e:
        print(f"Final App - Error in next_episode: {e}")
        await sio.emit("error", {"error": f"Next episode failed: {str(e)}"}, to=sid)

# Add periodic cleanup task
async def cleanup_stale_connections():
    """Periodic cleanup of stale connections and game controls"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            current_time = datetime.datetime.utcnow()
            
            # Clean up game controls for users with no active connections
            stale_users = []
            for user_id in list(final_game_controls.keys()):
                # Check if user has any active connections
                user_has_connection = any(u == user_id for u in final_sid_to_user.values())
                if not user_has_connection:
                    stale_users.append(user_id)
            
            for user_id in stale_users:
                if user_id in final_game_controls:
                    del final_game_controls[user_id]
                    print(f"Final App - Cleaned up stale game control for user: {user_id}")
                    
            if stale_users:
                print(f"Final App - Cleanup completed. Active connections: {len(final_sid_to_user)}, Active games: {len(final_game_controls)}")
                
        except Exception as e:
            print(f"Final App - Error in cleanup task: {e}")

save_to_db = True  # Set to True to enable database saving (only final scores, not individual actions)

if __name__ == "__main__":
    # Start cleanup task
    async def run_app():
        # Start the cleanup task
        cleanup_task = asyncio.create_task(cleanup_stale_connections())
        
        # Import uvicorn here to avoid import order issues
        import uvicorn
        config = uvicorn.Config(
            socket_app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8002)),  # Different port for final app
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(run_app())




