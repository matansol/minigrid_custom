
import os
import time
import copy
import random  # needed for random.choice in update_agent

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import socketio

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

from minigrid_custom_env import CustomEnv, ObjObsWrapper
from dpu_clf import *
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath

from typing import Dict, Any


# ---------------------- ENV & DATABASE SETUP ----------------------

load_dotenv()  # Load environment variables from .env

# FastAPI application
app = FastAPI()

# Socket.IO server (async_mode can be "asgi", "threading", etc. Here we use "asgi".)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Wrap the FastAPI app with Socket.IO's ASGI application
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# SQLAlchemy setup
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Action(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20))
    action_type = Column(String(50))
    agent_action = Column(Boolean)
    score = Column(Float)
    reward = Column(Float)
    done = Column(Boolean)
    episode = Column(Integer)
    timestamp = Column(Float)
    agent_index = Column(Integer)
    env_state = Column(String(1000))


class FeedbackAction(Base):
    __tablename__ = "feedback_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20))
    env_state = Column(String(1000))
    agent_action = Column(String(50))
    feedback_action = Column(String(50))
    action_index = Column(Integer)
    timestamp = Column(Float)


def create_database():
    """Drops all tables and recreates the database schema."""
    print("Creating database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


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
        obs, _ = self.env.reset()
        self.saved_env = copy.deepcopy(self.env)
        self.update_agent(None, None)
        print("reset - saved the env")
        self.score = 0
        self.invalid_moves = 0
        return obs

    def actions_to_moves_sequence(self, episode_actions):
        small_arrow = 'turn '  # small arrow is used to indicate the agent turning left or right
        agent_dir = "right"
        move_sequence = []
        for action in episode_actions:
            if action == 0:  # turn left
                agent_dir = turn_agent(agent_dir, "left")
                move_sequence.append((small_arrow + agent_dir, 'turn left'))
            elif action == 1:  # turn right
                agent_dir = turn_agent(agent_dir, "right")
                move_sequence.append((small_arrow + agent_dir, 'turn right'))
            elif action == 2:  # move forward
                move_sequence.append((agent_dir, 'forward'))
            elif action == 3:  # pickup
                move_sequence.append(('pickup ' + agent_dir, 'pickup'))
        return move_sequence

    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if not is_illegal_move(action, self.current_obs, observation, self.agent_last_pos, self.env.get_wrapper_attr('agent_pos')):
            self.episode_actions.append(action)
            self.episode_obs.append(self.env.get_full_obs())  # store the grid image for feedback page
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

    def get_initial_observation(self):
        self.current_obs = self.reset()
        self.episode_obs = [self.env.get_full_obs()]  # for the overview image
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

    def update_agent(self, data, sid):
        if self.ppo_agent is None:
            self.ppo_agent = load_agent(self.env, self.models_paths[0][0])
            self.prev_agent = self.ppo_agent
            print('load the first model, return')
            return None
        if data is None:
            print("Data is None, return")
            return None
        if data.get('updateAgent', False) == False:
            print("No need for update, return")
            return None
        self.user_feedback = data.get('userFeedback')
        if self.user_feedback is None or len(self.user_feedback) == 0:
            print("No user feedback, return")
            return None

        # DB code (only if save_to_db is enabled)
        if save_to_db and sid:
            for action_feedback in self.user_feedback:
                try:
                    session = SessionLocal()
                    _, obs = self.update_env_to_action(action_index=action_feedback['index'])
                    agent_action = data['actions'][action_feedback['index']]['action']
                    feedback_action = FeedbackAction(
                        user_id=self.user_id,
                        env_state="some state",  # Ensure env_state is passed correctly
                        agent_action = actions_dict[agent_action],
                        feedback_action = actions_dict[action_feedback['feedback_action']],
                        action_index = action_feedback['index'],
                        timestamp = time.time(),
                    )
                    session.add(feedback_action)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Database operation failed: {e}")
                    sio.emit("error", {"error": "Database operation failed"}, to=sid)
                finally:
                    session.close()

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

            if len(self.user_feedback) - model_correctness <= 2:  # number of mistakes allowed
                optional_models.append((agent, path[2]))

        print(f'optional_models: {optional_models}')
        if len(optional_models) == 0:
            print("No optional models, return")
            return None
        agent_tuple = random.choice(optional_models)  # choose one agent from the optional models
        self.ppo_agent = agent_tuple[0]
        print(f'load new model: {agent_tuple[1]}')
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent

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
        path_img_buffer, _, _ = plot_all_move_sequence(img, move_sequence, agent_actions, move_color='c', converge_action_location=converge_action_index)
        prev_path_img_buffer, _, _ = plot_all_move_sequence(img, prev_move_sequence, prev_agent_actions, converge_action_location=converge_action_index)

        return {'prev_path_image': prev_path_img_buffer, 'path_image': path_img_buffer}

    def end_of_episode_summary(self):
        imgs = self.episode_obs
        path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(
            imgs,
            self.actions_to_moves_sequence(self.episode_actions),
            self.episode_actions
        )
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
            if not deploy:  # for testing we do not care what the new env is
                return sim_env
            env_objects = env.grid_objects()
            sim_objects = sim_env.grid_objects()
            if state_distance(env_objects, sim_objects) < SIMMILARITY_CONST or j > 10:
                if j > 10:
                    print("No simillar env found")
                break
            j += 1
        return sim_env

# ---------------- Global Variables for Multi-user Support ----------------

# Instead of a single global game_control instance, we maintain a dictionary mapping
# user IDs to their respective GameControl instances.
game_controls: Dict[str, GameControl] = {}

# Also map Socket.IO session IDs to user IDs.
sid_to_user: Dict[str, str] = {}

# Pre-create a "template" for the environment and the model paths.
# (These will be used to create a new instance for each user.)
def create_new_env() -> CustomEnv:
    unique_env_id = 0
    env_instance = CustomEnv(
        grid_szie=8, render_mode="rgb_array", image_full_view=False,
        highlight=True, max_steps=100, lava_cells=3, partial_obs=True,
        unique_env=unique_env_id
    )
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=-3.0)
    env_instance.reset()
    return env_instance

model_paths = [
    ("models/LavaLaver8_20241112/iter_500000_steps.zip", (2, 2, 2, 0, -0.1), "LavaLaver8_20241112"),
    ("models/LavaHate8_20241112/iter_500000_steps.zip", (2, 2, 2, -3, -0.1), "LavaHate8_20241112"),
    ("models/2,2,2,-3,0.2Steps100Grid8_20241230/best_model.zip", (2, 2, 2, -3, -0.2), "LavaHate8_20241229"),
    ("models/0,5,0,-3,0.2Steps100Grid8_20241231/best_model.zip", (0, 5, 0, -3, -0.2), "GreenOnly8_20241231"),
]

actions_dict = {
    0: Actions.left,
    1: Actions.right,
    2: Actions.forward,
    3: Actions.pickup,
    4: Actions.drop,
    5: Actions.toggle,
    6: Actions.done,
    "turn left": Actions.left,
    "turn right": Actions.right,
    "forward": Actions.forward,
    "pickup": Actions.pickup
}

action_dir = {
    "ArrowLeft": "Turn left",
    "ArrowRight": "Turn right",
    "ArrowUp": "Move forward",
    "Space": "Toggle",
    "PageUp": "Pickup",
    "PageDown": "Drop",
    "1": "Pickup",
    "2": "Drop"
}

save_to_db = True

# ------------------ UTILITY FUNCTION -----------------------------
async def finish_turn(response: dict, user_game: GameControl):
    """Common logic after an action is processed."""
    if response["done"]:
        summary = user_game.end_of_episode_summary()
        # Send the summary to the front-end:
        await sio.emit("episode_finished", summary)
    else:
        await sio.emit("game_update", response)

# async def finish_turn(response):
#     """Common logic after an action is processed."""
#     if response["done"]:
#         summary = game_control.end_of_episode_summary()
#         # Send the summary to the front-end:
#         await sio.emit("episode_finished", summary)
#     else:
#         await sio.emit("game_update", response)

# -------------------- FASTAPI ROUTES ----------------------------
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    """
    Return index.html or a basic HTML if you don't have Jinja2 templates.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/update_action")
def update_action(payload: dict):
    index = payload["index"]
    action = payload["action"]
    return {"status": "action updated", "index": index, "action": action}

# -------------------- SOCKET.IO EVENTS ---------------------------
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Remove the sid mapping (but keep the game control instance for future reconnects)
    if sid in sid_to_user:
        del sid_to_user[sid]

@sio.on("start_game")
async def start_game(sid, data):
    """
    When a user starts the game, they send their identifier (playerName).
    Create (or re-use) the GameControl instance corresponding to that user.
    """
    print("starting the game")
    user_id = data["playerName"]
    sid_to_user[sid] = user_id
    if user_id not in game_controls:
        # Create a new environment and GameControl instance for the user.
        env_instance = create_new_env()
        new_game = GameControl(env_instance, model_paths)
        new_game.user_id = user_id
        game_controls[user_id] = new_game
        print(f"Created new game control for user {user_id}")
    else:
        new_game = game_controls[user_id]
        print(f"Reusing existing game control for user {user_id}")
    if data.get("updateAgent", False):
        new_game.update_agent(data, sid)
    response = new_game.get_initial_observation()
    response['action'] = None
    await sio.emit("game_update", response, to=sid)

@sio.on("send_action")
async def handle_send_action(sid, action):
    """
    Handle a user action. Look up the GameControl instance using the sid mapping.
    """
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.handle_action(action)
    response["action"] = action_dir.get(action, action)

    if save_to_db:
        print(f"try to save action to db: {action}")
        session = SessionLocal()
        try:
            new_action = Action(
                action_type=action,
                agent_action=response["agent_action"],
                score=response["score"],
                reward=response["reward"],
                done=response["done"],
                user_id=user_game.user_id,
                timestamp=time.time(),
                episode=response["episode"],
                agent_index=response["agent_index"],
                env_state="some state",
            )
            session.add(new_action)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database operation failed: {e}")
            await sio.emit("error", {"error": "Database operation failed"}, to=sid)
        finally:
            session.close()

    await finish_turn(response, user_game)
    return {"status": "success"}

@sio.on("next_episode")
async def next_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.get_initial_observation()
    await sio.emit("game_update", response, to=sid)

@sio.on("ppo_action")
async def ppo_action(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    action, response = user_game.agent_action()
    response["action"] = action
    await finish_turn(response, user_game)

@sio.on("play_entire_episode")
async def play_entire_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    while True:
        action, response = user_game.agent_action()
        time.sleep(0.3)
        await finish_turn(response, user_game)
        if response["done"]:
            print("Agent Episode finished")
            break

@sio.on("compare_agents")
async def compare_agents(sid, data):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    user_game.update_agent(data, sid)
    res = user_game.agents_different_routs()
    await sio.emit("compare_agents", res, to=sid)

@sio.on("finish_game")
async def finish_game(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    scores = user_game.scores_lst
    print("Scores:", scores)
    await sio.emit("finish_game", {"scores": scores}, to=sid)

# ---------------------- RUNNING THE APP -------------------------
if __name__ == "__main__":
    if save_to_db:
        create_database()

    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
