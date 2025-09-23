import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import numpy
from datetime import datetime
import random  # needed for random.choice in update_agent

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import socketio
import asyncio

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

from minigrid_custom_env import CustomEnv, ObjObsWrapper
from dpu_clf import *
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath
from minigrid.core.constants import IDX_TO_OBJECT , IDX_TO_COLOR 


from functools import reduce
from typing import Dict, Any


# ---------------------- ENV & DATABASE SETUP ----------------------

load_dotenv()  # Load environment variables from .env

# FastAPI application
app = FastAPI()

# Socket.IO server with strict WebSocket-only configuration for Azure Container Apps
sio_config = {
    "async_mode": "asgi",
    "cors_allowed_origins": "*",  # Wildcard for development/testing
    "logger": False,
    "engineio_logger": False,  # Reduce logging overhead
    "ping_timeout": 60,  # Increased timeout for load testing
    "ping_interval": 25,  # Increased interval for stability
    "transports": ['websocket'],  # ONLY WebSocket transport - completely disable polling
    "allow_upgrades": False,  # Never allow transport upgrades from polling
    "http_compression": False,  # Disable compression to avoid issues
    "compression": False,  # Disable compression
    "max_http_buffer_size": 2000000,  # Increased buffer size
    "max_connections": 100,  # Allow more concurrent connections
    "always_connect": True  # Always allow connections
}

sio = socketio.AsyncServer(**sio_config)

# Add middleware to explicitly reject Socket.IO polling requests
@app.middleware("http")
async def reject_polling_middleware(request: Request, call_next):
    """Reject any Socket.IO polling requests to force WebSocket-only connections"""
    if (request.url.path.startswith("/socket.io/") and 
        request.query_params.get("transport") == "polling"):
        print(f"REJECTED POLLING REQUEST: {request.url}")
        from fastapi.responses import Response
        return Response("WebSocket-only mode: Polling transport disabled", status_code=400)
    
    response = await call_next(request)
    return response

# Wrap the FastAPI app with Socket.IO's ASGI application
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# SQLAlchemy setup
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
# Global variable to control database saving

save_to_db = True

class Action(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    action_type = Column(String(20))
    agent_action = Column(Boolean)
    score = Column(Float)
    reward = Column(Float)
    done = Column(Boolean)
    episode = Column(Integer)
    timestamp = Column(String(30))
    agent_index = Column(Integer)
    env_state = Column(String(1000))

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    timestamp = Column(String(30))
    similarity_level = Column(Integer)
    final_score = Column(Float, default=0.0)  # Default to 0.0 if not set


class FeedbackAction(Base):
    __tablename__ = "feedback_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    env_state = Column(String(1000))
    agent_action = Column(String(20))
    feedback_action = Column(String(20))
    feedback_explanation = Column(String(500), nullable=True)
    action_index = Column(Integer)
    timestamp = Column(String(30))  
    episode_index = Column(Integer)
    agent_path = Column(String(100))
    similarity_level = Column(Integer)
    feedback_unique_env = Column(Integer, nullable=True)

class UserChoice(Base): 
    __tablename__ = "user_choices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    old_agent_path = Column(String(100)) 
    new_agent_path = Column(String(100))
    old_agent_score_list = Column(String(30))
    new_agent_score_list = Column(String(30))
    timestamp = Column(String(50))
    demonstration_time = Column(String(50))
    episode_index = Column(Integer)
    choice_to_update = Column(Boolean)
    choice_explanation = Column(String(500), nullable=True)
    similarity_level = Column(Integer)
    feedback_score = Column(Float, nullable=True)
    feedback_count = Column(Integer, nullable=True)
    unique_envs = Column(String(20), nullable=True)
    examples_shown = Column(Integer, nullable=True)

def clear_database():
    """Clears the database tables."""
    print("Clearing database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def create_database():
    """Creates the database tables if they do not already exist."""
    print("Ensuring database tables are created...")
    Base.metadata.create_all(bind=engine)


# Helper
async def in_thread(func, *args, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kw))


class GameControl:
    def __init__(self, env, models_paths, models_distance, user_id, similar_level_env=0, feedback_partial_view=True):
        self.env = env
        # self.saved_env = None
        self.agent_index = 1
        self.models_paths = models_paths
        self.models_distance = models_distance
        self.episode_num = 0
        self.scores_lst = []
        self.last_obs = None
        self.episode_actions = []
        self.episode_cumulative_rewards = []
        self.agent_last_pos = None
        self.episode_images = []
        self.episode_obs = []
        self.episode_agent_locations = []
        self.invalid_moves = 0
        self.user_id = user_id
        self.lava_penalty: float = -3.0
        self.last_score: float = 0.0
        self.similar_level_env: int = int(similar_level_env)
        # self.saved_env_info = {}
        self.ppo_agent = None
        self.prev_agent = None
        self.current_agent_path = ""  
        self.prev_agent_path = ""
        self.prev_agent_index: int = -1
        self.current_obs = {}
        self.feedback_partial_view: bool = feedback_partial_view
        self.feedback_score: int = 0 # the number of good feedbacks the user gave
        self.number_of_feedbacks: int = 0 # total number of feedbacks the user gave
        self.board_seen: list = []
        self.demonstration_unique_envs: list = []  # List to store unique environments for demonstrations
        self.prev_agent_score_list: list = []
        self.current_agent_score_list: list = []
        self.past_choices: set = set()  # To avoid repeating the same choice
        # self.last_unique_env: int = 0 # the last unique env that was given to the user
        
    @timeit
    def reset(self):
        self.update_agent(None, None)
        from_unique_env = True
        if self.similar_level_env == 1:
            from_unique_env = True

        init_kwargs = {"optional_unique_env": self.models_paths[self.agent_index]['optional_unique_env'], 
                       "from_unique_env": from_unique_env, 
                       "board_seen": self.board_seen}
        agent_will_stuck = True
        c = 0
        while agent_will_stuck and c < 10:
            obs, _ = self.env.unwrapped.reset(**init_kwargs)
            agent_will_stuck = will_it_stuck(self.ppo_agent, self.env)
            c += 1
# <----------
        # obs, _ = self.env.unwrapped.reset(**init_kwargs)

        if 'direction' in obs:
            obs = {'image': obs['image']}
        self.episode_obs = [obs]
        self.episode_agent_locations = [(self.env.get_wrapper_attr('agent_pos'), self.env.get_wrapper_attr('agent_dir'))]
        self.infront_objects = []
        self.infront_base_objects = []
        self.infront_feedback_objects = []
        # self.demonstraion_unique_envs = []
        self.feedback_score = 0
        self.saved_env = copy.deepcopy(self.env)
        grid_objects = self.env.grid_objects() #{"balls": [], "wall": (False, None, None), "key" : (False, None), "lava": []}
        self.saved_env_info = {
            'initial_balls': grid_objects['balls'],
            'other_lava_cells': grid_objects['lava'],
            'num_lava_cells': self.env.num_lava_cells,
            'max_steps': self.env.max_steps,
        }

        self.score = 0
        self.invalid_moves = 0
        return obs

    @timeit
    def actions_to_moves_sequence(self, episode_actions):
        small_arrow = 'turn '  # small arrow is used to indicate the agent turning left or right
        agent_dir = "right"
        move_sequence = []
        for action in episode_actions:
            if action == 0:  # turn left
                agent_dir = turn_agent(agent_dir, "left")
                move_sequence.append((agent_dir, 'turn left'))
            elif action == 1:  # turn right
                agent_dir = turn_agent(agent_dir, "right")
                move_sequence.append((agent_dir, 'turn right'))
            elif action == 2:  # move forward
                move_sequence.append((agent_dir, 'forward'))
            elif action == 3:  # pickup
                move_sequence.append((agent_dir, 'pickup'))
            else:
                move_sequence.append(("invalide move", "invalide move"))
        return move_sequence

    # @timeit
    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.score += reward
        self.score = round(self.score, 1)
        if done:
            self.scores_lst.append(self.score)
            self.last_score = self.score
        
        illegal_action = is_illegal_move(action, self.current_obs, observation, self.agent_last_pos, self.env.get_wrapper_attr('agent_pos'))
        if illegal_action:
            self.invalid_moves += 1
            return {'illegal_action': True,}
        else:
            self.episode_actions.append(action)
            if self.episode_cumulative_rewards:
                self.episode_cumulative_rewards.append(round(self.episode_cumulative_rewards[-1] + reward, 1))
            else:
                self.episode_cumulative_rewards.append(round(reward, 1))
            self.episode_obs.append(observation)
            self.episode_agent_locations.append((self.env.get_wrapper_attr('agent_pos'), self.env.get_wrapper_attr('agent_dir')))
        
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
        self.episode_images.append(img if self.feedback_partial_view else self.env.get_full_image())  # store the grid image for feedback page

        self.current_obs = observation
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        return {'image': image_base64, 
                'episode': self.episode_num, 
                'reward': reward, 
                'done': done, 
                'score': self.score, 
                'last_score': self.last_score, 
                'agent_action': agent_action,
                'step_count': self.env.step_count, 
                'agent_index': self.agent_index}

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

    @timeit
    def get_initial_observation(self):
    
        self.current_obs = self.reset()
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        self.episode_actions = []
        self.episode_cumulative_rewards = []
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        ep_img = img if self.feedback_partial_view else self.env.get_full_image()
        self.episode_images = [ep_img]  # for the overview image
        self.episode_num += 1
        print(f"User {self.user_id} Episode {self.episode_num} started ________________________________________________________________________________________")
        return {'image': image_base64, 
                'last_score': self.last_score, 
                'action': None, 
                'reward': 0, 
                'done': False, 
                'score': 0, 
                'episode': self.episode_num, 
                'agent_action': False, 
                'step_count': self.env.step_count,
                'agent_index': self.agent_index,
                # 'direction': self.env.get_wrapper_attr('agent_pos'),
                }
    
    def agent_action(self):
        obs = {'image': self.current_obs['image']}
        action, _ = self.ppo_agent.predict(self.current_obs, deterministic=True)
        action = action.item()
        while True:
            result = self.step(action, True)
            if result.get('illegal_action', False):
                print(f"Illegal action {action}, trying again")
                action = random.choice([0, 1, 2, 3]) # Actions.left, Actions.right, Actions.forward, Actions.pickup
            else:
                break
        result['action'] = action
        return result

    def update_env_to_action(self, action_index):
        tmp_env = copy.deepcopy(self.saved_env)
        obs = tmp_env.get_wrapper_attr('current_state')
        for action in self.episode_actions[:action_index]:
            obs, r, ter, tru, info = tmp_env.step(action)
        return tmp_env, obs

    def revert_to_old_agent(self):
        self.ppo_agent = self.prev_agent
        self.agent_index = self.prev_agent_index
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        print(f'(revert) prev agent index={self.prev_agent_index}, prev agent path={self.prev_agent_path}')
        print(f"(revert) current agent index={self.agent_index}")

    def count_similar_actions(self, env, other_agent, feedback_indexes):
        similar_actions = 0
        for i, action in enumerate(self.episode_actions):
            if i in feedback_indexes:
                continue
            saved_obs = self.episode_obs[i]
            if action == other_agent.predict(saved_obs, deterministic=True)[0].item():
                similar_actions += 1
        return similar_actions

    def is_good_feedback(self, base_front_object, agent_front_object, feedback_front_object):
        """Check if the feedback action was towards a good ball (blue or green) or the agent's action was thowards a bad object (red ball or lava)."""
        if base_front_object and (IDX_TO_OBJECT[base_front_object[0]] == 'lava' or (IDX_TO_OBJECT[base_front_object[0]] == 'ball' and IDX_TO_COLOR[base_front_object[1]] == 'red')):
            return True  # base object is bad, so any feedback is good
        if agent_front_object and (IDX_TO_OBJECT[agent_front_object[0]] == 'lava' or (IDX_TO_OBJECT[agent_front_object[0]] == 'ball' and IDX_TO_COLOR[agent_front_object[1]] == 'red')):
            return True  # agent action was bad, so any feedback is good
        if feedback_front_object and (IDX_TO_OBJECT[feedback_front_object[0]] == 'ball' and IDX_TO_COLOR[feedback_front_object[1]] in ['blue', 'green']):
            return True  # feedback action was good, so it's a good feedback
        return False

    @timeit
    def update_agent(self, data, sid):
        # When loading the first agent
        print(f"(update agent), data={data}")
        if self.ppo_agent is None:
            self.ppo_agent = load_agent(self.env, self.models_paths[self.agent_index]['path'])
            self.current_agent_path = self.models_paths[self.agent_index]['path']  # Save current agent path
            self.prev_agent = self.ppo_agent
            self.prev_agent_path = self.current_agent_path  # Initially, same as current
            self.prev_agent_index = self.agent_index
            print('Loaded the first model, returning')
            print(f'(update agent None) prev agent index={self.prev_agent_index}')
            print(f"(update agent None) current agent index={self.agent_index}")
            return None
        if data is None:
            print("Data is None, return")
            return None
        if data.get('updateAgent', False) == False:
            print("No need for update, return")
            return None
        user_feedback = data.get('userFeedback') # [{ index: index, feedback_action: newAction }, ...]user_feedback_explanation
        if user_feedback is None or len(user_feedback) == 0:
            print("No user feedback, return")
            return None
        
        # Remove duplicate feedbacks by index, keeping only the last feedback for each index
        unique_feedback = {}
        for feedback in user_feedback:
            unique_feedback[feedback['index']] = feedback
        user_feedback = list(unique_feedback.values())
        self.number_of_feedbacks = len(user_feedback) # count all the feedbacks the user gave

        # DB code (only if save_to_db is enabled)
        if save_to_db and sid:
            for action_feedback in user_feedback:
                try:
                    session = SessionLocal()
                    # _, obs = self.update_env_to_action(action_index=action_feedback['index'])
                    action_index = action_feedback['index']
                    if action_index >= len(self.episode_obs):
                        print(f"action_index {action_index} is out of range for episode_obs")
                        obs_str = "No observation available"
                    else:
                        obs = self.episode_obs[action_feedback['index']]
                        img = obs['image']
                        obs_str = json.dumps(img.tolist())  # Convert image to string for storage
                    agent_action = action_feedback['agent_action']
                    feedback_action = FeedbackAction(
                        user_id=self.user_id,
                        env_state=obs_str,  # TODO: Ensure env_state is passed correctly
                        agent_action=actions_dict[agent_action],
                        feedback_action=actions_dict[action_feedback['feedback_action']],
                        feedback_explanation=action_feedback.get('feedback_explanation', ''),
                        action_index=action_feedback['index'],
                        episode_index=self.episode_num,
                        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        agent_path=self.current_agent_path,
                        similarity_level=self.similar_level_env,
                        feedback_unique_env=self.board_seen[-1] if self.board_seen else 0
                    )
                    session.add(feedback_action)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Database operation failed: {e}")
                finally:
                    session.close()

        feedback_indexes = [feedback['index'] for feedback in user_feedback]

        optimal_agents = []
        first_model_checked = True
        target_models_indexes = self.models_distance[self.agent_index]
        for model_i, model_name, _ in target_models_indexes:
            # if (self.models_paths[self.agent_index]['name'], model_name) in self.past_choices:
            #     print(f"Skipping model {model_name} to avoid repeating the same choice")
            #     continue
            agent_data = self.models_paths[model_i]
            path = agent_data['path']
            agent = load_agent(self.env, path)
            # print(f'checking model: {agent_data}')

            agent_correctness = 0
            for action_feedback in user_feedback:
                if action_feedback['index'] >= len(self.episode_obs):
                    return
                saved_obs = self.episode_obs[action_feedback['index']]
                agent_predict_action = agent.predict(saved_obs, deterministic=True)
                agent_predict_action = actions_dict[agent_predict_action[0].item()]
                # agent_pos, agent_dir = self.episode_agent_locations[action_feedback['index']]
                
                base_agent_action = action_feedback['agent_action']
                if agent_predict_action == actions_dict[action_feedback['feedback_action']]:
                    agent_correctness += 1

                ''' TO MAKE A SIMILAR ENV - we take the first object in front of the agent before the action, 
                after the action and after the user feedback action'''
                
                if first_model_checked:
                    # <---------------- old code - keeps to feedback score ------------------>
                    base_face_object = get_infront_object(saved_obs)
                    self.infront_base_objects.append(base_face_object)
                    if  action_feedback['index'] + 1 <= len(self.episode_obs):
                        agent_face_object = get_infront_object(self.episode_obs[action_feedback['index'] + 1])
                        self.infront_objects.append(agent_face_object)
                    tmp_env, _ = self.update_env_to_action(action_feedback['index'])
                    tmp_obs, _, _, _, _= tmp_env.step(actions_dict[action_feedback['feedback_action']])
                    feedback_face_object = get_infront_object(tmp_obs)
                    self.infront_feedback_objects.append(feedback_face_object)
                    # <------------------------------------------->

                    if self.is_good_feedback(base_face_object, agent_face_object, feedback_face_object):
                        self.feedback_score += 1


            if agent_correctness > 0:  # minimum correctness
                # optional_agents.append({"agent":agent, "path": path[2], "correctness": agent_correctness})
                similar_actions = self.count_similar_actions(copy.deepcopy(self.saved_env), agent, feedback_indexes)
                optimal_agents.append({"agent":agent, "name": model_name, "path": path, "correctness_feedback": agent_correctness, "similar_actions": similar_actions, "model_index": model_i})

            first_model_checked = False

        print(f"----------------------------------------------------  counted the feedback - optimal_agents:")
        for agent_dict in optimal_agents:
            print(f"{agent_dict['name']} correctness_feedback= {agent_dict['correctness_feedback']}, similar_actions= {agent_dict['similar_actions']}")

        if len(optimal_agents) == 0:
            optimal_agents.append({"agent":agent, "path": path, "name": model_name, "correctness_feedback": agent_correctness, "model_index": model_i})
        # for agent_dict in optimal_agents:
            # print(f"{agent_dict['name']} correctness_feedback= {agent_dict['correctness_feedback']}")
        # In case of tie on correctness_feedback, take the one with larger "similar_actions"
        def agent_cmp(a, b):
            if a["correctness_feedback"] > b["correctness_feedback"]:
                return a
            elif a["correctness_feedback"] < b["correctness_feedback"]:
                return b
            else:
            # Tie: pick the one with more similar_actions
                return a if a["similar_actions"] >= b["similar_actions"] else b

        new_agent_dict = reduce(agent_cmp, optimal_agents)
        print(f"User_id={self.user_id},  current agent is {self.agent_index}: {self.models_paths[self.agent_index]['name']}")
        print(f"User_id={self.user_id},  new agent picked is:{new_agent_dict['name']}")
        self.prev_agent = self.ppo_agent
        self.prev_agent_path = self.current_agent_path  # Save the old path before updating
        self.prev_agent_index = self.agent_index
        self.ppo_agent = new_agent_dict["agent"]
        self.agent_index = new_agent_dict["model_index"]
        print(f'(update agent) prev agent index={self.prev_agent_index}, prev agent path ={self.prev_agent_path}')
        print(f"(update_agent) current agent index={self.agent_index}")
        self.current_agent_path = self.models_paths[self.agent_index]['path']  # New current agent path
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent
        # Reset examples counter after agent update
        return True

    @timeit
    def agents_different_routs(self, similarity_level=5, stuck_count=0, same_path_count=0):
        if self.ppo_agent == None or self.prev_agent == None:
            print(f"No two agents to compare ppo_agent: {self.ppo_agent}, prev_agent: {self.prev_agent}")
            if self.ppo_agent == None and self.prev_agent == None:
                self.ppo_agent = self.prev_agent = self.models_paths[self.agent_index]['agent']
                print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                print(f"User_id={self.user_id}, both agents are None, set to current agent: {self.ppo_agent}")
            if self.ppo_agent == None:
                self.ppo_agent = self.prev_agent
                self.agent_index = self.prev_agent_index
                self.current_agent_path = self.models_paths[self.agent_index]['path']
            else:
                self.prev_agent_index = 1
                self.prev_agent_path = self.models_paths[self.prev_agent_index]['path']
                self.prev_agent = self.models_paths[self.prev_agent_index]['agent']
        if int(self.similar_level_env) == 0:
            env = self.saved_env
        else: # similarity level > 0
            
            env, unique_env = self.find_similar_env(similarity_level) # TODO: for each 2 agents the special env between them
            if stuck_count < 7 and (will_it_stuck(self.ppo_agent, env) or will_it_stuck(self.prev_agent, env)) :
                print(f"(User_id={self.user_id})  One of the agents will stuck, return")
                return self.agents_different_routs(similarity_level=similarity_level, stuck_count=stuck_count+1)
            
            self.board_seen.append(unique_env)
            self.demonstration_unique_envs.append(unique_env)
                
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_image()
        updated_move_sequence, _, current_score, agent_actions = capture_agent_path(copy_env, self.ppo_agent)
        self.current_agent_score_list.append(current_score)

        # prev_agent_path
        copy_env = copy.deepcopy(env)
        prev_move_sequence, _, prev_score, prev_agent_actions = capture_agent_path(copy_env, self.prev_agent)
        self.prev_agent_score_list.append(prev_score)
        got_to_max = len(prev_move_sequence) == self.env.max_steps or len(updated_move_sequence) == self.env.max_steps
        if (prev_move_sequence == updated_move_sequence) and same_path_count < 3:
            print(f"User_id={self.user_id}, ^^^^^^ agents_different_routs {same_path_count} times, trying again  ^^^^^^^^^^")
            return self.agents_different_routs(similarity_level=similarity_level, same_path_count=same_path_count+1)
        
        # Save the demonstration choice that id given to the user to avoid repeating it
        self.past_choices.add((self.models_paths[self.prev_agent_index]['name'], self.models_paths[self.agent_index]['name']))
        converge_action_index = -1
        image_base64 = image_to_base64(img)
        return {'rawImage': image_base64, 
                'prevMoveSequence': convert_move_sequence_to_jason(prev_move_sequence), 
                'updatedMoveSequence': convert_move_sequence_to_jason(updated_move_sequence), 
                'converge_action_index': converge_action_index
                }


    @timeit
    def end_of_episode_summary(self, need_feedback_data:bool = True):
        if need_feedback_data:
            move_sequence = self.actions_to_moves_sequence(self.episode_actions)
            path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(
                self.episode_images,
                move_sequence,
                self.episode_actions,
            )
            actions_cells = actions_cells_locations(move_sequence)
        else:
            # path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(
            #     self.episode_images,
            #     self.actions_to_moves_sequence(self.episode_actions),
            #     self.episode_actions
            # )
            path_img_base64 = None
            actions_locations = None
            images_buf_list = None
            actions_cells = None

        return {'path_image': path_img_base64,
                'actions': actions_locations, # actions :[{'action', 'action_dir', 'x', 'y', 'width', 'height'},..]
                'cumulative_rewards': self.episode_cumulative_rewards,
                'invalid_moves': self.invalid_moves,
                'score': self.last_score,
                'feedback_images': images_buf_list,
                'actions_cells': actions_cells, 
                'feedback_score': self.feedback_score - (self.number_of_feedbacks - self.feedback_score), # the number of good feedbacks minus the number of bad feedbacks}
        }

    @timeit
    def find_similar_env(self, similarity_level=4, deploy=False) -> tuple[NoDeath, int]:        
        '''Create the initial balls list based on the similarity level and the other initial_balls passed in the kwargs.
        similarity_level (group): 0  - no demonstration
        similarity_level (group): 1 - same board
        similarity_level (group): 2 - special env that shows a big change in the agents demonstration
        similarity_level (group): 3 - random env
        '''
        # env = self.saved_env           
        if similarity_level == 1:
            unique_env = self.env.get_wrapper_attr('last_unique_env')
            print(f"similarity_level=1, unique_env={unique_env}")
        elif similarity_level == 2:
            other_agents_list = self.models_distance[self.prev_agent_index]
            unique_envs_options = next((tup[2] for tup in other_agents_list if tup[0] == self.agent_index), [1,2,3]) # get a list of 3 optional envs
            unique_env = random.choice(unique_envs_options)
        elif similarity_level == 3:
            old_optinal_envs = self.models_paths[self.prev_agent_index]['optional_unique_env']
            new_optinal_envs = self.models_paths[self.agent_index]['optional_unique_env']
            optinal_envs = [x for x in old_optinal_envs if (x in new_optinal_envs and x not in self.board_seen)]
            unique_env = random.choice(optinal_envs)

        initial_kwargs = {
            # 'initial_balls': self.saved_env_info['initial_balls'],
            # 'other_lava_cells': self.saved_env_info['other_lava_cells'],
            # 'infront_objects': [self.infront_base_objects, self.infront_objects, self.infront_feedback_objects],
            "similarity_level": similarity_level,
            "from_unique_env": True,
            "unique_env": unique_env,
            'check_optional_unique_env': False,
            # "optional_unique_env": self.models_paths[self.agent_index]['optional_unique_env'],
            # "old_optional_envs": self.models_paths[self.prev_agent_index]['optional_unique_env'],
            # "board_seen": self.board_seen,
            # "demonstraion_unique_envs": self.demonstraion_unique_envs,
            }

        # random_env = random.randint(1, 11)
        sim_env = CustomEnv(grid_size=8,
                            render_mode="rgb_array",
                            image_full_view=False,
                            highlight=True,
                            max_steps=self.saved_env_info['max_steps'],
                            num_lava_cells=4, #self.saved_env_info['num_lava_cells'],
                            partial_obs=True,
                            similarity_level=self.similar_level_env,
                            )
        sim_env = NoDeath(ObjObsWrapper(sim_env), no_death_types=("lava",), death_cost=self.lava_penalty)
        sim_env.unwrapped.reset(**initial_kwargs) 
        return sim_env, unique_env

    def save_no_user_feedback(self, data, sid):
        user_explanation = data.get('userExplanation')
        # DB code (only if save_to_db is enabled)
        if save_to_db and sid:
            try:
                session = SessionLocal()
                # _, obs = self.update_env_to_action(action_index=action_feedback['index'])
                action_index = 0
                if action_index >= len(self.episode_obs):
                    obs_str = "No observation available"
                else:
                    obs = self.episode_obs[action_index]
                    img = obs['image']
                    obs_str = json.dumps(img.tolist())  # Convert image to string for storage
                agent_action = -1
                print(f"____________________ save to DB NO update agent")
                feedback_action = FeedbackAction(
                    user_id=self.user_id,
                    env_state=obs_str,  # TODO: Ensure env_state is passed correctly
                    agent_action=actions_dict[agent_action],
                    feedback_action=-1,
                    feedback_explanation=user_explanation,
                    action_index=-1,
                    episode_index=self.episode_num,
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    agent_path=self.current_agent_path,
                    similarity_level=self.similar_level_env,
                    feedback_unique_env=self.board_seen[-1] if self.board_seen else 0
                )
                session.add(feedback_action)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Database operation failed: {e}")
            finally:
                session.close()
    
    def save_user_choice(self, choice_to_update, choice_explanation, demonstration_time_fmt):
        session = SessionLocal()
        try:
            unique_env = self.demonstration_unique_envs[-1] if self.demonstration_unique_envs else self.board_seen[-1]
            print(f'(save to DB user choice): old_agent_path={self.prev_agent_path},  new_agent_path={self.current_agent_path}')
            user_choice = UserChoice(
                user_id=self.user_id,
                old_agent_path=str(self.models_paths[self.prev_agent_index]['path']),
                new_agent_path=str(self.current_agent_path),
                old_agent_score_list=','.join(str(x) for x in self.prev_agent_score_list),
                new_agent_score_list=','.join(str(x) for x in self.current_agent_score_list),
                timestamp=datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
                demonstration_time=demonstration_time_fmt,
                episode_index=self.episode_num,
                choice_to_update=choice_to_update,
                choice_explanation=choice_explanation,
                similarity_level=self.similar_level_env,
                feedback_score=self.feedback_score,
                feedback_count=self.number_of_feedbacks,
                unique_envs=unique_env,
                examples_shown=1,
            )
            session.add(user_choice)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"UserChoice saving failed: {e}")
        finally:
            session.close()

# ---------------- Global Variables for Multi-user Support ----------------

# Instead of a single global game_control instance, we maintain a dictionary mapping
# user IDs to their respective GameControl instances.
game_controls: Dict[str, GameControl] = {}

# Also map Socket.IO session IDs to user IDs.
sid_to_user: Dict[str, str] = {}

# Pre-create a "template" for the environment and the model paths.
# (These will be used to create a new instance for each user.)
def create_new_env(lava_penalty) -> CustomEnv:
    env_instance = CustomEnv(grid_szie=8, 
                             render_mode="rgb_array", 
                             image_full_view=False,
                             highlight=True, 
                             max_steps=70, 
                             num_objects=5, 
                             lava_cells=4, 
                             partial_obs=True,
    )
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=lava_penalty)
    # env_instance.unwrapped.reset()
    return env_instance # type: ignore

''' Models that need to be:
AllColors LL - 2,2,2,0,0.1    -   models/3,3,3,0.1,0.1Steps100Grid8_20250602/best_model.zip   /   models/3,3,4,0.2,0.05Steps50Grid8_20250604/best_model.zip
AllColors LH - 2,2,2,-4,0.1    -  models/2,2,4,-4,0.1Steps50Grid8_20250617/best_model.zip   /  models/2,2,4,-3,0.1Steps50Grid8_20250611/best_model.zip  
OnlyBlue LH - -0.5,-0.5,4,-3,0.1  -    
OnlyBlue LL - 0, 0, 4, 0, 0,1  -  models/-1,-1,4,0.2,0.1Steps70Grid8_20250625/best_model.zip
NoRed LL - 0, 3, 3, 0, 0.1  -  models/-0.5,3,4,0.2,0.1Steps50Grid8_20250616/best_model.zip   /   models/-1,3,4,0.2,0.2Steps50Grid8_20250617/best_model.zip
NoRed LH - 0, 3, 3, -3, 0.1  -  models/-1,3,4,-3,0.1Steps60Grid8_20250618/best_model.zip   /   models/-0.5,2,4,-3,0.1Steps50Grid8_20250612_good/best_model.zip   /   
                                models/-0.5,3,4,-3,0.1Steps50Grid8_20250616/best_model.zip 
NoGreen LL - 3, 0, 3, 0, 0.1  -   
NoGreen LH - 3, 0, 3, -3, 0.1  -   
OnlyGreen LL - -0.1, 3, -0.1, 0, 0.01  -   models/-1,4,-1,0.2,0.1Steps60Grid8_20250618/best_model

'''


sub_models_dict = {
      1: {'path': 'models/3,3,3,0.1,0.1Steps100Grid8_20250602/best_model.zip', 'name': 'AllColorsLL1_0526', 'vector': (3, 3, 3, 0.1, 0.1), 'optional_unique_env': list(range(1, 20))},
      2: {'path': 'models/2,2,4,-4,0.1Steps50Grid8_20250617/best_model.zip', 'name': 'AllColorsLH_0617', 'vector': (2, 2, 4, -3, 0.1), 'optional_unique_env':  list(range(1, 20))},
      3: {'path': 'models/-1,-1,4,0.2,0.1Steps70Grid8_20250625/best_model.zip', 'name': 'OnlyBlueLL_0625', 'vector': (-1, -1, 4, 0.2, 0.1), 'optional_unique_env':  list(range(1, 20))},
      5: {'path': 'models/-1,4,-1,0.2,0.1Steps60Grid8_20250618/best_model.zip', 'name': 'OnlyGreenLL_0429', 'vector': (-0.1, 3, -0.1, 0, 0.01), 'optional_unique_env':  [1,2,3] + list(range(5, 20))},
      6: {'path': 'models/-1,-1,4,-3,0.1Steps100Grid8_20250706/best_model.zip', 'name': 'OnlyBlueLH_0706', 'vector': (-1, -1, 4, -3, 0.1), 'optional_unique_env': list(range(1, 20))},
      7: {'path': 'models/-0.5,2,4,-3,0.1Steps50Grid8_20250612_good/best_model.zip', 'name': 'NoRedLH1_0612', 'vector': (-0.5, 2, 4, -3, 0.1), 'optional_unique_env': list(range(1, 20))},
}

sub_models_distance = {
    1: [(2, 'AllColorsLH_0617', [3, 17, 6]), (3, 'OnlyBlueLL_0625', [3, 18, 17]), (5, 'OnlyGreenLL_0429', [17, 10, 6]), (6, 'OnlyBlueLH_0706', [17, 3, 9])], # AllColorsLL1_0526
	2: [(1, 'AllColorsLL1_0526', [3, 17, 6]), (3, 'OnlyBlueLL_0625', [8, 5, 14]), (6, 'OnlyBlueLH_0706', [11, 9, 14]), (7, 'NoRedLH1_0612', [10, 8, 9])], # AllColorsLH_0617
	3: [(1, 'AllColorsLL1_0526', [3, 18, 17]), (6, 'OnlyBlueLH_0706', [8, 16, 5]), (2, 'AllColorsLH_0617', [8, 5, 14]), (5, 'OnlyGreenLL_0429', [17, 10, 6])], # OnlyBlueLL_0625
	5: [(1, 'AllColorsLL1_0526', [17, 10, 6]), (3, 'OnlyBlueLL_0625', [17, 6, 10]), (6, 'OnlyBlueLH_0706', [17, 6, 10]), (2, 'AllColorsLH_0617', [17, 10, 6])], # OnlyGreenLL_0429
	6: [(2, 'AllColorsLH_0617', [11, 9, 14]), (3, 'OnlyBlueLL_0625', [8, 16, 5]), (7, 'NoRedLH1_0612', [1, 18, 11]), (1, 'AllColorsLL1_0526', [17, 3, 9])], # OnlyBlueLH_0706
	7: [(2, 'AllColorsLH_0617', [10, 8, 9]), (6, 'OnlyBlueLH_0706', [1, 18, 11]), (1, 'AllColorsLL1_0526', [3, 18, 17]), (3, 'OnlyBlueLL_0625', [8, 1, 16])], # NoRedLH1_0612
    }


agent_groups = {
    "AllColorsLL1_0526": 1,
    "AllColorsLL2_0604": 1,
    "AllColorsLH_0617": 2,
    "OnlyBlueLL_0625": 1,
    "NoRedLH1_0612": 2,
    "NoRedLL_0616": 2,
    "OnlyGreenLL_0429": 1,
    "NoRedLH2_0618": 3,
    "NoRedLH3_0612": 3,
    "NoRedLL_G_0617": 2,
    "OnlyBlueLH_0706": 2
}

actions_dict = {
    -1: "-1", # No action (for no feedback cases)
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
    "move forward": Actions.forward,
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
    "2": "Drop",
    0: "turn left",
    1: "turn right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


# ------------------ UTILITY FUNCTION -----------------------------
async def finish_turn(response: dict, user_game: GameControl, sid: str, need_feedback_data: bool = True):
    """Common logic after an action is processed."""
    if response["done"]:
        summary = user_game.end_of_episode_summary(need_feedback_data)
        # Send the summary to the front-end:
        await sio.emit("episode_finished", summary, to=sid)
    else:
        await sio.emit("game_update", response, to=sid)

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
    return {"status": "action updated", "index_example": index, "action": action}

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
async def start_game(sid, data, callback=None):
    """
    When a user starts the game, they send their identifier (playerName).
    Create (or re-use) the GameControl instance corresponding to that user.
    """
    print("starting the game")
    user_id = data["playerName"]
    
    sid_to_user[sid] = user_id
    if user_id not in game_controls:
        # Safely convert group to int, default to 1 if invalid
        try:
            group_val = data.get("group", "1")
            # Check if it's a template variable or other invalid value
            if isinstance(group_val, str) and "${" in group_val:
                similarity_level = 1
            else:
                similarity_level = int(group_val)
        except (ValueError, TypeError):
            similarity_level = 1
            print(f"(user_ID={user_id})  Invalid group value: {data.get('group')}, defaulting to 1")

        # Create a new environment and GameControl instance for the user.
        env_instance = create_new_env(lava_penalty=-3)
        # new_game = GameControl(env_instance, new_models_dict, new_models_distance, user_id, similar_level_env=similarity_level, feedback_partial_view=True)
        new_game = GameControl(env_instance, sub_models_dict, sub_models_distance, user_id, similar_level_env=similarity_level, feedback_partial_view=True) #True)
        game_controls[user_id] = new_game
        print(f"Created new game control for user {user_id} with similarity level {similarity_level}")
    else:
        new_game = game_controls[user_id]
        print(f"Reusing existing game control for user {user_id}")
    if data.get("updateAgent", False):
        new_game.update_agent(data, sid)
    if data.get("userNoFeedback", False):
        new_game.save_no_user_feedback(data, sid)
    if data.get("setEnv", False):
        new_game.env.update_from_unique_env(data.get("setEnv"))
    
    response = new_game.get_initial_observation() # TODO: chack if the agent will finish the episode
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
    response["action"] = action_dir[action]

    if save_to_db:
        session = SessionLocal()
        try:
            new_action = Action(
                action_type=action,
                agent_action=response["agent_action"],
                score=response["score"],
                reward=response["reward"],
                done=response["done"],
                user_id=user_game.user_id,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                episode=response["episode"],
                env_state= json.dumps(user_game.current_obs['image'].tolist()) if user_game.current_obs['image'] else "no avaliable obs"  # TODO: Ensure env_state is passed correctly
            )
            session.add(new_action)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database operation failed: {e}")
            await sio.emit("error", {"error": "Database operation failed"}, to=sid)
        finally:
            session.close()

    await finish_turn(response, user_game, sid, need_feedback_data=False)
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
    response = user_game.agent_action()
    # response["action"] = action
    await finish_turn(response, user_game, sid)

@sio.on("play_entire_episode")
async def play_entire_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    while True:
        response = user_game.agent_action()
        # response["action"] = action_dir[action]
        await asyncio.sleep(0.3)
        await finish_turn(response, user_game, sid)
        if response["done"]:
            await asyncio.sleep(0.1)
            break
    user_game.prev_agent_score_list = []
    user_game.current_agent_score_list = []


agent_name_to_path = {v['name']: v['path'] for v in sub_models_dict.values()}
agent_path_to_name = {v['path']: v['name'] for v in sub_models_dict.values()}

@sio.on("compare_agents")
async def compare_agents(sid, data): # data={ playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions, similarity_level: similarity_level }  
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    # another_example = data.get('another_example', False)
    # if another_example:
    #     res = user_game.agents_different_routs(similarity_level=user_game.similar_level_env+user_game.similar_level_env%2) #similarity level[1,2->2, 3,4->4]
    #     await sio.emit("compare_agents", res, to=sid)
    #     return
    res = user_game.update_agent(data, sid)
    if res is None:
        await next_episode(sid)
        return
    if user_game.similar_level_env == 0:
        # just showing a simple text : "the agent has been updated"
        current_path = user_game.current_agent_path
        print(f"Current agent path: {current_path}")
        print(f"Available paths in agent_path_to_name: {list(agent_path_to_name.keys())}")
        print(f"User has updated the agent, similarity level is 0, agent idx = {user_game.agent_index}")
        await sio.emit("update_agent_group", {'agent_group': user_game.agent_index}, to=sid)
        user_game.save_user_choice(1, '', datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"))
        print(f"User {user_id} updated the agent, similarity level is 0, no comparison, going to next episode ---------------------------------------")
        return
    # Increment examples counter for regular agent comparison after feedback
    # user_game.examples_shown_count += 1
    res = user_game.agents_different_routs(user_game.similar_level_env)#data['similarity_level'])
    
    await sio.emit("compare_agents", res, to=sid)

@sio.on("finish_game")
async def finish_game(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    scores = user_game.scores_lst
    await sio.emit("finish_game", {"scores": scores}, to=sid)

@sio.on("start_cover_page")
async def start_cover_page(sid):
    """
    Handle the transition from the cover page to the welcome page.
    """
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return

    # Call the start_game function in the background
    await start_game(sid, {"playerName": user_id})

    # Emit an event to transition to the welcome page
    await sio.emit("go_to_welcome_page", {}, to=sid)

@sio.on("agent_selected")
async def agent_selected(sid, data):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return

    user_game = game_controls[user_id]

    # Convert demonstration_time to desired format if present
    demonstration_time_str = data.get('demonstration_time', None)
    if demonstration_time_str:
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(demonstration_time_str.replace('Z', '+00:00'))
            demonstration_time_fmt = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Failed to parse demonstration_time: {demonstration_time_str}, error: {e}")
            demonstration_time_fmt = demonstration_time_str
    else:
        demonstration_time_fmt = datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    # # Update the agent to the old one
    # if user_game.prev_agent is None:
    #     # print(f"User {user_id} has no previous agent to switch to.")
    #     # await sio.emit("agent_updated", {"status": "error", "message": "No previous agent available"}, to=sid)
    #     # return
    #     pass
    # else:
    # Save the user choice in the DB.
    if save_to_db:
        user_game.save_user_choice(not data['use_old_agent'], data.get('choiceExplanation', ''), demonstration_time_fmt)
        # session = SessionLocal()
        # try:
        #     print(f'(save to DB user choice): old_agent_path={user_game.prev_agent_path},  new_agent_path={user_game.current_agent_path}')
        #     user_choice = UserChoice(
        #         user_id=user_game.user_id,
        #         old_agent_path=str(user_game.models_paths[user_game.prev_agent_index]['path']),
        #         new_agent_path=str(user_game.current_agent_path),
        #         old_agent_score_list=','.join(str(x) for x in user_game.prev_agent_score_list),
        #         new_agent_score_list=','.join(str(x) for x in user_game.current_agent_score_list),
        #         timestamp=datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
        #         demonstration_time=demonstration_time_fmt,
        #         episode_index=user_game.episode_num,
        #         choice_to_update=not data['use_old_agent'],
        #         choice_explanation=data.get('choiceExplanation', ''),
        #         similarity_level=user_game.similar_level_env,
        #         feedback_score=user_game.feedback_score,
        #         feedback_count=user_game.number_of_feedbacks,
        #         unique_envs=",".join(str(x) for x in user_game.demonstraion_unique_envs),
        #         examples_shown=user_game.examples_shown_count,
        #     )
        #     session.add(user_choice)
        #     session.commit()
        # except Exception as e:
        #     session.rollback()
        #     print(f"UserChoice saving failed: {e}")
        # finally:
        #     session.close()
        
        # Update the agent to the old one
    if data['use_old_agent']:
        user_game.revert_to_old_agent()
        print(f"User {user_id} switched to the old agent.")
            
        # agent_groups[agent_path_to_name[user_game.prev_agent_path]]}, to=sid)
    else:
        print(f"User {user_id} keep with the new agent.")
        # await sio.emit("agent_selection_result", {'agent_group': agent_groups[agent_path_to_name[user_game.current_agent_path]]}, to=sid)
    await sio.emit("agent_selection_result", {'agent_group': user_game.agent_index}, to=sid)

# ---------------------- RUNNING THE APP -------------------------
if __name__ == "__main__":
    save_to_db = True
    if save_to_db:
        # clear_database()
        create_database()

    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="warning",
        access_log=False
    )
