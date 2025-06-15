import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
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
# Global variable to control database saving
save_to_db = True

class Action(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(30))
    action_type = Column(String(20))
    agent_action = Column(Boolean)
    score = Column(Float)
    reward = Column(Float)
    done = Column(Boolean)
    episode = Column(Integer)
    timestamp = Column(String(30))
    agent_index = Column(Integer)
    env_state = Column(String(1000))


class FeedbackAction(Base):
    __tablename__ = "feedback_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(30))
    env_state = Column(String(1000))
    agent_action = Column(String(20))
    feedback_action = Column(String(20))
    feedback_explanation = Column(String(500), nullable=True)
    action_index = Column(Integer)
    timestamp = Column(String(30))
    episode_index = Column(Integer)
    agent_path = Column(String(100))

class UserChoice(Base): 
    __tablename__ = "user_choices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(30))
    old_agent_path = Column(String(100)) 
    new_agent_path = Column(String(100))
    timestamp = Column(String(30))
    episode_index = Column(Integer)
    choice_to_update = Column(Boolean)
    choice_explanation = Column(String(500), nullable=True)
    simillarity_level = Column(Integer)

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


SIMMILARITY_CONST = 500
class GameControl:
    def __init__(self, env, models_paths, models_distance, user_id, simillar_level_env=0):
        self.env = env
        # self.saved_env = None
        self.agent_index = next(iter(models_paths.keys()))
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
        self.user_feedback = None
        self.user_id = user_id
        self.current_session = None
        self.lava_penalty = -3.0
        self.last_score = 0
        self.agent_switch_distance = 3
        self.simillar_level_env = simillar_level_env
        self.saved_env_info = {}
        self.ppo_agent = None
        self.prev_agent = None
        self.current_agent_path = ""  
        self.prev_agent_path = ""
        self.current_obs = {}
        
    @timeit
    def reset(self):
        obs, _ = self.env.reset()
        if 'direction' in obs:
            obs = {'image': obs['image']}
        self.episode_obs = [obs]
        self.episode_agent_locations = [(self.env.get_wrapper_attr('agent_pos'), self.env.get_wrapper_attr('agent_dir'))]
        self.infront_objects = []
        self.infront_base_objects = []
        self.infront_feedback_objects = []
        self.saved_env = copy.deepcopy(self.env)
        self.saved_env_info = {
            'initial_balls': self.env.initial_balls,
            'other_lava_cells': self.env.lava_cells,
            'num_lava_cells': self.env.num_lava_cells,
            'max_steps': self.env.max_steps,
        }
        self.update_agent(None, None)
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
                move_sequence.append((small_arrow + agent_dir, 'turn left'))
            elif action == 1:  # turn right
                agent_dir = turn_agent(agent_dir, "right")
                move_sequence.append((small_arrow + agent_dir, 'turn right'))
            elif action == 2:  # move forward
                move_sequence.append((agent_dir, 'forward'))
            elif action == 3:  # pickup
                move_sequence.append(('pickup ' + agent_dir, 'pickup'))
            else:
                move_sequence.append(("invalide move", "invalide move"))
        return move_sequence

    # @timeit
    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        illegal_action = is_illegal_move(action, self.current_obs, observation, self.agent_last_pos, self.env.get_wrapper_attr('agent_pos'))
        if not illegal_action:
            self.episode_actions.append(action)
            if self.episode_cumulative_rewards:
                self.episode_cumulative_rewards.append(self.episode_cumulative_rewards[-1] + reward)
            else:
                self.episode_cumulative_rewards.append(reward)
            self.episode_images.append(self.env.get_full_image())  # store the grid image for feedback page
            self.episode_obs.append(observation)
            self.episode_agent_locations.append((self.env.get_wrapper_attr('agent_pos'), self.env.get_wrapper_attr('agent_dir')))
        
            # pass
            # self.episode_actions.append(action)
            # self.episode_images.append(self.env.get_full_image())  # store the grid image for feedback page
            # self.episode_obs.append(observation)
            # self.episode_agent_locations.append((self.env.get_wrapper_attr('agent_pos'), self.env.get_wrapper_attr('agent_dir')))
        
        

        self.score += reward
        self.score = round(self.score, 2)
        if done:
            self.scores_lst.append(self.score)
            self.last_score = self.score
        
        if illegal_action:
            self.invalid_moves += 1
            return {'illegal_action': True,}
            
        img = self.env.render()
        image_base64 = image_to_base64(img)  # Convert to base64
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
        self.episode_images = [self.env.get_full_image()]  # for the overview image
        self.agent_last_pos = self.env.get_wrapper_attr('agent_pos')
        self.episode_actions = []
        self.episode_cumulative_rewards = []
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        self.episode_num += 1
        print(f"Episode {self.episode_num} started ________________________________________________________________________________________")
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
        action, _ = self.ppo_agent.predict(obs, deterministic=True)
        action = action.item()
        while True:
            result = self.step(action, True)
            if result.get('illegal_action', False):
                print(f"Illegal action {action}, trying again")
                action = random.choice([0, 1, 2, 3]) # Actions.left, Actions.right, Actions.forward, Actions.pickup
            else:
                break

        return action, result

    def update_env_to_action(self, action_index):
        tmp_env = copy.deepcopy(self.saved_env)
        obs = tmp_env.get_wrapper_attr('current_state')
        for action in self.episode_actions[:action_index]:
            obs, r, ter, tru, info = tmp_env.step(action)
        return tmp_env, obs

    @timeit
    def update_agent(self, data, sid):
        # When loading the first agent
        if self.ppo_agent is None:
            self.ppo_agent = load_agent(self.env, self.models_paths[self.agent_index]['path'])
            self.current_agent_path = self.models_paths[self.agent_index]['path']  # Save current agent path
            self.prev_agent = self.ppo_agent
            self.prev_agent_path = self.current_agent_path  # Initially, same as current
            print('Loaded the first model, returning')
            return None
        if data is None:
            print("Data is None, return")
            return None
        if data.get('updateAgent', False) == False:
            print("No need for update, return")
            return None
        self.user_feedback = data.get('userFeedback') # [{ index: index, feedback_action: newAction }, ...]user_feedback_explanation
        if self.user_feedback is None or len(self.user_feedback) == 0:
            print("No user feedback, return")
            return None
        
        # Remove duplicate feedbacks by index, keeping only the last feedback for each index
        unique_feedback = {}
        for feedback in self.user_feedback:
            unique_feedback[feedback['index']] = feedback
        self.user_feedback = list(unique_feedback.values())

        # DB code (only if save_to_db is enabled)
        if save_to_db and sid:
            for action_feedback in self.user_feedback:
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
                    print(f"____________________ save to DB action_feedback: {actions_dict[action_feedback['feedback_action']]}, agent_action: {actions_dict[agent_action]}")
                    feedback_action = FeedbackAction(
                        user_id=self.user_id,
                        env_state=obs_str,  # TODO: Ensure env_state is passed correctly
                        agent_action=actions_dict[agent_action],
                        feedback_action=actions_dict[action_feedback['feedback_action']],
                        feedback_explanation=action_feedback.get('feedback_explanation', ''),
                        action_index=action_feedback['index'],
                        episode_index=self.episode_num,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        agent_path=self.current_agent_path,
                    )
                    session.add(feedback_action)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Database operation failed: {e}")
                    sio.emit("error", {"error": "Database operation failed"}, to=sid)
                finally:
                    session.close()

        optimal_agents = []
        first_model_chacked = True
        target_models_indexes = self.models_distance[self.agent_index][:self.agent_switch_distance]
        for model_i, model_name in target_models_indexes:
            agent_data = self.models_paths[model_i]
            path = agent_data['path']
            agent = load_agent(self.env, path)
            print(f'checking model: {agent_data}, model_name: {model_name}')

            agent_correctness = 0
            for action_feedback in self.user_feedback:
                if action_feedback['index'] >= len(self.episode_obs):
                    return
                saved_obs = self.episode_obs[action_feedback['index']]
                agent_predict_action = agent.predict(saved_obs, deterministic=True)
                agent_predict_action = actions_dict[agent_predict_action[0].item()]
                # agent_pos, agent_dir = self.episode_agent_locations[action_feedback['index']]

                
                base_agent_action = action_feedback['agent_action']
                if agent_predict_action == actions_dict[action_feedback['feedback_action']]:
                    print("agent is correct")
                    agent_correctness += 1


                ''' TO MAKE A SIMILAR ENV - we take the first object in front of the agent before the action, 
                after the action and after the user feedback action'''
                
                if first_model_chacked:
                    base_face_object = get_infront_object(saved_obs)
                    self.infront_base_objects.append(base_face_object)
                    # print("get normal face object:")
                    if  action_feedback['index'] + 1 <= len(self.episode_obs):
                        agent_face_object = get_infront_object(self.episode_obs[action_feedback['index'] + 1])
                        self.infront_objects.append(agent_face_object)
                    # print("get feedback face object:")            
                    tmp_env, _ = self.update_env_to_action(action_feedback['index'])
                    tmp_obs, _, _, _, _= tmp_env.step(actions_dict[action_feedback['feedback_action']])
                    feedback_face_object = get_infront_object(tmp_obs)
                    self.infront_feedback_objects.append(feedback_face_object)
                

                

            if len(self.user_feedback)==0 or agent_correctness > 0:  # minimum correctness
                # optional_agents.append({"agent":agent, "path": path[2], "correctness": agent_correctness})
                optimal_agents.append({"agent":agent, "path": path, "correctness": agent_correctness, "model_index": model_i})
            
            first_model_chacked = False

        if len(optimal_agents) == 0:
            optimal_agents.append({"agent":agent, "path": path, "correctness": agent_correctness, "model_index": model_i})
        new_agent_dict = reduce(lambda a, b: a if a["correctness"] >= b["correctness"] else b, optimal_agents) # get the agent with the most correctness
        self.prev_agent = self.ppo_agent
        self.prev_agent_path = self.current_agent_path  # Save the old path before updating
        self.ppo_agent = new_agent_dict["agent"]
        self.agent_index = new_agent_dict["model_index"]
        self.current_agent_path = self.models_paths[self.agent_index]['path']  # New current agent path
        print(f'load new model: {new_agent_dict["path"]}')#, self.models_paths[new_agent_dict['index']]["name"]}')
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent
        return True

    @timeit
    def agents_different_routs(self, simillarity_level=5, count=0):
        if self.ppo_agent == None or self.prev_agent == None:
            print(f"No two agents to compare ppo_agent: {self.ppo_agent}, prev_agent: {self.prev_agent}")
            if self.ppo_agent == None:
                self.ppo_agent = self.prev_agent
            else:
                self.prev_agent = self.ppo_agent
        # self.saved_env.reset()
        print("(agents_different_routs)  simillarity_level: ", simillarity_level)
        if int(self.simillar_level_env) == 0:
            env = self.saved_env
            print("simillarity level is 0 so we take the base env")
        else: # simillarity level = 1
            env = self.find_simillar_env(simillarity_level)
            print(f"simillarity level is {self.simillar_level_env} so we create a simillar env based on the feedback")
        copy_env = copy.deepcopy(env)
        img = copy_env.get_full_image()
        updated_move_sequence, _, _, agent_actions = capture_agent_path(copy_env, self.ppo_agent)

        # prev_agent_path
        copy_env = copy.deepcopy(env)
        prev_move_sequence, _, _, prev_agent_actions = capture_agent_path(copy_env, self.prev_agent)
        if prev_move_sequence == updated_move_sequence and count < 3:
            count += 1
            return self.agents_different_routs(simillarity_level=simillarity_level, count=count)
        print(f"agents_different_routs {count} times")
        converge_action_index = -1
        for i in range(len(updated_move_sequence)):
            if i >= len(prev_move_sequence):
                break
            if updated_move_sequence[i] != prev_move_sequence[i]:
                converge_action_index = i
                break
        # path_img_buffer, _, _ = plot_all_move_sequence(img, move_sequence, agent_actions, move_color='c', converge_action_location=converge_action_index)
        # prev_path_img_buffer, _, _ = plot_all_move_sequence(img, prev_move_sequence, prev_agent_actions, converge_action_location=converge_action_index)

        # return {'prev_path_image': prev_path_img_buffer, 'path_image': path_img_buffer}
        image_base64 = image_to_base64(img)
        return {'rawImage': image_base64, 
                'prevMoveSequence': convert_move_sequence_to_jason(prev_move_sequence), 
                'updatedMoveSequence': convert_move_sequence_to_jason(updated_move_sequence), 
                'converge_action_index': converge_action_index}


    @timeit
    def end_of_episode_summary(self, need_feedback_data:bool = True):
        if need_feedback_data:
            path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(
                self.episode_images,
                self.actions_to_moves_sequence(self.episode_actions),
                self.episode_actions,
            )
        else:
            # path_img_base64, actions_locations, images_buf_list = plot_move_sequence_by_parts(
            #     self.episode_images,
            #     self.actions_to_moves_sequence(self.episode_actions),
            #     self.episode_actions
            # )
            path_img_base64 = None
            actions_locations = None
            images_buf_list = None

        return {'path_image': path_img_base64,
                'actions': actions_locations, # actions :[{'action', 'x', 'y', 'width', 'height'},..]
                # 'cumulative_rewards': self.episode_cumulative_rewards,
                'invalid_moves': self.invalid_moves,
                'score': self.last_score,
                'feedback_images': images_buf_list}

    @timeit
    def find_simillar_env(self, simillarity_level=5, deploy=False):
        # env = self.saved_env

        initial_kwargs = {
            'initial_balls': self.saved_env_info['initial_balls'],
            'other_lava_cells': self.saved_env_info['other_lava_cells'],
            'infront_objects': [self.infront_base_objects, self.infront_objects, self.infront_feedback_objects],
            "simillarity_level": simillarity_level,
            "from_unique_env": False,
            }

        # random_env = random.randint(1, 11)
        sim_env = CustomEnv(grid_size=8,
                            render_mode="rgb_array",
                            image_full_view=False,
                            highlight=True,
                            max_steps=self.saved_env_info['max_steps'],
                            num_lava_cells=self.saved_env_info['num_lava_cells'],
                            partial_obs=True,
                            simillarity_level=simillarity_level,
                            )
        sim_env = NoDeath(ObjObsWrapper(sim_env), no_death_types=("lava",), death_cost=self.lava_penalty)
        sim_env.unwrapped.reset(**initial_kwargs) 
        
        return sim_env

    def save_no_user_feedback(self, data, sid):
        user_explanation = data.get('userExplanation')
        # DB code (only if save_to_db is enabled)
        if save_to_db and sid:
            try:
                session = SessionLocal()
                # _, obs = self.update_env_to_action(action_index=action_feedback['index'])
                action_index = 0
                if action_index >= len(self.episode_obs):
                    print(f"action_index {action_index} is out of range for episode_obs")
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
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    agent_path=self.current_agent_path,
                )
                session.add(feedback_action)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Database operation failed: {e}")
                sio.emit("error", {"error": "Database operation failed"}, to=sid)
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
                             max_steps=50, 
                             num_objects=5, 
                             lava_cells=4, 
                             partial_obs=True,
    )
    env_instance = NoDeath(ObjObsWrapper(env_instance), no_death_types=("lava",), death_cost=lava_penalty)
    # env_instance.unwrapped.reset()
    return env_instance

''' Models that need to be:
AllColors LL - 2,2,2,0,0.1    -   models/2,2,2,0,0.1Steps100Grid8_20250526/best_model.zip   /   models/3,3,4,0.2,0.05Steps50Grid8_20250604/best_model.zip  /  models/3,3,3,0.1,0.1Steps100Grid8_20250602/best_model.zip
AllColors LH - 2,2,2,-4,0.1    -  models/2,2,4,-3,0.1Steps50Grid8_20250611/best_model.zip   /  models/2,2,2,-4,0.02Steps100Grid8_20250422/best_model.zip  
OnlyBlue LH - -0.5,-0.5,4,-3,0.1  -    models/-0.5,-0.5,4,-3,0.1Steps50Grid8_20250612/best_model.zip
OnlyBlue LL - 0, 0, 4, 0, 0,1
NoRed LL - 0, 3, 3, 0, 0.1  -  models/-0.1,3,3,0,0.1Steps50Grid8_20250604/best_model.zip
NoRed LH - 0, 3,3, -3, 0.1  -  models/-0.5,2,4,-3,0.1Steps50Grid8_20250612_good/best_model.zip   /    models/-0.5,2,4,-2,0.1Steps50Grid8_20250612/best_model.zip
NoGreen LL - 3,0,3,0, 0.1  -   
NoGreen LH - 3, 0, 3, -3, 0.1  -   
OnlyGreen LL - -0.1, 3, -0.1, 0, 0.01  -   models/-0.1,3,-0.1,0,0.01Steps100Grid8_20250429/best_model.zip

'''
new_models_dict = {
    1: {"path": "models/3,3,3,0.1,0.1Steps100Grid8_20250602/best_model.zip", "name": "AllColorsLL_0526", "vector": (3, 3, 3, 0.1, 0.1)},
    2: {"path": "models/3,3,4,0.2,0.05Steps50Grid8_20250604/best_model.zip", "name": "AllColorsLL_0604", "vector": (3, 3, 4, 0.2, 0.05)},
    3: {"path": "models/2,2,4,-3,0.1Steps50Grid8_20250611/best_model.zip", "name": "AllColorsLH_0611", "vector": (2, 2, 4, -3, 0.1)},
    4: {"path": "models/2,2,2,-4,0.02Steps100Grid8_20250422/best_model.zip", "name": "AllColorsLH_0422", "vector": (2, 2, 2, -4, 0.02)},
    5: {"path": "models/-0.5,-0.5,4,-3,0.1Steps50Grid8_20250612/best_model.zip", "name": "OnlyBlueLH_0507", "vector": (-0.5, -0.5, 4, -3, 0.1)},
    6: {"path": "models/-0.1,3,3,0,0.1Steps50Grid8_20250604/best_model.zip", "name": "NoRedLL_0604", "vector": (-0.1, 3, 3, 0, 0.1)},
    7: {"path": "models/-0.1,3,-0.1,0,0.01Steps100Grid8_20250429/best_model.zip", "name": "OnlyGreenLL_0429", "vector": (-0.1, 3, -0.1, 0, 0.01)},
    8: {"path": "models/-0.5,2,4,-3,0.1Steps50Grid8_20250612_good/best_model.zip", "name": "NoRedLH_G_0612", "vector": (-0.5, 2, 4, -3, 0.1)},
    9: {"path": "models/-0.5,2,4,-2,0.1Steps50Grid8_20250612/best_model.zip", "name": "NoRedLH_W_0612", "vector": (-0.5, 2, 4, -2, 0.1)},
}

new_models_distance = {
    1: [(2, 'AllColorsLL_0604'), (6, 'NoRedLL_0604'), (3, 'AllColorsLH_0611'), (7, 'OnlyGreenLL_0429')],
    2: [(1, 'AllColorsLL_0526'), (6, 'NoRedLL_0604'), (3, 'AllColorsLH_0611')],
    3: [(4, 'AllColorsLH_0422'), (9, 'NoRedLH_W_0612'), (2, 'AllColorsLL_0604'), (5, 'OnlyBlueLH_0507')],
    4: [(3, 'AllColorsLH_0611'), (8, 'NoRedLH_G_0612'), (9, 'NoRedLH_W_0612'), (5, 'OnlyBlueLH_0507'), (1, 'AllColorsLL_0526')],
    5: [(8, 'NoRedLH_G_0612'), (9, 'NoRedLH_W_0612'), (3, 'AllColorsLH_0611'), (4, 'AllColorsLH_0422'), (6, 'NoRedLL_0604')],
    6: [(9, 'NoRedLH_W_0612'), (7, 'OnlyGreenLL_0429'), (1, 'AllColorsLL_0526'), (2, 'AllColorsLL_0604'), (8, 'NoRedLH_G_0612')],
    7: [(6, 'NoRedLL_0604'), (1, 'AllColorsLL_0526'), (9, 'NoRedLH_W_0612'), (4, 'AllColorsLH_0422'), (2, 'AllColorsLL_0604')],
    8: [(9, 'NoRedLH_W_0612'), (3, 'AllColorsLH_0611'), (5, 'OnlyBlueLH_0507'), (6, 'NoRedLL_0604'), (4, 'AllColorsLH_0422')],
    9: [(8, 'NoRedLH_G_0612'), (6, 'NoRedLL_0604'), (3, 'AllColorsLH_0611'), (5, 'OnlyBlueLH_0507'), (4, 'AllColorsLH_0422')]
}

# new_models_dict = {
#     1: {"path": "models/2,2,2,0,0.1Steps100Grid8_20250526/best_model.zip", "name": "AllColorsLL_0526", "vector":(2,2,2,0,0.1)},
#     2: {"path": "models/3,-1,3,0.1,0.05Steps50Grid8_20250601/best_model.zip", "name": "NoGreenLL_0601", "vector":(3,-1,3,0.1,0.05)},
#     3: {"path": "models/2,2,2,-4,0.02Steps100Grid8_20250422/best_model.zip", "name": "AllColorsLH_0422", "vector":(2,2,2,-4,0.02)},
#     4: {"path": "models/-0.1,-0.1,4,-4,0.1Steps100Grid8_20250507/best_model.zip", "name": "OnlyBlueLH_0427", "vector":(-0.1,-0.1,4,-4,0.1)},
#     5: {"path": "models/-0.1,3,-0.1,0,0.01Steps100Grid8_20250429/best_model.zip", "name": "OnlyGreenLL_0429", "vector":(-0.1,3,-0.1,0,0.01)},
#     6: {"path": "models/-1,3,3,0.1,0.05Steps50Grid8_20250601/best_model.zip", "name": "NoRedLL_0601", "vector":(-1,3,3,0.1,0.05)},
#     7: {"path": "models/3,3,-1,0,0.1Steps100Grid8_20250422/best_model.zip", "name": "NoBlueLL_0421", "vector":(3,3,-1,-2,0.1)},
#     8: {"path": "models/3,3,-1,0,0.1Steps100Grid8_20250422/best_model.zip", "name": "NoBlueLL_0422", "vector":(3,3,-1,0,0.1)},
    
# }


# new_models_distance = {
#     1: [(8, 'AllColorsLL_0421'), (2, 'OnlyRedLL_0429'), (3, 'AllColorsLH_0422'), (5, 'OnlyGreenLL_0429'), (6, 'OnlyRedLL_0429')],
#     2: [(1, 'AllColorsLL_0421'), (8, 'AllColorsLL_0421'), (5, 'OnlyGreenLL_0429'), (6, 'OnlyRedLL_0429'), (7, 'NoBlueLH_0422')],
#     3: [(4, 'OnlyBlueLH_0427'), (7, 'NoBlueLH_0422'), (2, 'OnlyRedLL_0429'), (1, 'AllColorsLL_0421'), (8, 'AllColorsLL_0421')],
#     4: [(3, 'AllColorsLH_0422'), (2, 'OnlyRedLL_0429'), (1, 'AllColorsLL_0421'), (8, 'AllColorsLL_0421'), (5, 'OnlyGreenLL_0429')],
#     5: [(2, 'OnlyRedLL_0429'), (7, 'NoBlueLH_0422'), (6, 'OnlyRedLL_0429'), (1, 'AllColorsLL_0421'), (8, 'AllColorsLL_0421')],
#     6: [(2, 'OnlyRedLL_0429'), (7, 'NoBlueLH_0422'), (5, 'OnlyGreenLL_0429'), (1, 'AllColorsLL_0421'), (8, 'AllColorsLL_0421')],
#     7: [(5, 'OnlyGreenLL_0429'), (6, 'OnlyRedLL_0429'), (3, 'AllColorsLH_0422'), (2, 'OnlyRedLL_0429'), (1, 'AllColorsLL_0421')],
#     8: [(1, 'AllColorsLL_0421'), (2, 'OnlyRedLL_0429'), (3, 'AllColorsLH_0422'), (5, 'OnlyGreenLL_0429'), (6, 'OnlyRedLL_0429')]
# }


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
async def start_game(sid, data, callback=None):
    """
    When a user starts the game, they send their identifier (playerName).
    Create (or re-use) the GameControl instance corresponding to that user.
    """
    print("starting the game")
    user_id = data["playerName"]
    
    sid_to_user[sid] = user_id
    if user_id not in game_controls:
        simillarity_level = data.get("group", 0)
        # Create a new environment and GameControl instance for the user.
        env_instance = create_new_env(lava_penalty=-3)
        # new_game = GameControl(env_instance, model_paths, simillar_level_env=0)
        # simillarity_level = random.randint(0, 5)
        new_game = GameControl(env_instance, new_models_dict, new_models_distance, user_id, simillar_level_env=simillarity_level)
        game_controls[user_id] = new_game
        print(f"Created new game control for user {user_id} with simillarity level {simillarity_level}")
    else:
        new_game = game_controls[user_id]
        print(f"Reusing existing game control for user {user_id}")
    if data.get("updateAgent", False):
        new_game.update_agent(data, sid)
    if data.get("userNoFeedback", False):
        new_game.save_no_user_feedback(data, sid)
    if data.get("setEnv", False):
        new_game.env.update_from_unique_env(data.get("setEnv"))
    
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
    response["action"] = action_dir[action]

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
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    action, response = user_game.agent_action()
    response["action"] = action
    await finish_turn(response, user_game, sid)

@sio.on("play_entire_episode")
async def play_entire_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    while True:
        action, response = user_game.agent_action()
        response["action"] = action_dir[action]
        await asyncio.sleep(0.3)
        await finish_turn(response, user_game, sid)
        if response["done"]:
            print("Agent Episode finished")
            await asyncio.sleep(0.1)
            break

@sio.on("compare_agents")
async def compare_agents(sid, data): # data={ playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions, simillarity_level: simillarity_level })
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    res = user_game.update_agent(data, sid)
    if res is None:
        await next_episode(sid)
        return
    if user_game.simillar_level_env == 2:
        # just showing a simple text : "the agent has been updated"
        return
    res = user_game.agents_different_routs(user_game.simillar_level_env)#data['simillarity_level'])
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

@sio.on("use_old_agent")
async def use_old_agent(sid, data):
    """
    Update the agent to the old one.
    """
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return

    user_game = game_controls[user_id]

    # Update the agent to the old one
    if user_game.prev_agent is None:
        print(f"User {user_id} has no previous agent to switch to.")
        await sio.emit("agent_updated", {"status": "error", "message": "No previous agent available"}, to=sid)
    else:
        # Save the user choice in the DB.
        if save_to_db:
            session = SessionLocal()
            try:
                user_choice = UserChoice(
                    user_id=user_game.user_id,
                    old_agent_path=str(user_game.prev_agent_path),  # adjust as needed
                    new_agent_path=str(user_game.current_agent_path),     # after switching, still same objects
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    episode_index=user_game.episode_num,
                    choice_to_update=data['use_old_agent'],
                    choice_explanation=data.get('choiceExplanation', ''),
                    simillarity_level=user_game.simillar_level_env,
                )
                session.add(user_choice)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"UserChoice saving failed: {e}")
            finally:
                session.close()

        # Update the agent to the old one
        if data['use_old_agent']:
            user_game.ppo_agent = user_game.prev_agent
            print(f"User {user_id} switched to the old agent.")
            await sio.emit("agent_updated", {"status": "success", "message": "Switched to the old agent"}, to=sid)

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
        port=int(os.environ.get("PORT", 8000))
    )
