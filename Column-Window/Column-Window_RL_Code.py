import time
import csv
import os
import sys
import shutil
import numpy as np
import requests
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.geometry
import ifcopenshell.api.geometry.edit_object_placement
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.util.selector

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import get_linear_fn

class GymEnv_IFC_Column(gym.Env):

    def __init__(self):
        """Define observation space:
            number of conflicts: spaces.Discrete(100)
            severity of conflict: spaces.Discrete(4)
            number of created conflicts: spaces.Discrete(10)            
            severity of created conflicts: spaces.Discrete(4)
            ele1 type: space.Discrete(30)
            ele1 rotation: spaces.Discrete(4)
            ele1 vertices: spaces.Box(np.array([-20.0, -20.0, -20.0, -20.0]),
                                        np.array([20.0, 20.0, 20.0, 20.0]), dtype=np.float32)          
            ele2 type: spaces.Discrete(30)
            ele2 rotation: spaces.Discrete(4)
            ele2 vertices: spaces.Box(np.array([-20.0, -20.0, -20.0, -20.0])
                                        np.array([20.0, 20.0, 20.0, 20.0]), dtype=np.float32)
        Complete action space:
            self._index_to_action = {
                0: self._move_up,
                1: self._move_down,
                2: self._move_left,
                3: self._move_right,
                4: self._move_forward,
                5: self._move_backward,
                6: self._rotate_clockwise,
                7: self._rotate_anticlockwise,
            }"""
        # define observation space
        self._index_to_severity = {0: "None", 1: "LOW", 2: "MODERATE", 3: "CRITICAL"}
        self._severity_to_index = {v: k for k, v in self._index_to_severity.items()}
        self._index_to_type = {0: "IfcColumn", 1: "IfcWindow"}
        self._type_to_index = {v: k for k, v in self._index_to_type.items()}
        self._index_to_rotation = {
            0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # 0 grad
            1: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 grad
            2: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # 180 grad
            3: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])   # 270 grad
        }
        self._rotation_to_index = {tuple(map(tuple, v)): k for k, v in self._index_to_rotation.items()}
        self.observation_space = spaces.Box(np.array([0,  0, 0,  0,  0, 0, -20.0, -20.0, -20.0, -20.0, 0, 0, -20.0, -20.0, -20.0, -20.0]),
                                            np.array([10, 3, 10, 10, 1, 3,  20.0,  20.0,  20.0,  20.0, 1, 3,  20.0,  20.0,  20.0,  20.0]), dtype=np.float32)
        # define action space
        self.action_space = spaces.Discrete(4)
        self._index_to_action = {
            0: self._move_up,
            1: self._move_down,
            2: self._move_left,
            3: self._move_right,
            }
        # initialise variables
        # observation
        self.conflicts = 0
        self.severity = 3
        self.created_conf = 0
        self.created_severity = 0
        self.column_type = 0
        self.column_rotation = 0
        self.column_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.window_type = 0
        self.window_rotation = 0
        self.window_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # additional information
        self.column_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.window_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.steps = 0
        self.destination_path = 'H:\\Model\\Column-Window_AC20-FZK-Haus.ifc'
        self.index_conflict = 0
        self.conflicts_origin = 0
        self.created_conf_origin = 0
        self.severity_origin = 3
        self.conflicted_components_list = []
        self.conflicts_severity_list = []
        self.created_severity_origin = 0
        self.created_severity_list = []
        self.model = ifcopenshell.file()
        self.settings = ifcopenshell.geom.settings()
        self.column = ifcopenshell.entity_instance

    def reset(self, seed=None): # reset the env for each episode
        super().reset(seed=seed)
        self.steps = 0
        # copy an ifc model from the original file
        source_path = 'H:\\Model\\Column-Window_AC20-FZK-Haus.ifc'
        self.destination_path = 'H:\\Model\\Column-Window_AC20-FZK-Haus_train.ifc'
        shutil.copy(source_path, self.destination_path)
        self.model = ifcopenshell.open(self.destination_path)
        # reset the column in different location using IfcOpenShell
        random_column = np.random.randint (0, 9)
        x, y = self._random_placement(random_column)
        column = self.model.by_guid("2kE9PIH$bDtv5uhrQ2Zd6R")
        column_shape = ifcopenshell.geom.create_shape(self.settings, column)
        column_matrix = ifcopenshell.util.shape.get_shape_matrix(column_shape)
        column_matrix[0][3] = x
        column_matrix[1][3] = y
        usecase = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=column, matrix=column_matrix)
        usecase.execute()
        self.model.write()
        # update the ifc model in solibri using RestAPI
        self._update()
        # check the model using solibri RestAPI
        self._check()
        # solibri JavaAPI was used for CSV generation everytime a check is done
        # analyse the CSV file
        self.conflicts, self.created_conf, filtered_checking_results = self._current_results()
        self.conflicts_origin = self.conflicts
        self.created_conf_origin = self.created_conf
        conflicted_components = filtered_checking_results['Component']
        self.conflicted_components_list = conflicted_components.tolist()
        conflicts_severity = filtered_checking_results['Severity']
        self.conflicts_severity_list = conflicts_severity.tolist()
        # get the components information of the first conflict
        self.index_conflict = 0
        self.severity = self._get_information(self.index_conflict)
        self.severity_origin = self.severity

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # count the length of one episode
        self.steps += 1
        # take action
        self.column_matrix = self._index_to_action[action](self.column_matrix)
        self.window_matrix = self.window_matrix
        usecase = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=self.column, matrix=self.column_matrix)
        usecase.execute()
        self.model.write(self.destination_path)
        # update and check the new ifc file
        self._update()
        self._check()
        self.conflicts, self.created_conf, _ = self._current_results()
        self.severity = self._get_information(self.index_conflict)
        """define the rewards:
        resolve a targeted conflict +1
        created another conflict: -1
        lower the severity: +0.2"""
        reward = 0
        terminated = False
        truncated =False
        if self.steps < 30: # limit the length of one episode, so it won't last forever
            if self.conflicts == self.conflicts_origin:
                if self.conflicts != 0 and self.severity < self.severity_origin:
                    reward += 0.2
                    self.severity_origin = self.severity
                elif self.conflicts != 0 and self.severity > self.severity_origin:
                    reward -= 0.2
                    self.severity_origin = self.severity
                if self.created_conf > self.created_conf_origin:
                    reward -= 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf < self.created_conf_origin:
                    reward += 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                    reward -= 0.2
                    self.created_severity_origin = self.created_severity
                elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                    reward += 0.2
                    self.created_severity_origin = self.created_severity
            elif self.conflicts < self.conflicts_origin:
                reward += 1
                if self.conflicts != 0:
                    self.index_conflict += 1
                    self.severity = self._get_information(self.index_conflict)
                else:
                    terminated = True
                self.conflicts_origin = self.conflicts
                if self.created_conf > self.created_conf_origin:
                    reward -= 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf < self.created_conf_origin:
                    reward += 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                    reward -= 0.2
                    self.created_severity_origin = self.created_severity
                elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                    reward += 0.2
                    self.created_severity_origin = self.created_severity
            else: #self.conflicts > self.conflicts_origin:
                reward -= 1
                self.conflicts_origin = self.conflicts
                if self.created_conf > self.created_conf_origin:
                    reward -= 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf < self.created_conf_origin:
                    reward += 1
                    self.created_conf_origin = self.created_conf
                elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                    reward -= 0.2
                    self.created_severity_origin = self.created_severity
                elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                    reward += 0.2
                    self.created_severity_origin = self.created_severity                    
        else:
            reward = -2
            truncated = True
            terminated = True

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        observation = np.array([self.conflicts, self.severity, self.created_conf, self.created_severity, self.column_type, self.column_rotation,
                                self.column_vertices[0], self.column_vertices[1], self.column_vertices[2], self.column_vertices[3],
                                self.window_type, self.window_rotation,
                                self.window_vertices[0], self.window_vertices[1], self.window_vertices[2], self.window_vertices[3]], dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def _get_info(self):
        return {
            "ele1 matrix": self.column_matrix,
            "ele2 matrix": self.window_matrix,
            }

    def _move_up(self, matrix): # positive y
        matrix[1, 3] += 0.1
        return matrix
    def _move_down(self, matrix): # negative y
        matrix[1, 3] -= 0.1
        return matrix
    def _move_left(self, matrix): # positive x
        matrix[0, 3] -= 0.1
        return matrix
    def _move_right(self, matrix): # negative y
        matrix[0, 3] += 0.1
        return matrix

    def _random_placement(self, random_window):
        if random_window == 0:
            x = np.random.uniform(0.6, 0.7)
            y = np.random.uniform(7.0, 8.0)
        elif random_window == 1:
            x = np.random.uniform(0.6, 0.7)
            y = np.random.uniform(1.7, 2.7)
        elif random_window == 2:
            x = np.random.uniform(2.3, 3.3)
            y = np.random.uniform(0.6, 0.7)
        elif random_window == 3:
            x = np.random.uniform(8.7, 9.7)
            y = np.random.uniform(0.6, 0.7)
        elif random_window == 4:
            x = np.random.uniform(11.3, 11.4)
            y = np.random.uniform(1.7, 2.7)
        elif random_window == 5:
            x = np.random.uniform(11.3, 11.4)
            y = np.random.uniform(7.0, 8.0)
        elif random_window == 6:
            x = np.random.uniform(9.3, 10.2)
            y = np.random.uniform(9.3, 9.4)
        elif random_window == 7:
            x = np.random.uniform(5.2, 6.1)
            y = np.random.uniform(9.3, 9.4)
        else:
            x = np.random.uniform(1.6, 2.5)
            y = np.random.uniform(9.3, 9.4)
        return np.around(x, 3), np.around(y, 3)

    def _update(self):
        """to get the uuid of the model:
        solibri_server_url = 'http://localhost:10876/solibri/v1'
        url = f'{solibri_server_url}/models'
        response = requests.get(url, headers={'accept': 'application/json'})
        response_body = response.json()"""
        modelUUID = "1ba10e33-68f8-4a56-acee-ea6681a59f07"
        solibri_server_url = 'http://localhost:10876/solibri/v1'
        url = f'{solibri_server_url}/models/{modelUUID}/update'
        headers = {'Content-Type': 'application/octet-stream'}
        with open(self.destination_path, 'rb') as ifc_file:
            ifc_data = ifc_file.read()
        response = requests.put(url, headers=headers, data=ifc_data)
        if response.status_code != 201:
            print('Error:', response.text)
            sys.exit(f'Failed to update IFC model with status code: {response.status_code}')

    def _check(self):
        solibri_server_url = 'http://localhost:10876/solibri/v1'
        url = f'{solibri_server_url}/checking?checkSelected=false'
        response = requests.post(url, headers={'accept': 'application/json'}, data='')
        if response.status_code != 200:
            print("Error:", response.text)
            sys.exit(f"Request failed with status code: {response.status_code}")

    def _get_information(self, index):
        if self.conflicts == 0:
            severity_index = 0
            self.column = self.column
            self.created_severity = 0
            for other_severity in self.created_severity_list:
                _severity = self._severity_to_index.get(other_severity)
                self.created_severity += _severity
            # all observed information for the column element
            column_shape = ifcopenshell.geom.create_shape(self.settings, self.column)
            self.column_matrix = ifcopenshell.util.shape.get_shape_matrix(column_shape)
            self.column_type = self._type_to_index.get(self.column.is_a())
            self.column_rotation = self._rotation_to_index.get(tuple(map(tuple, self.column_matrix[:3,:3])))
            self.column_vertices = self._get_vertices(column_shape, self.column_matrix)
        else:
            conflicts = self.conflicted_components_list[index]
            severity = self.conflicts_severity_list[index]
            severity_index = self._severity_to_index.get(severity)
            self.created_severity = 0
            for other_severity in self.created_severity_list:
                _severity = self._severity_to_index.get(other_severity)
                self.created_severity += _severity
            guids = conflicts.split(';')
            for guid in guids:
                ele = self.model.by_guid(guid)
                ele_type = ele.is_a()
                if ele_type == "IfcColumn":
                    self.column = ele
                elif ele_type == "IfcWindow":
                    window = ele
            # all observed information for the column element
            column_shape = ifcopenshell.geom.create_shape(self.settings, self.column)
            self.column_matrix = ifcopenshell.util.shape.get_shape_matrix(column_shape)
            self.column_type = self._type_to_index.get(self.column.is_a())
            self.column_rotation = self._rotation_to_index.get(tuple(map(tuple, self.column_matrix[:3,:3])))
            self.column_vertices = self._get_vertices(column_shape, self.column_matrix)
            # all observed information for the window element
            window_shape = ifcopenshell.geom.create_shape(self.settings, window)
            self.window_matrix = ifcopenshell.util.shape.get_shape_matrix(window_shape)
            self.window_type = self._type_to_index.get(window.is_a())
            self.window_rotation = self._rotation_to_index.get(tuple(map(tuple, self.window_matrix[:3,:3])))
            self.window_vertices = self._get_vertices(window_shape, self.window_matrix)
            return severity_index

    def _get_vertices(self, shape, matrix): #get the vertices in world coordinates of the element
        verts = shape.geometry.verts
        x = verts[::3]
        y = verts[1::3]
        location = matrix[:,3][0:2]
        min_x = min(x) + location[0]
        min_y = min(y) + location[1]
        max_x = max(x) + location[0]
        max_y = max(y) + location[1]
        vertices = np.array([min_x, min_y, max_x, max_y])
        return vertices

    def _current_results(self): # analyse the CSV file
        # this CSV file_path is determined by the Solibri Java API
        file_path = 'H:/Checking_results/checking_results.csv'
        max_retries = 3
        retry_delay = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                checking_results = pd.read_csv(file_path)
                break
            except pd.errors.EmptyDataError:
                time.sleep(retry_delay)
                retry_count += 1
        else:
            print("EmptyDataError")
        # the name of the main rule, that detects the conflict the agent aim to resolve
        filtered_checking_results = checking_results[(checking_results['Rule'] == 'Clearance in Front of Windows')]
        filtered_other_conflicts = checking_results[(checking_results['Rule'] != 'Clearance in Front of Windows')]
        conflicts_number= len(filtered_checking_results)
        other_conflicts = len(filtered_other_conflicts)
        conflicts_severity = filtered_checking_results['Severity']
        other_severity = filtered_other_conflicts['Severity']
        self.conflicts_severity_list = conflicts_severity.tolist()
        self.created_severity_list = other_severity.tolist()
        return conflicts_number, other_conflicts, filtered_checking_results

# for testing the env
env = GymEnv_IFC_Column()
check_env(env, warn=True, skip_render_check=True)
observation, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("action:", action)
    print("observation:", observation)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("info:", info)
    if terminated or truncated:
        observation, info = env.reset()
env.close()

# for new training
env = GymEnv_IFC_Column()
# define loggings method
class CustomCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_idx = 0
        self.current_episode = []
        self.save_path = save_path
    def _on_step(self) -> bool:
        # log the information i need
        action = self.locals.get("actions", None)
        reward = self.locals.get("rewards", None)
        done = self.locals.get("dones", None)
        observation = self.locals.get("new_obs", None)
        # check if there is a NaN
        if np.any(np.isnan(action)) or np.any(np.isnan(reward)) or np.any(np.isnan(observation)):
            print(f"NaN detected. Skipping update. Action: {action}, Reward: {reward}, Obs: {observation}")
            self.current_episode = []
            action = np.nan_to_num(action, nan=0.0)
            reward = np.nan_to_num(reward, nan=0.0)
            observation = np.nan_to_num(observation, nan=0.0)
            return True
        self.current_episode.append((observation, action, reward, done))
        # store the csv data when one episode is finished
        if done[0]:
            custom_dir = os.path.join(self.save_path, "custom/")
            # Create the directory if it doesn't exist
            os.makedirs(custom_dir, exist_ok=True)
            # construct the file path
            file_path = os.path.join(custom_dir, "training_data.csv")
            with open(file_path, "a", newline='') as f:
                writer = csv.writer(f)
                # write the first row for the first episode
                if self.episode_idx == 0:
                    writer.writerow(["Episode", "Step", "Observation", "Action", "Reward", "Done"])
                for step_idx, (obs, a, r, d) in enumerate(self.current_episode):
                    writer.writerow([self.episode_idx, step_idx, obs, a, r, d])
            self.episode_idx += 1
            self.current_episode = []  # reset for the next episode
        return True
eval_env = Monitor(env)
timestamp = int(time.time())
save_path = f"./databank/{timestamp}/"
eval_callback = EvalCallback(eval_env, best_model_save_path=save_path+ "eval/",
                            log_path=save_path + "eval/",
                            eval_freq=64, n_eval_episodes=2,
                            deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=64, save_path=save_path + "checkpoint/", name_prefix="rl_model")
custom_callback = CustomCallback(save_path)
callback = [eval_callback, checkpoint_callback, custom_callback]
# choose one algorithm
model = DQN("MlpPolicy", env, batch_size=128,buffer_size=2048,gamma=0.99,learning_starts=128,learning_rate=0.00063,target_update_interval=64,train_freq=4,gradient_steps=-1,exploration_fraction=0.5, exploration_final_eps=0.1,verbose=1, tensorboard_log=save_path + "dqn_tensorboard/", device="auto")
model = PPO("MlpPolicy", env, learning_rate=0.0005, n_steps=64, batch_size=64, n_epochs=4, verbose=1, tensorboard_log=save_path + "tensorboard/", device="cpu")
# for continue the training, choose one model
# previous_model_path = "./databank/final_model.zip"
# model = DQN.load(previous_model_path, env=env)
# model = PPO.load(previous_model_path, env=env)
model.learn(total_timesteps=1024, callback=callback, progress_bar=True)
model.save(save_path + "final_model/")
env.close()

# for testing the model
env = GymEnv_IFC_AirTerminal()
observation, info = env.reset()
test_model_path = "./databank/final_model.zip"
model = PPO.load(test_model_path, env=env)
terminated = False
while not terminated:
    action, _states = model.predict(observation)
    action = int(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print("action:", action)
    print("observation:", observation)
    print("reward:", reward)
else:
    env.close()
