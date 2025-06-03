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
import ifcopenshell.util.shape

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback

class GymEnv_IFC_AirTerminal(gym.Env):

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
        self._index_to_type = {0: "IfcAirTerminal", 1: "IfcDoor"}
        self._type_to_index = {v: k for k, v in self._index_to_type.items()}
        self._index_to_rotation = {
            0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # 0 grad
            1: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 grad
            2: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # 180 grad
            3: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])   # 270 grad
        }
        self._rotation_to_index = {tuple(map(tuple, v)): k for k, v in self._index_to_rotation.items()}
        self.observation_space = spaces.Box(np.array([0,  0,  0, 0,  0, 0, -20.0, -20.0, -20.0, -20.0, 0, 0, -20.0, -20.0, -20.0, -20.0]),
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
        self.airterminal_type = 0
        self.airterminal_rotation = 0
        self.airterminal_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.door_type = 0
        self.door_rotation = 0
        self.door_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # additional information
        self.airterminal_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.door_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.steps = 0
        self.destination_path = 'H:\\Model\\AirTerminal-Door_AC20-FZK-Haus.ifc'
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
        self.airterminal = ifcopenshell.entity_instance

    def reset(self, seed=None): # reset the env for each episode
        super().reset(seed=seed)
        self.steps = 0
        # copy an ifc model from the original file
        source_path = 'H:\\Model\\AirTerminal-Door_AC20-FZK-Haus.ifc'
        self.destination_path = 'H:\\Model\\AirTerminal-Door_AC20-FZK-Haus_train.ifc'
        shutil.copy(source_path, self.destination_path)
        self.model = ifcopenshell.open(self.destination_path)
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
        # get all elements related to the branch
        if "Supply" in self.airterminal.Name: # the flow direction is different for supply and return system
            system = "Supply"
        else:
            system = "Return"
            
        """If a series of 'wrong' actions prevents the further process of element's geometry, 
	        e.g. the length is less than 0, an exception is thrown and a penalty is applied."""
        try:
            duct = self._find_connected_element(self.airterminal, system)
            fitting = self._find_connected_element(duct, system)
            duct_final = self._find_connected_element(fitting, system)
            direction_fitting = self._find_connected_element(duct_final, system)
            duct_final_shape = ifcopenshell.geom.create_shape(self.settings, duct_final)
            duct_final_verts = duct_final_shape.geometry.verts
        except Exception as e:
            reward = -20
            truncated = True
            terminated = True
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # only shorten or length the branch, no move, therefore two allowed actions for each conflict
        x = duct_final_verts[::3]
        y = duct_final_verts[1::3]
        length_x = max(x) - min(x)
        length_y = max(y) - min(y)
        if length_y > length_x:
            allowed_actions = [0,1]
        else:
            allowed_actions = [2,3]
        #take action
        if self.steps < 100: # limit the length of one episode, so it won't last forever
            if action not in allowed_actions:
                reward = -0.5
                terminated = False
                truncated = False
                observation = self._get_obs()
                info = self._get_info()
            else: #real action
                self.airterminal_matrix = self._index_to_action[action](self.airterminal_matrix)
                # move the duct connecte directly to the air terminal
                duct_shape = ifcopenshell.geom.create_shape(self.settings, duct)
                duct_matrix = ifcopenshell.util.shape.get_shape_matrix(duct_shape)
                duct_matrix = self._index_to_action[action](duct_matrix)
                # move the fitting connected to the former duct
                fitting_shape = ifcopenshell.geom.create_shape(self.settings, fitting)
                fitting_matrix = ifcopenshell.util.shape.get_shape_matrix(fitting_shape)
                fitting_matrix = self._index_to_action[action](fitting_matrix)
                # shorten or lengthen the long duct that at the end of this branch
                duct_final_matrix = self._shorten_or_lengthen_duct(action, duct_final, direction_fitting)
                self.door_matrix = self.door_matrix
                usecase1 = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=self.airterminal, matrix=self.airterminal_matrix)
                usecase2 = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=duct, matrix=duct_matrix)
                usecase3 = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=fitting, matrix=fitting_matrix)
                usecase4 = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=duct_final, matrix=duct_final_matrix)
                usecase1.execute()
                usecase2.execute()
                usecase3.execute()
                usecase4.execute()
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
                truncated = False
                if self.conflicts == self.conflicts_origin:
                    if self.conflicts != 0 and self.severity < self.severity_origin:
                        reward += 5
                        self.severity_origin = self.severity
                    elif self.conflicts != 0 and self.severity > self.severity_origin:
                        reward -= 5
                        self.severity_origin = self.severity
                    if self.created_conf > self.created_conf_origin:
                        reward -= 20
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf < self.created_conf_origin:
                        reward += 0
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                        reward -= 5
                        self.created_severity_origin = self.created_severity
                    elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                        reward += 5
                        self.created_severity_origin = self.created_severity
                elif self.conflicts < self.conflicts_origin:
                    reward += 50
                    if self.conflicts != 0:
                        self.index_conflict += 1
                        self.severity = self._get_information(self.index_conflict)
                    else:
                        terminated = True
                    self.conflicts_origin = self.conflicts
                    if self.created_conf > self.created_conf_origin:
                        reward -= 10
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf < self.created_conf_origin:
                        reward += 0
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                        reward -= 5
                        self.created_severity_origin = self.created_severity
                    elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                        reward += 5
                        self.created_severity_origin = self.created_severity
                else: #self.conflicts > self.conflicts_origin:
                    reward -= 10
                    self.conflicts_origin = self.conflicts
                    if self.created_conf > self.created_conf_origin:
                        reward -= 20
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf < self.created_conf_origin:
                        reward += 0
                        self.created_conf_origin = self.created_conf
                    elif self.created_conf == self.created_conf_origin and self.created_severity > self.created_severity_origin:
                        reward -= 5
                        self.created_severity_origin = self.created_severity
                    elif self.created_conf == self.created_conf_origin and self.created_severity < self.created_severity_origin:
                        reward += 5
                        self.created_severity_origin = self.created_severity
        else:
            reward = 0
            truncated = True
            terminated = True

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        observation = np.array([self.conflicts, self.severity, self.created_conf, self.created_severity, self.airterminal_type, self.airterminal_rotation,
                                self.airterminal_vertices[0], self.airterminal_vertices[1], self.airterminal_vertices[2], self.airterminal_vertices[3],
                                self.door_type, self.door_rotation,
                                self.door_vertices[0], self.door_vertices[1], self.door_vertices[2], self.door_vertices[3]], dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def _get_info(self):
        return {
            "ele1 matrix": self.airterminal_matrix,
            "ele2 matrix": self.door_matrix
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

    def _shorten_or_lengthen_duct(self, action, duct, fitting):
        # shorten or lengthen the duct according to the aciton direction and the direction of the duct, the duct's representation is a IfcExtrudedAreaSolid
        duct_shape = ifcopenshell.geom.create_shape(self.settings, duct)
        duct_matrix = ifcopenshell.util.shape.get_shape_matrix(duct_shape)
        fitting_shape = ifcopenshell.geom.create_shape(self.settings, fitting)
        fitting_matrix = ifcopenshell.util.shape.get_shape_matrix(fitting_shape)
        representation = duct.Representation
        shape_rep = representation.Representations
        profile = shape_rep[0].Items[0].SweptArea
        if action == 0:
            if duct_matrix[1,3] > fitting_matrix[1,3]:
                profile.XDim += 0.2
            else:
                profile.XDim -= 0.2
            duct_matrix[1,3] += 0.1
        elif action == 1:
            if duct_matrix[1,3] > fitting_matrix[1,3]:
                profile.XDim -= 0.2
            else:
                profile.XDim += 0.2
            duct_matrix[1,3] -= 0.1
        elif action == 2:
            if duct_matrix[0,3] > fitting_matrix[0,3]:
                profile.XDim -= 0.2
            else:
                profile.XDim += 0.2
            duct_matrix[0, 3] -= 0.1
        else:
            if duct_matrix[0,3] > fitting_matrix[0,3]:
                profile.XDim += 0.2
            else:
                profile.XDim -= 0.2
            duct_matrix[0, 3] += 0.1
        return duct_matrix
        
    def _update(self):
        """to get the uuid of the model:
        solibri_server_url = 'http://localhost:10876/solibri/v1'
        url = f'{solibri_server_url}/models'
        response = requests.get(url, headers={'accept': 'application/json'})
        response_body = response.json()"""
        modelUUID = "846463d1-03bb-41f4-89a2-006489cbe3c6"
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
            self.airterminal = self.airterminal
            self.created_severity = 0
            for other_severity in self.created_severity_list:
                _severity = self._severity_to_index.get(other_severity)
                self.created_severity += _severity
            # all observed information for the airterminal element
            airterminal_shape = ifcopenshell.geom.create_shape(self.settings, self.airterminal)
            self.airterminal_matrix = ifcopenshell.util.shape.get_shape_matrix(airterminal_shape)
            self.airterminal_type = self._type_to_index.get(self.airterminal.is_a())
            self.airterminal_rotation = self._rotation_to_index.get(tuple(map(tuple, self.airterminal_matrix[:3,:3])))
            self.airterminal_vertices = self._get_vertices(airterminal_shape, self.airterminal_matrix)
        else:
            conflicts = self.conflicted_components_list[index]
            severity = self.conflicts_severity_list[0]
            severity_index = self._severity_to_index.get(severity)
            self.created_severity = 0
            for other_severity in self.created_severity_list:
                _severity = self._severity_to_index.get(other_severity)
                self.created_severity += _severity
            guids = conflicts.split(';')
            for guid in guids:
                ele = self.model.by_guid(guid)
                ele_type = ele.is_a()
                if ele_type == "IfcAirTerminal":
                    self.airterminal = ele
                    print(self.airterminal)
                if ele_type == "IfcDoor":
                    door = ele
                    print(door)
            # all observed information for the airterminal element
            airterminal_shape = ifcopenshell.geom.create_shape(self.settings, self.airterminal)
            self.airterminal_matrix = ifcopenshell.util.shape.get_shape_matrix(airterminal_shape)
            self.airterminal_type = self._type_to_index.get(self.airterminal.is_a())
            self.airterminal_rotation = self._rotation_to_index.get(tuple(map(tuple, self.airterminal_matrix[:3,:3])))
            self.airterminal_vertices = self._get_vertices(airterminal_shape, self.airterminal_matrix)
            # all observed information for the space element
            space_shape = ifcopenshell.geom.create_shape(self.settings, door)
            self.door_matrix = ifcopenshell.util.shape.get_shape_matrix(space_shape)
            self.door_type = self._type_to_index.get(door.is_a())
            self.door_rotation = self._rotation_to_index.get(tuple(map(tuple, self.door_matrix[:3,:3])))
            self.door_vertices = self._get_vertices(space_shape, self.door_matrix)
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
        filtered_checking_results = checking_results[(checking_results['Rule'] == 'Distance Between Doors  and MEP components')]
        filtered_other_conflicts = checking_results[(checking_results['Rule'] != 'Distance Between Doors  and MEP components')]
        conflicts_number= len(filtered_checking_results)
        other_conflicts = len(filtered_other_conflicts)
        conflicts_severity = filtered_checking_results['Severity']
        other_severity = filtered_other_conflicts['Severity']
        self.conflicts_severity_list = conflicts_severity.tolist()
        self.created_severity_list = other_severity.tolist()
        return conflicts_number, other_conflicts, filtered_checking_results

    def _find_connected_element(self, element, system):
        def get_related_inports(element, system): # for the air termial to duct
            for rel in self.model.by_type("IfcRelNests"):
                if rel.RelatingObject == element:
                    if element.is_a == "IfcAirTerminal":
                        element_related_inport = rel.RelatedObjects[0]
                    else:
                        if system == "Supply":
                            for ele in rel.RelatedObjects:
                                if ele.FlowDirection == 'SINK':
                                    element_related_inport = ele
                        if system == "Return":
                            for ele in rel.RelatedObjects:
                                if ele.FlowDirection == 'SOURCE':
                                    element_related_inport = ele
            return element_related_inport
        def get_connected_outports(port): # for the duct to fitting
            for connection in self.model.by_type("IfcRelConnectsPorts"):
                if connection.RelatingPort == port:
                    inport_connected_outport = connection.RelatedPort
                if connection.RelatedPort == port:
                    inport_connected_outport = connection.RelatingPort
            return inport_connected_outport
        def get_element_from_port(port): # for the fitting to another long duct
            for connection in self.model.by_type("IfcRelNests"):
                if port in connection.RelatedObjects:
                    connected_element = connection.RelatingObject
            return connected_element
        inport = get_related_inports(element, system)
        outport = get_connected_outports(inport)
        connected_element = get_element_from_port(outport)
        return connected_element

# for testing the env
env = GymEnv_IFC_AirTerminal()
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
env = GymEnv_IFC_AirTerminal()
# define loggings method
class CustomCallback(BaseCallback): #
    def __init__(self, save_path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_idx = 0
        self.current_episode = []
        self.save_path = save_path
        self.steps = 0 #count the steps, restart Solibri after a certain time steps, but wait to the end of the episode
        self.need_restart = False
    def _on_step(self) -> bool:
        action = self.locals.get("actions", None)
        reward = self.locals.get("rewards", None)
        done = self.locals.get("dones", None)
        observation = self.locals.get("new_obs", None)
        
        #check if there is n NaN
        if np.any(np.isnan(action)) or np.any(np.isnan(reward)) or np.any(np.isnan(observation)):
            print(f"NaN detected. Skipping update. Action: {action}, Reward: {reward}, Obs: {observation}")
            self.current_episode = []
            action = np.nan_to_num(action, nan=0.0)
            reward = np.nan_to_num(reward, nan=0.0)
            observation = np.nan_to_num(observation, nan=0.0)
            return True
        
        self.current_episode.append((observation, action, reward, done))
        
        self.steps += 1
        if self.steps >= 128:
            self.need_restart = True
            
        # Store the data after each episode
        if done[0]:
            custom_dir = os.path.join(self.save_path, "custom/")
            # Create the directory if it doesn't exist
            os.makedirs(custom_dir, exist_ok=True)
            # Now construct the file path
            file_path = os.path.join(custom_dir, "training_data.csv")
            with open(file_path, "a", newline='') as f:
                writer = csv.writer(f)
                # For the first episode, the first row will be wirtten
                if self.episode_idx == 0:
                    writer.writerow(["Episode", "Step", "Observation", "Action", "Reward", "Done"])
                for step_idx, (obs, a, r, d) in enumerate(self.current_episode):
                    writer.writerow([self.episode_idx, step_idx, obs, a, r, d])
            self.episode_idx += 1
            self.current_episode = []  # reset the data for a new episode
            
            if self.need_restart: # restart solibri
                self.restart_solibri()
                self.need_restart = False
                self.steps = 0
        return True
	    
    def restart_solibri(self): #restart Solibri
        print("Attempting to shut down Solibri...")

        try:
            response = requests.post('http://localhost:10876/solibri/v1/shutdown', params={"force": "true"})
            if response.status_code == 200:
                print("Solibri has been successfully shut down.")
            else:
                print(f"Failed to shut down Solibri. Status code: {response.status_code}, response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error shutting down Solibri: {e}")

        time.sleep(30)

        # reopen Solibri
        print("Restarting Solibri...")
        port_list = [10876, 8080, 8081, 8090, 8100, 8200, 8300, 8500, 8600, 8700, 8800, 8888, 8900, 9000, 9090, 10000]

        for port in port_list:
            command = [
                r"C:\Program Files\Solibri\SOLIBRI\Solibri.exe",
                f"--rest-api-server-port={port}",
                "--rest-api-server-local-content",
                "--rest-api-server-http"
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            print("Solibri has been restarted.")

            time.sleep(30)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    print(f"Solibri is successfully running on the port {port}.")
                    break
                else:
                    print(f"Faied to run Solibri on port {port}, try the next one...")
            
        # Open the needed SMC File
        solibri_url = f"http://localhost:{port}/solibri/v1"
        smc_file_path = "H:/Solibri/Model/Column-Window_AC20-FZK-Haus.smc"

        # check if there is already model opened
        check_project_url = f"{solibri_url}/model"
        response = requests.get(check_project_url)
        if response.status_code == 200:
            print("Project opened, you need to close the current project first.")
            close_project_url = f"{solibri_url}/model/close"
            close_response = requests.post(close_project_url)
            
            if close_response.status_code == 200:
                print("Current project closed, ready to open new project...")
            else:
                print(f"Failed to close the project: {close_response.status_code}, response: {close_response.text}")
        else:
            print("No opened project currently, open the new project directly")

        # Open the given SMC model
        open_project_url = f"{solibri_url}/project"
        with open(smc_file_path, "rb") as smc_file:
            headers = {"Content-Type": "application/octet-stream"}
            response = requests.post(open_project_url, params={"name": "A.smc"}, headers=headers, data=smc_file)

        if response.status_code == 201:
            print("Successfully open the project!")
            print("Project Info:", json.loads(response.text))
        else:
            print(f"Failed to open the project: {response.status_code}, response: {response.text}")

        time.sleep(30) 
	    
eval_env = Monitor(env)
timestamp = int(time.time())
save_path = f"./databank/{timestamp}/"
eval_callback = EvalCallback(eval_env, best_model_save_path=save_path + "eval/",
                            log_path=save_path + "eval/",
                            eval_freq=64, n_eval_episodes=4,
                            deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=64, save_path=save_path + "checkpoint/", name_prefix="rl_model")
custom_callback = CustomCallback(save_path)
callback = [eval_callback, checkpoint_callback, custom_callback]

# train with PPO
model = PPO("MlpPolicy", env, ent_coef=0.1, learning_rate=0.0003, n_steps=64, batch_size=64, n_epochs=4, verbose=1, tensorboard_log=save_path + "tensorboard/", device="cpu")
model.learn(total_timesteps=1024, callback=callback, progress_bar=True)
model.save(save_path + "final_model/")
env.close()

# #train with DQN
# model = DQN("MlpPolicy", env, batch_size=128,buffer_size=2048,gamma=0.99,learning_starts=128,learning_rate=0.00063,target_update_interval=64,train_freq=4,gradient_steps=-1,exploration_fraction=0.5, exploration_final_eps=0.1,verbose=1, tensorboard_log=save_path + "dqn_tensorboard/", device="auto")
# model.learn(total_timesteps=1024, callback=callback, progress_bar=True)
# model.save(save_path + "final_model/")
# env.close()
