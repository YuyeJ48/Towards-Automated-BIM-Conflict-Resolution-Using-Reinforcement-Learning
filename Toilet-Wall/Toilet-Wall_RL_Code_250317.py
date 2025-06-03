import gc
import time
import csv
import os
import sys
import shutil
import numpy as np
import pandas as pd
import subprocess
import requests
import socket
import json

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
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch

class GymEnv_IFC_Toilet(gym.Env):

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
        self._index_to_type = {0: "IfcFlowTerminal", 1: "IfcSpace"}
        self._type_to_index = {v: k for k, v in self._index_to_type.items()}
        self._index_to_rotation = {
            0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # 0 grad
            1: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 grad
            2: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # 180 grad
            3: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])   # 270 grad
        }
        self._rotation_to_index = {tuple(map(tuple, v)): k for k, v in self._index_to_rotation.items()}
        # self.observation_space = spaces.Box(np.array([0,  0,  0, 0,  0, 0, -20.0, -20.0, -20.0, -20.0, 0, 0, -20.0, -20.0, -20.0, -20.0]),
        #                                     np.array([10, 3, 10, 10, 1, 3,  20.0,  20.0,  20.0,  20.0, 1, 3,  20.0,  20.0,  20.0,  20.0]), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([0,  0,  0, 0,  0, 0, -1.0, -1.0, -1.0, -1.0]),
                                            np.array([10, 3, 10, 10, 1, 3,  1.0,  1.0,  1.0,  1.0]), dtype=np.float32)
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
        self.toilet_type = 0
        self.toilet_rotation = 0
        self.toilet_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # self.space_type = 0
        # self.space_rotation = 0
        # self.space_vertices = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # additional information
        self.toilet_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.space_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.steps = 0
        self.destination_path = r'C:\Users\ge58quh\Desktop\Thesis-BIM-Conflict-Resolution-cy_training\Model\AC20-FZK-Haus_toilet.ifc'
        self.index_conflict = 0
        self.conflicts_origin = 0
        self.created_conf_origin = 0
        self.severity_origin = 3
        self.conflicted_components_list = []
        self.conflicts_severity_list = []
        self.created_severity_origin = 0
        self.created_severity_list = []
        self.model = ifcopenshell.open(self.destination_path)
        self.settings = ifcopenshell.geom.settings()
        self.toilet = ifcopenshell.entity_instance
        self.no_conflicts_at_start = False
        self.stagnation_counter = 0

        # in this example, the space will not be changed, so we preloaded it
        space = self.model.by_guid("0e_hbkIQ5DMQlIJ$2V3j_m")
        space_shape = ifcopenshell.geom.create_shape(self.settings, space)
        self.space_matrix = ifcopenshell.util.shape.get_shape_matrix(space_shape)
        self.space_type = self._type_to_index.get(space.is_a())
        self.space_rotation = self._rotation_to_index.get(tuple(map(tuple, self.space_matrix[:3,:3])))
        self.space_vertices = self._get_vertices(space_shape, self.space_matrix)


    def reset(self, seed=None): # reset the env for each episode
        super().reset(seed=seed)
        self.steps = 0
        # copy an ifc model from the original file
        source_path = r'C:\Users\ge58quh\Desktop\Thesis-BIM-Conflict-Resolution-cy_training\AC20-FZK-Haus_toilet.ifc'
        self.destination_path = r'C:\Users\ge58quh\Desktop\Thesis-BIM-Conflict-Resolution-cy_training\Model\AC20-FZK-Haus_toilet.ifc'
        shutil.copy(source_path, self.destination_path)
        self.model = ifcopenshell.open(self.destination_path)
        # reset the toilet in different location using IfcOpenShell
        # random_toilet = np.random.randint (0, 8)
        random_toilet = np.random.randint (0, 2)
        toilet, x, y = self._random_placement(random_toilet)
        toilet_shape = ifcopenshell.geom.create_shape(self.settings, toilet)
        toilet_matrix = ifcopenshell.util.shape.get_shape_matrix(toilet_shape)
        if x != 0:
            toilet_matrix[0][3] = x
        else:
            toilet_matrix[1][3] = y
        usecase = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=toilet, matrix=toilet_matrix)
        usecase.execute()
        self.model.write(self.destination_path)
        # update the ifc model in solibri using RestAPI
        self._update()
        # check the model using solibri RestAPI
        self._check()
        # solibri JavaAPI was used for CSV generation everytime a check is done
        # analyse the CSV file
        self.no_conflicts_at_start = False
        self.conflicts, self.created_conf, filtered_checking_results = self._current_results()
        self.conflicts_origin = self.conflicts
        self.created_conf_origin = self.created_conf
        self.stagnation_counter = 0
        
        # If there are no conflicts initially, set default values
        if self.conflicts == 0 and self.created_conf == 0:
            self.severity = 0
            self.severity_origin = 0
            self.conflicted_components_list = []
            self.conflicts_severity_list = []
            self.index_conflict = 0
            # Set a flag to indicate this episode should end immediately in the first step
            self.no_conflicts_at_start = True
        else:
            self.no_conflicts_at_start = False
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
        gc.collect()
        return observation, info

    def step(self, action):
        # count the length of one episode
        self.steps += 1
        
        # If no conflicts at start, end episode immediately with positive reward
        if hasattr(self, 'no_conflicts_at_start') and self.no_conflicts_at_start:
            observation = self._get_obs()
            info = self._get_info()
            return observation, 1.0, True, True, info  # Give positive reward for already solved state
            
        # take action
        self.toilet_matrix = self._index_to_action[action](self.toilet_matrix)
        self.space_matrix = self.space_matrix
        usecase = ifcopenshell.api.geometry.edit_object_placement.Usecase(file=self.model, product=self.toilet, matrix=self.toilet_matrix)
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
        truncated = False

        # if the conflicts and created conflicts are both 0, the problem is solved we close the episode
        if self.conflicts == 0 and self.created_conf == 0:
            reward += 2  # solve the problem
            terminated = True
            truncated = True
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # for not solved conflicts
        if self.steps < 50: # limit the length of one episode, so it won't last forever
            # case 1: main conflicts are not changed
            if self.conflicts == self.conflicts_origin:
                if self.conflicts != 0 and self.severity < self.severity_origin:
                    reward += 0.2
                    self.severity_origin = self.severity
                    self.stagnation_counter = 0
                elif self.conflicts != 0 and self.severity > self.severity_origin:
                    reward -= 0.2
                    self.severity_origin = self.severity
                    self.stagnation_counter = 0
                elif self.conflicts != 0 and self.severity == self.severity_origin:
                    self.stagnation_counter += 1
                    if self.stagnation_counter > 5:
                        reward -= 0.1
 
                if self.created_conf > self.created_conf_origin:
                    reward -= 1
                    self.created_conf_origin = self.created_conf
                    self.stagnation_counter = 0
                elif self.created_conf < self.created_conf_origin:
                    reward += 1
                    self.created_conf_origin = self.created_conf
                    self.stagnation_counter = 0
                elif self.created_conf == self.created_conf_origin and self.created_conf != 0 and self.created_severity > self.created_severity_origin:
                    reward -= 0.2
                    self.created_severity_origin = self.created_severity
                    self.stagnation_counter = 0
                elif self.created_conf == self.created_conf_origin and self.created_conf != 0 and self.created_severity < self.created_severity_origin:
                    reward += 0.2
                    self.created_severity_origin = self.created_severity
                    self.stagnation_counter = 0
                elif self.created_conf == self.created_conf_origin and self.created_conf != 0 and self.created_severity == self.created_severity_origin:
                    self.stagnation_counter += 1
                    if self.stagnation_counter > 5:
                        reward -= 0.1

            # case 2: main conflicts are reduced
            elif self.conflicts < self.conflicts_origin:
                reward += 1
                self.stagnation_counter = 0
                # intermediate reward when main conflicts are solved
                if self.conflicts == 0 and self.created_conf != 0:
                    reward += 0.5
                
                # self.index_conflict += 1
                self.severity = self._get_information(self.index_conflict)
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
            
            # case 3: main conflicts are increased
            else: #self.conflicts > self.conflicts_origin:
                reward -= 1
                self.stagnation_counter = 0
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
        # observation = np.array([self.conflicts, self.severity, self.created_conf, self.created_severity, self.toilet_type, self.toilet_rotation,
        #                         self.toilet_vertices[0], self.toilet_vertices[1], self.toilet_vertices[2], self.toilet_vertices[3],
        #                         self.space_type, self.space_rotation,
        #                         self.space_vertices[0], self.space_vertices[1], self.space_vertices[2], self.space_vertices[3]], dtype=np.float32)
        observation = np.array([self.conflicts, self.severity, self.created_conf, self.created_severity, self.toilet_type, self.toilet_rotation,
                                self.toilet_vertices[0], self.toilet_vertices[1], self.toilet_vertices[2], self.toilet_vertices[3]], dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def _get_info(self):
        return {
            "ele1 matrix": self.toilet_matrix,
            "ele2 matrix": self.space_matrix
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

    def _random_placement(self, random_toilet):
        x = 0
        y = 0
        if random_toilet == 0:
            #up, left
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            x = np.random.uniform(4.3, 4.7)
        elif random_toilet == 1:
            #up, right
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            x = np.random.uniform(6.6, 7.0)
        elif random_toilet == 2:
            #down, left
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            x = np.random.uniform(4.3, 4.7)
        elif random_toilet == 3:
            #down, right
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            x = np.random.uniform(6.6, 7.0)
        elif random_toilet == 4:
            #left, up
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            y = np.random.uniform(9, 9.4)
        elif random_toilet == 5:
            #left, down
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            y = np.random.uniform(6.3, 6.6)
        elif random_toilet == 6:
            #right, up
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            y = np.random.uniform(9, 9.4)
        else:
            #right, down
            toilet = self.model.by_guid("1ArZ91WIzBru99O26vz9Ao")
            toilet_1 = self.model.by_guid("1ArZ91WIzBru99O26vz9CT")
            toilet_2 = self.model.by_guid("1ArZ91WIzBru99O26vz92r")
            toilet_3 = self.model.by_guid("1ArZ91WIzBru99O26vz90s")
            self.model.remove(toilet_1)
            self.model.remove(toilet_2)
            self.model.remove(toilet_3)
            y = np.random.uniform(6.3, 6.6)
        return toilet, x, y

    def _update(self):
        """to get the uuid of the model:
        solibri_server_url = 'http://localhost:10876/solibri/v1'
        url = f'{solibri_server_url}/models'
        response = requests.get(url, headers={'accept': 'application/json'})
        response_body = response.json()"""
        modelUUID = "016c357c-bfcd-4e18-a81d-8db995aab864"
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
            self.toilet = self.toilet
            self.created_severity = 0
            for other_severity in self.created_severity_list:
                _severity = self._severity_to_index.get(other_severity)
                self.created_severity += _severity
            # all observed information for the toilet element
            toilet_shape = ifcopenshell.geom.create_shape(self.settings, self.toilet)
            self.toilet_matrix = ifcopenshell.util.shape.get_shape_matrix(toilet_shape)
            self.toilet_type = self._type_to_index.get(self.toilet.is_a())
            self.toilet_rotation = self._rotation_to_index.get(tuple(map(tuple, self.toilet_matrix[:3,:3])))
            self.toilet_vertices = self._get_vertices_relative(toilet_shape, self.toilet_matrix, ref_space_vertices=self.space_vertices)
            # self.toilet_vertices = self._get_vertices(toilet_shape, self.toilet_matrix)
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
                if ele_type == "IfcFlowTerminal":
                    self.toilet = ele
                if ele_type == "IfcSpace":
                    space = ele
            # all observed information for the toilet element
            toilet_shape = ifcopenshell.geom.create_shape(self.settings, self.toilet)
            self.toilet_matrix = ifcopenshell.util.shape.get_shape_matrix(toilet_shape)
            self.toilet_type = self._type_to_index.get(self.toilet.is_a())
            self.toilet_rotation = self._rotation_to_index.get(tuple(map(tuple, self.toilet_matrix[:3,:3])))
            # self.toilet_vertices = self._get_vertices(toilet_shape, self.toilet_matrix)
            self.toilet_vertices = self._get_vertices_relative(toilet_shape, self.toilet_matrix, ref_space_vertices=self.space_vertices)
            # all observed information for the space element
            # space_shape = ifcopenshell.geom.create_shape(self.settings, space)
            # self.space_matrix = ifcopenshell.util.shape.get_shape_matrix(space_shape)
            # self.space_type = self._type_to_index.get(space.is_a())
            # self.space_rotation = self._rotation_to_index.get(tuple(map(tuple, self.space_matrix[:3,:3])))
            # self.space_vertices = self._get_vertices(space_shape, self.space_matrix)
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
    
    def _get_vertices_relative(self, shape, matrix, ref_space_vertices=None): #get the vertices in relative and normalized coordinates of the element
        verts = shape.geometry.verts
        x = verts[::3]
        y = verts[1::3]
        location = matrix[:,3][0:2]
        abs_min_x = min(x) + location[0]
        abs_min_y = min(y) + location[1]
        abs_max_x = max(x) + location[0]
        abs_max_y = max(y) + location[1]

        if ref_space_vertices is None:
            return np.array([abs_min_x, abs_min_y, abs_max_x, abs_max_y], dtype=np.float32)

        space_min_x, space_min_y, space_max_x, space_max_y = ref_space_vertices
        space_width = max(abs(space_max_x - space_min_x), 1e-6)  # avoid division by zero
        space_height = max(abs(space_max_y - space_min_y), 1e-6)

        norm_min_x = 2 * ((abs_min_x - space_min_x) / space_width) - 1
        norm_min_y = 2 * ((abs_min_y - space_min_y) / space_height) - 1
        norm_max_x = 2 * ((abs_max_x - space_min_x) / space_width) - 1
        norm_max_y = 2 * ((abs_max_y - space_min_y) / space_height) - 1

        return np.clip(
            np.array([norm_min_x, norm_min_y, norm_max_x, norm_max_y], dtype=np.float32),
            -1.0, 1.0
        )


    def _current_results(self): # analyse the CSV file
        # this CSV file_path is determined by the Solibri Java API
        file_path = r'S:/YuyeJiang/Solibri/CheckingResults/checking_results.csv'
        max_retries = 3
        retry_delay = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                checking_results = pd.read_csv(file_path)
                # print(checking_results)
                break
            except pd.errors.EmptyDataError:
                time.sleep(retry_delay)
                retry_count += 1
        else:
            print("EmptyDataError")
        # the name of the main rule, that detects the conflict the agent aim to resolve
        filtered_checking_results = checking_results[(checking_results['Rule'] == '26.15 & 26.17 Shower & Bathrooms')]
        filtered_other_conflicts = checking_results[(checking_results['Rule'] != '26.15 & 26.17 Shower & Bathrooms')]
        # if the rule Space Intersections is in the filtered_checking_results, we should remove it, beacuse the toilet is in the space
        if 'Space Intersections' in filtered_other_conflicts['Rule'].values:
            filtered_other_conflicts = filtered_other_conflicts[(filtered_other_conflicts['Rule'] != 'Space Intersections')]
            other_conflicts = len(filtered_other_conflicts)
            other_severity = filtered_other_conflicts['Severity']
            self.created_severity_list = other_severity.tolist()
        else: # if not, the toliet is not in the space, we should increase the other_conflicts by 1
            other_conflicts = len(filtered_other_conflicts) + 1
            other_severity = filtered_other_conflicts['Severity']
            self.created_severity_list = other_severity.tolist()
            self.created_severity_list.append('CRITICAL')
            
        conflicts_number= len(filtered_checking_results)
        conflicts_severity = filtered_checking_results['Severity']
        self.conflicts_severity_list = conflicts_severity.tolist()

        return conflicts_number, other_conflicts, filtered_checking_results

if __name__ == "__main__":
    # # for testing the env
    # env = GymEnv_IFC_Toilet()
    # check_env(env, warn=True, skip_render_check=True)
    # observation, info = env.reset()
    # for _ in range(10):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print("action:", action)
    #     print("observation:", observation)
    #     print("reward:", reward)
    #     print("terminated:", terminated)
    #     print("truncated:", truncated)
    #     print("info:", info)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # env.close()

    # for new training
    env = GymEnv_IFC_Toilet()
    # define loggings method
    class CustomCallback(BaseCallback):
        def __init__(self, save_path, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.episode_idx = 0
            self.current_episode = []
            self.save_path = save_path
            self.steps = 0 # count the steps, restart Solibri after 512 time steps, but wait to the end of the episode
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
            if self.steps >= 256:
                self.need_restart = True
                
            # Store the data after each episode
            if done[0]:
                custom_dir = os.path.join(self.save_path, "custom/")
                # Create the directory if it doesn't exist
                os.makedirs(custom_dir, exist_ok=True)
                # construct the file path
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
                
                if self.need_restart:
                    self.restart_solibri()
                    self.need_restart = False
                    self.steps = 0

            gc.collect()                   
            return True
        
        def restart_solibri(self):
            # shut off solibri 
            print("Attempting to shut down Solibri...")
            SOLIBRI_API_URL = "http://localhost:10876/solibri/v1/shutdown"
            try:
                response = requests.post(SOLIBRI_API_URL, params={"force": "true"})
                if response.status_code == 200:
                    print("Solibri has been successfully shut down.")
                else:
                    print(f"Failed to shut down Solibri. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error shutting down Solibri: {e}")

            time.sleep(30) 

            #restart solibri
            print("Restarting Solibri...")
            try:
                command = [
                    r"C:\Program Files\Solibri\SOLIBRI\Solibri.exe",
                    "--rest-api-server-port=10876",
                    "--rest-api-server-local-content",
                    "--rest-api-server-http"
                ]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                print("Solibri has been restarted.")
            except Exception as e:
                print(f"Error starting Solibri: {e}")

            # Wait for Solibri to be ready instead of fixed sleep
            solibri_url = "http://localhost:10876/solibri/v1"
            # max_attempts = 30
            # wait_time = 2
            # attempts = 0
            
            # print("Waiting for Solibri to become available...")
            # while attempts < max_attempts:
            #     try:
            #         # Try to connect to Solibri REST API
            #         response = requests.get(f"{solibri_url}")
            #         if response.status_code == 200:
            #             print(f"Solibri is available after {attempts * wait_time} seconds")
            #             break
            #     except requests.exceptions.ConnectionError:
            #         pass
                
            #     attempts += 1
            #     print(f"Waiting for Solibri... Attempt {attempts}/{max_attempts}")
            #     time.sleep(wait_time)
            
            # if attempts >= max_attempts:
            #     print("Timed out waiting for Solibri to become available")
            #     return

            time.sleep(30) 

            # open the smc project file
            smc_file_path = r"C:\Users\ge58quh\Desktop\Thesis-BIM-Conflict-Resolution-cy_training\Toilet-Wall_AC20-FZK-Haus.smc"
            open_project_url = f"{solibri_url}/project"
            with open(smc_file_path, "rb") as smc_file:
                headers = {"Content-Type": "application/octet-stream"}
                response = requests.post(open_project_url, params={"name": "Toilet-Wall_AC20-FZK-Haus.smc"}, headers=headers, data=smc_file)
            if response.status_code == 201:
                print("the smc file is opened")
            else:
                print(f"failed to open the smc file: {response.status_code}, response: {response.text}")
            gc.collect()
        
    eval_env = Monitor(env)
    # eval_env = Monitor(GymEnv_IFC_Toilet())  # standalone evaluation environment
    timestamp = int(time.time())
    save_path = f"./TrainingData/{timestamp}/"
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path + "eval/",
                                log_path=save_path + "eval/",
                                eval_freq=2048, n_eval_episodes=4,
                                deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=2048, save_path=save_path + "checkpoint/", name_prefix="rl_model")
    custom_callback = CustomCallback(save_path)
    callback = [eval_callback, checkpoint_callback, custom_callback]

    #choose one algorithm
    # train with PPO
    model = PPO("MlpPolicy", 
                env, 
                learning_rate=0.0001, 
                n_steps=256, 
                batch_size=256, 
                n_epochs=4, 
                verbose=1, 
                tensorboard_log=save_path + "tensorboard/", 
                device="cpu",
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 64], vf=[128, 64])
                ))
    # for continue the training
    # previous_model_path = "./databank/eval/best_model.zip"
    # model = PPO.load(previous_model_path, env=env)
    model.learn(total_timesteps=32768, callback=callback, progress_bar=True)
    model.save(save_path + "final_model/")
    env.close()

    # #train with DQN
    # model = DQN("MlpPolicy", env, batch_size=128,buffer_size=2048,gamma=0.99,learning_starts=128,learning_rate=0.00063,target_update_interval=64,train_freq=4,gradient_steps=-1,exploration_fraction=0.5, exploration_final_eps=0.1,verbose=1, tensorboard_log=save_path + "dqn_tensorboard/", device="auto")
    # model.learn(total_timesteps=16384, callback=callback, progress_bar=True)
    # # for continue the training
    # # model = DQN.load(previous_model_path, env=env)
    # model.save(save_path + "final_model/")
    # env.close()