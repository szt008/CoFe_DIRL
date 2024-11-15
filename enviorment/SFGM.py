import numpy as np
import math
from typing import Optional, Sequence
from smarts.core.colors import Colors
from smarts.core.actor import ActorRole
from common.print_color import printColor
import enviorment.SMARTS_correct as sc

class SemanticFrenetGridMap:
    def __init__(self) -> None:
        self.semantic_index = {"ego_vehicle":1, "other_vehicle":2, "road":0, "null":-1}
        self.resolution_s = 1 # m
        self.lane_discrete_num = 6

        self.grid_num_s = 40
        self.grid_num_l = 3 * self.lane_discrete_num
        
        self.lowest_s = -10

        self.rgb_grid_length = 10
        self.rgb_grid_width = 10
        self.color_index = {"ego_vehicle":np.array(Colors.Red.value[0:3], ndmin=3) * 255,
                          "other_vehicle":np.array(Colors.Silver.value[0:3], ndmin=3) * 255, 
                          "road":np.array(Colors.DarkGrey.value[0:3], ndmin=3) * 255,
                          "null":np.array(Colors.Black.value[0:3], ndmin=3) * 255,}
        
        self.rgb = np.zeros((self.rgb_grid_length * self.grid_num_s, 
                             self.rgb_grid_width  * self.grid_num_l, 
                             3), 'int8')
        
        self.reversed_semantic_index = {}
        for key in self.semantic_index.keys():
            self.reversed_semantic_index[self.semantic_index[key]] = key

    def samplePoints(self, frenet_info, width, length):
        self.sample_num_x = math.ceil(length / self.resolution_s) * 2 # longitudinal
        self.sample_num_y = math.ceil(width / self.resolution_l) * 2 # lateral

        frenet_s = frenet_info[0]
        frenet_l = frenet_info[1]
        frenet_yaw = frenet_info[2]
        res_x = length / (self.sample_num_x - 1)
        res_y = width / (self.sample_num_y - 1)
        # 静态采样点
        sampled_points = []
        for i in range(self.sample_num_x):
            for j in range(self.sample_num_y):
                sampled_points.append(np.array([res_x * i - length / 2, res_y * j - width / 2]))
        # 先旋转后平移
        for i in range(len(sampled_points)):
            # rotate_matrix = np.array([[ math.cos(frenet_yaw), math.sin(frenet_yaw)], 
            #                           [-math.sin(frenet_yaw), math.cos(frenet_yaw)]])
            # sampled_points[i] = sampled_points[i].dot(rotate_matrix)
            sampled_points[i] = sampled_points[i] + np.array([frenet_s, frenet_l])
        return sampled_points

    def updateFromSMARTS(self, obs):
        self.data = np.zeros((self.grid_num_s, self.grid_num_l))
        reference_line, lane_width = sc.generateReferenceLine(obs, 
                                        back_rate=0.3,
                                        reference_lane_index=obs.ego_vehicle_state.lane_index)
        self.resolution_l = lane_width / self.lane_discrete_num
        self.lowest_l = self.resolution_l * self.lane_discrete_num * 1.5
        # get the information of the reference line
        self.reference_info = np.zeros(reference_line.sumPointsNum())
        back_points_num = len(reference_line.back_points)
        for i in range(reference_line.sumPointsNum()):
            if i < back_points_num:
                self.reference_info[i] = reference_line.back_points[i].cur 
            else:
                self.reference_info[i] = reference_line.points[i - back_points_num].cur
        # get sementic bev
        for vehicle in obs.neighborhood_vehicle_states:
            lane_index_diff = vehicle.lane_index - obs.ego_vehicle_state.lane_index
            if abs(lane_index_diff) > 2:
                continue
            (is_too_far, frenet_info) =  reference_line.toFrenet(x=vehicle.position[0], 
                                                                 y=vehicle.position[1],
                                                                 yaw=sc.SMARTS_yawCorrect(vehicle.heading),
                                                                 lane_with=lane_width,
                                                                 lane_index_diff=lane_index_diff,
                                                                 offset_from_lane_center=vehicle.lane_position.t)
            if is_too_far:
                continue
            sampling_points = self.samplePoints(frenet_info, 
                                                vehicle.bounding_box.width, 
                                                vehicle.bounding_box.length)
            printColor(frenet_info)
            for p in sampling_points:
                s_index = math.floor((p[0] - self.lowest_s) / self.resolution_s)
                l_index = math.floor((p[1] - self.lowest_l) / self.resolution_l)
                if 0 <= s_index < self.grid_num_s and 0 <= l_index < self.grid_num_l:
                    self.data[s_index][l_index] = self.semantic_index["other_vehicle"]
        # add ego occupancy
        (_, frenet_info) =  reference_line.toFrenet(x=obs.ego_vehicle_state.position[0], 
                                                    y=obs.ego_vehicle_state.position[1],
                                                    yaw=sc.SMARTS_yawCorrect(obs.ego_vehicle_state.heading),
                                                    lane_with=lane_width,
                                                    lane_index_diff=0,
                                                    offset_from_lane_center=obs.ego_vehicle_state.lane_position.t,)
        ego_sampling_points = self.samplePoints(frenet_info, 
                                                obs.ego_vehicle_state.bounding_box.width, 
                                                obs.ego_vehicle_state.bounding_box.length)
        for p in ego_sampling_points:
            s_index = math.floor((p[0] - self.lowest_s) / self.resolution_s)
            l_index = math.floor((p[1] - self.lowest_l) / self.resolution_l)
            self.data[s_index][l_index] = self.semantic_index["ego_vehicle"]
        # add null area
        lane_num = len(obs.waypoint_paths)
        if obs.ego_vehicle_state.lane_index == 0: # 右侧无道路
            self.data[:, self.grid_num_l-self.lane_discrete_num:self.grid_num_l] = -1
        if obs.ego_vehicle_state.lane_index == lane_num - 1: # 左侧无道路
            self.data[:, 0:self.lane_discrete_num] = -1

    def toRGB(self):
        self.rgb = np.zeros((self.rgb_grid_length * self.grid_num_s, 
                             self.rgb_grid_width  * self.grid_num_l, 
                             3), 'int8')
        for i in range(self.grid_num_s):
            for j in range(self.grid_num_l):
                sementic = self.reversed_semantic_index[self.data[i][j]]
                self.rgb[i*self.rgb_grid_length : (i+1)*self.rgb_grid_length, \
                         j*self.rgb_grid_width : (j+1)*self.rgb_grid_width, \
                         :] = self.color_index[sementic]
        return self.rgb
    
    def data(self):
        return self.data
    
    def refLineInfo(self):
        return self.reference_info