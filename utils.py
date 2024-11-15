#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
from common.print_color import printColor
from safety.safety_access import safetyAccess
from safety.safe_shielding import ShieldingState
import hyper_param as hp
import enviorment.SMARTS_correct as sc
from enviorment.get_grid_index import getGridIndex, getLaneChangeGridIndex

def toOneHot(lane_change):
    if lane_change == -1:
        return np.array([1, 0, 0])
    elif lane_change == 0:
        return np.array([0, 1, 0])
    elif lane_change == 1:
        return np.array([0, 0, 1])
    else:
        return np.array([0, 0, 0])

def actionAdapter(action, ego_vehicle, is_deterministic=False):
    dt = 0.1
    # velocity = ego_vehicle.speed + action[0] * dt # 加速度
    velocity = action[0]

    lane_change_array = action[1]
    if torch.is_tensor(velocity):
        velocity = velocity.item()
    if torch.is_tensor(lane_change_array):
        lane_change_array = lane_change_array.cpu().detach().numpy().squeeze()

    lane_change_num = len(lane_change_array)
    if not is_deterministic:
        lane_change = int(np.random.choice(lane_change_num,1,p=lane_change_array))-int(lane_change_num/2)
    else:
        lane_change = int(np.argmax(lane_change_array) - int(lane_change_num/2))
    return (velocity, lane_change)

def normalizeList(lst):
    total = sum(lst)
    return [x / total for x in lst]

def actionAdapterLat(action, is_deterministic=False):
    lane_change_array = action
    if torch.is_tensor(lane_change_array):
        lane_change_array = lane_change_array.cpu().detach().numpy().squeeze()
    lane_change_num = len(lane_change_array)
    if not is_deterministic:
        lane_change_array = normalizeList(lane_change_array)
        lane_change = int(np.random.choice(lane_change_num,1,p=lane_change_array))-int(lane_change_num/2)
    else:
        lane_change = int(np.argmax(lane_change_array) - int(lane_change_num/2))
    return lane_change
    
def actionAdapterLon(acc_decision, lane_change, current_speed):
    dt = 0.1
    velocity = current_speed + acc_decision[lane_change + int(np.size(acc_decision) / 2)]* dt
    # if velocity < 1: #现在通过shielding来实现
    #     velocity = 0 # 防止向前溜车
    return velocity

def extractState(obs, reference_line, debug=False):
    feature_num = 5
    # get ego vehicle state
    ego_vehicle = obs.ego_vehicle_state
    ego_heading = sc.SMARTS_yawCorrect(ego_vehicle.heading, reference_line.points[0].yaw)
    frenet_info = reference_line.toFrenet(ego_vehicle.position[0], \
                                            ego_vehicle.position[1], \
                                            ego_heading)
    
    if frenet_info == None:
        printColor("extractState: Ego vehicle is far from reference_line.", 'r')
        return None
    else:
        frenet_s, frenet_l, frenet_heading = frenet_info

    l_gap = frenet_l - ego_vehicle.lane_position.t
    if abs(l_gap) > 1: # 检查并处理reference_line偏移了一个车道的情况
        frenet_l -= l_gap
        for p in reference_line.points + reference_line.back_points: # 对本车l和参考线进行偏移
            p.x += math.cos(p.yaw + math.pi / 2) * l_gap
            p.y += math.sin(p.yaw + math.pi / 2) * l_gap

    # FEATURE ID:-1=no road, 0=nothing, 1=ego vehicle, 2=traffic vechicle，3=walker
    ego_traffic_feature = np.array([frenet_s, frenet_l, frenet_heading, obs.ego_vehicle_state.speed, 1])
    
    # get traffic feature (d for directions) (g for grids)
    traffic_feature_d = np.zeros((hp.traffic_direction_num, feature_num))
    min_distances_d = np.ones(hp.traffic_direction_num) * np.inf
    direction_resolution = 2 * math.pi / hp.traffic_direction_num
    frenet_origin = reference_line.points[0]

    traffic_feature_g = np.zeros((20, feature_num))
    lane_changes = [-2, -1, 0, 1, 2]
    for lane_change in lane_changes: # 处于最右侧车道
        target_lane_index = lane_change + ego_vehicle.lane_index
        
        if target_lane_index < 0 or target_lane_index >= len(obs.waypoint_paths):
            for grid_index in getLaneChangeGridIndex(lane_change):
                traffic_feature_g[grid_index, 4] = -1
                
    min_distances_g = np.ones(20) * np.inf
    for vehicle in obs.neighborhood_vehicle_states:
        distance = math.sqrt((vehicle.position[1]-ego_vehicle.position[1])**2 + (vehicle.position[0]-ego_vehicle.position[0])**2)
        frenet_info = reference_line.toFrenet(vehicle.position[0], \
                                              vehicle.position[1], \
                                              sc.SMARTS_yawCorrect(vehicle.heading, reference_line.points[0].yaw))
        if frenet_info == None:
            continue
        (frenet_s, frenet_l, frenet_heading) = frenet_info
        
        # add to traffic feature (GRID)
        lane_widths = sc.getLaneWidths(obs)
        grid_index = getGridIndex(frenet_s, frenet_l, lane_widths, ego_vehicle.speed)
        if not grid_index == -1 and distance < min_distances_g[grid_index]:
            min_distances_g[grid_index] = distance
            traffic_feature_g[grid_index] = np.array([frenet_s, frenet_l, frenet_heading, vehicle.speed, 2])
    
    # get reference information
    reference_info = np.zeros(hp.reference_info_num)
    for i in range(min(hp.reference_info_num, len(reference_line.points))):
        reference_info[i] = reference_line.points[i].cur 

    state = {"traffic_feature_d":traffic_feature_d, # by direction
             "traffic_feature_g":traffic_feature_g, # by grid
             "ego_traffic_feature": ego_traffic_feature,
             "reference_info":reference_info,}
    return state

def observationAdapter(curr_state, obs, agent=None):
    if obs.ego_vehicle_state.lane_id == 'off_lane':
        return np.array(curr_state)

    reference_line = sc.generateRefLineFromRoad(obs)
    state = extractState(obs, reference_line)
    if not agent == None:
        agent.reference_line = reference_line

    if state == None:
        return np.array(curr_state)

    for i in range(hp.history_frame_num-1):
        curr_state[:, :, i] = curr_state[:, :, i+1]
    curr_state[0, :, hp.history_frame_num-1] = state["ego_traffic_feature"]
    curr_state[1:21, :, hp.history_frame_num-1] = state["traffic_feature_g"]

    return np.array(curr_state)
    # return (obs.ego_vehicle_state.speed)

def rewardAdapter(obs, action=None, safe_shielding_state=None):
    if obs.ego_vehicle_state.lane_id == 'off_lane':
        is_off_lane = True # no lane id
    else:
        is_off_lane = False
    
    if not is_off_lane:
        reference_line = sc.generateRefLineFromRoad(obs)
        safety_features = safetyAccess(obs, reference_line)
    else:
        safety_features = {'surrounding_vehicle_num':0, 
                            'mean_approaching_velocity':0,
                            'ttc':np.inf}
        
    if safe_shielding_state == None:
        safe_shielding_state = ShieldingState()

    performance = {'event': obs.events, 
                   'ego_state': obs.ego_vehicle_state,
                   'safety_features': safety_features,
                   'off_lane': is_off_lane,
                   'safe_shielding_state':safe_shielding_state,
                   'action': action}
    
    reward, reward_components = performanceToReward(performance)
    return reward, performance, reward_components

def performanceToReward(performance):
    if not 'safe_shielding_is_active' in performance:
        performance['safe_shielding_is_active'] = 0
        
    reward_components = {}

    reward_components['alive'] = 0.1

    target_speed = 15 # 54 km/h
    if performance['ego_state'].speed >= target_speed:
        reward_components['speed'] = 1
    else:
        reward_components['speed'] = performance['ego_state'].speed / target_speed
    reward_components['speed'] *= 0.2
    
    if performance['event'].collisions:
        reward_components['crash'] = -10

    if 'safe_shielding_state' in performance:
        if performance['safe_shielding_state'].is_lon_active:
            reward_components['safe_shielding_lon'] = -0.2 # -0.2
        if performance['safe_shielding_state'].is_lat_active:
            reward_components['safe_shielding_lat'] = -0.2 # -0.2

    if 'ttc_lon' in performance['safety_features']:    
        if performance['safety_features']['ttc_lon'] < 3:
            reward_components['ttc_lon'] = max(- 1 / (performance['safety_features']['ttc_lon'] + 0.000001), -20) # ttc为0.1时，惩罚为-10

    if 'ttc_lat' in performance['safety_features']:    
        if performance['safety_features']['ttc_lat'] < 3:
            reward_components['ttc_lat'] = max(- 1 / (performance['safety_features']['ttc_lat'] + 0.000001), -20) # ttc为0.1时，惩罚为-10

    reward = 0
    for key in reward_components:
        reward += reward_components[key]
    
    return reward, reward_components