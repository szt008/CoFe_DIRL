import math
import numpy as np
import itertools as it

import enviorment.SMARTS_correct as sc
from common.print_color import printColor

class RSS:
    def __init__(self, lat_safety_margin=0.2, 
                        reaction_time=0.01, 
                        max_acc=4.0, min_dec=2.0, max_dec=8.0,
                        min_lat_dcc=4.0, max_lat_acc=4.0):
        self.reaction_time = reaction_time

        self.max_acc = max_acc
        self.min_dec = min_dec # 越小导致需要的安全距离越大
        self.max_dec = max_dec

        self.max_lat_acc = max_lat_acc
        self.min_lat_dec = min_lat_dcc # 越小导致需要的安全距离越大

    def minLonDistance(self, v_front, v_rear):
        return max(0, 
                v_rear * self.reaction_time \
                + self.max_acc * self.reaction_time**2 / 2 \
                + (v_rear + self.max_acc * self.reaction_time)**2 / (2 * self.min_dec) \
                - v_front**2 / (2 * self.max_dec) \
                )

    def minLatDistance(self, v_left, v_right):
        v_left_reaction = v_left + self.reaction_time * self.max_lat_acc
        v_right_reaction = v_left + self.reaction_time * self.max_lat_acc
        return self.lat_safety_margin \
                + max(0, ( \
                    self.reaction_time * (v_left + v_left_reaction) / 2 \
                    + v_left_reaction**2 / (2 * self.min_lat_acc) 
                ))

def safetyAccess(obs, reference_line):
    ego_lane = 1
    ego_v = obs.ego_vehicle_state.speed
    # get neighbor_vehicle_num and mean_approaching_velocity
    neighbor_vehicle_num = 0
    sum_approching_velocity = 0
    mean_approaching_velocity = 0 # 所有车接近本车的速度
    for neighborhood_vehicle in obs.neighborhood_vehicle_states:
        neighbor_vehicle_num += 1

    # get time to collision
    ttc = np.inf
    min_relative_s = np.inf
    # for (relative_s, _, _, v) in it.chain(grids[1][ego_lane], grids[2][ego_lane]):
    #     if min_relative_s > relative_s: # get min_s_gap and front v
    #         min_relative_s = relative_s
    #         front_v = v

    (ego_frenet_s, ego_frenet_l, ego_frenet_yaw) = reference_line.toFrenet(obs.ego_vehicle_state.position[0], 
                                                                            obs.ego_vehicle_state.position[1], 
                                                                            sc.SMARTS_yawCorrect(obs.ego_vehicle_state.heading))
    for neighbor_vehicle in obs.neighborhood_vehicle_states:
        # 选取距离参考线一定范围内的车辆
        frenet_info = reference_line.toFrenet(neighbor_vehicle.position[0], 
                                                neighbor_vehicle.position[1], 
                                                sc.SMARTS_yawCorrect(neighbor_vehicle.heading))
        if frenet_info == None:
            continue
        else:
            (frenet_s, frenet_l, frenet_yaw) = frenet_info
        tmp_relative_s = frenet_s - ego_frenet_s
        tmp_relative_l = frenet_l - ego_frenet_l
        if tmp_relative_l > -obs.ego_vehicle_state.bounding_box.width/2 - neighbor_vehicle.bounding_box.width/2 and \
            tmp_relative_l < obs.ego_vehicle_state.bounding_box.width/2 + neighbor_vehicle.bounding_box.width/2 and \
            tmp_relative_s > 0: # 判断车辆是否在沿着参考线正前方
            if tmp_relative_s < min_relative_s:
                min_relative_s = tmp_relative_s
                front_v_s = neighbor_vehicle.speed * math.cos(frenet_yaw)
                front_vehicle_length_s = neighbor_vehicle.bounding_box.length * math.cos(frenet_yaw)

    ego_v_s = obs.ego_vehicle_state.speed * math.cos(ego_frenet_yaw)
    if min_relative_s != np.inf and ego_v_s > front_v_s:
        front_gap = max(min_relative_s - front_vehicle_length_s/2 - obs.ego_vehicle_state.bounding_box.length/2, 0) # 5 is vehicle length
        ttc = front_gap / (ego_v_s - front_v_s)

    # get lateral ttc
    ttc_lat = np.inf
    min_relative_l = np.inf
    ego_v_l = obs.ego_vehicle_state.speed * math.sin(ego_frenet_yaw)
    for neighbor_vehicle in obs.neighborhood_vehicle_states:
        # 选取在侧向上有碰撞可能的车辆
        frenet_info = reference_line.toFrenet(neighbor_vehicle.position[0], 
                                                neighbor_vehicle.position[1], 
                                                sc.SMARTS_yawCorrect(neighbor_vehicle.heading))
        if frenet_info == None:
            continue
        else:
            (frenet_s, frenet_l, frenet_yaw) = frenet_info
        tmp_relative_s = frenet_s - ego_frenet_s
        tmp_relative_l = frenet_l - ego_frenet_l

        if abs(tmp_relative_s) < (obs.ego_vehicle_state.bounding_box.length/2 + neighbor_vehicle.bounding_box.length/2) * 1.1 and \
            tmp_relative_l * ego_v_l >= 0: # 确保侧向车辆与侧向速度方向一致，且纵向位置可能发生碰撞
            if abs(tmp_relative_l) < min_relative_l:
                min_relative_l = abs(tmp_relative_l)
                front_v_l = neighbor_vehicle.speed * math.sin(frenet_yaw)

    if min_relative_l != np.inf and ego_v_l != front_v_l:
        lateral_gap = max(min_relative_l - neighbor_vehicle.bounding_box.width/2 - obs.ego_vehicle_state.bounding_box.width/2, 0) # 5 is vehicle length
        ttc_lat = lateral_gap / abs(ego_v_l - front_v_l)
                
            
    safety_features = {'surrounding_vehicle_num':neighbor_vehicle_num, 
                       'mean_approaching_velocity':mean_approaching_velocity,
                       'ttc_lon':ttc,
                       'ttc_lat': ttc_lat}
    # if ttc < 10:
    #     printColor('front_gap is ' + str(front_gap), 'r')
    #     printColor('ttc is ' + str(ttc), 'r') 
    # if ttc_lat < 10:
    #     printColor('lateral_gap is ' + str(lateral_gap), 'r') 
    #     printColor('ttc_lat is ' + str(ttc_lat), 'r') 
    return safety_features