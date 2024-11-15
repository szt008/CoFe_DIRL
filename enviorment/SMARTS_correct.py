import math
from common.print_color import printColor
from common.trajectory import Trajectory, TrajPoint

def SMARTS_yawCorrect(yaw, ref_yaw=None):
    #对SMARTS进行坐标转换
    yaw += math.pi / 2
    #使得航向角不会由于到了180与-180处突变
    if not ref_yaw == None:
        while abs(yaw - ref_yaw) > math.pi:
            if yaw < ref_yaw:
                yaw += math.pi * 2
            else:
                yaw -= math.pi * 2
    return yaw

def toSMARTS_yaw(yaw):
    return yaw - math.pi / 2
    
def generateReferenceLine(obs, ref_lane_index=1, back_rate=0):
    reference_line = Trajectory()
    num_trajectory_points = len(obs.waypoint_paths[ref_lane_index])
    for i in range(num_trajectory_points):
        if i == 0:
            yaw=SMARTS_yawCorrect(obs.waypoint_paths[ref_lane_index][i].heading, obs.waypoint_paths[ref_lane_index][i].heading)
        else:
            yaw=SMARTS_yawCorrect(obs.waypoint_paths[ref_lane_index][i].heading, reference_line.points[-1].yaw)
        p = TrajPoint(x=obs.waypoint_paths[ref_lane_index][i].pos[0], y=obs.waypoint_paths[ref_lane_index][i].pos[1], yaw = yaw)
        if i < num_trajectory_points-1:
            p.ds_to_next = math.sqrt((p.x - obs.waypoint_paths[ref_lane_index][i+1].pos[0])**2 + (p.y - obs.waypoint_paths[ref_lane_index][i+1].pos[1])**2)
            
            p.cur = (SMARTS_yawCorrect(obs.waypoint_paths[ref_lane_index][i+1].heading, p.yaw) - p.yaw) / p.ds_to_next
        if i > 0:
            p.s = reference_line.points[-1].s + reference_line.points[-1].ds_to_next
        reference_line.points.append(p)
    reference_line.points[-1].s= reference_line.points[-2].s
    return reference_line, obs.waypoint_paths[0][0].lane_width

def generateRefLineFromRoad(obs):
    reference_line = Trajectory()
    ego_lane = obs.road_waypoints.lanes[obs.ego_vehicle_state.lane_id][0]
    ego_x = obs.ego_vehicle_state.position[0]
    ego_y = obs.ego_vehicle_state.position[1]
    
    for i in range(len(ego_lane)):
        distance = math.sqrt((ego_x - ego_lane[i].pos[0])**2 + (ego_y - ego_lane[i].pos[1])**2)
        if i == 0 or distance < min_distance:
            min_distance = distance
            ego_index = i
    forward_num = len(ego_lane) - ego_index
    backward_num = ego_index
    # add forward points
    for i in range(forward_num):
        if i == 0:
            yaw=SMARTS_yawCorrect(ego_lane[ego_index+i].heading)
        else:
            yaw=SMARTS_yawCorrect(ego_lane[ego_index+i].heading, reference_line.points[-1].yaw)
        p = TrajPoint(x=ego_lane[ego_index+i].pos[0], y=ego_lane[ego_index+i].pos[1], yaw = yaw)
        if i < forward_num-1:
            p.ds_to_next = math.sqrt((p.x - ego_lane[ego_index+i+1].pos[0])**2 + (p.y - ego_lane[ego_index+i+1].pos[1])**2)
            p.cur = (SMARTS_yawCorrect(ego_lane[ego_index+i+1].heading, p.yaw) - p.yaw) / p.ds_to_next
        if i > 0:
            p.s = reference_line.points[-1].s + reference_line.points[-1].ds_to_next
        reference_line.points.append(p)
    # add back points
    for i in range(backward_num):
        reference_line.back_points.append(TrajPoint())
    last_point = reference_line.points[0]
    for i in range(backward_num):
        index = ego_index - i - 1
        yaw = SMARTS_yawCorrect(ego_lane[index].heading, last_point.yaw)
        reference_line.back_points[-i-1].x = ego_lane[index].pos[0]
        reference_line.back_points[-i-1].y = ego_lane[index].pos[1]
        reference_line.back_points[-i-1].yaw = yaw
        ds_to_next = reference_line.back_points[-i-1].distanceTo(last_point)
        reference_line.back_points[-i-1].ds_to_next = ds_to_next
        reference_line.back_points[-i-1].s = last_point.s - ds_to_next
        reference_line.back_points[-i-1].cur = (last_point.yaw - yaw) / ds_to_next
        last_point = reference_line.back_points[-i-1]

    return reference_line

def egoFrenetHeading(obs):
    lane_num = len(obs.waypoint_paths)
    ref_lane_index = math.floor(lane_num / 2)
    ref_heading = SMARTS_yawCorrect(obs.waypoint_paths[ref_lane_index][0].heading, None)
    ego_heading = SMARTS_yawCorrect(obs.ego_vehicle_state.heading, ref_heading)
    frenet_heading = ego_heading - ref_heading
    return frenet_heading


def getLaneWidths(obs):
    lane_num = len(obs.waypoint_paths)
    lane_widths = {-2:3.5, -1:3.5, 0:3.5, 1:3.5, 2:3.5}
    lane_diffs = [-2, -1, 0, 1, 2]
    for lane_diff in lane_diffs:
        lane_index = obs.ego_vehicle_state.lane_index + lane_diff
        if lane_index >= 0 and lane_index <= lane_num-1:
            lane_widths[lane_index] = obs.waypoint_paths[lane_index][0].lane_width
    return lane_widths