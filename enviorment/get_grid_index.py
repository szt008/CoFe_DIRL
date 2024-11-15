from common.print_color import printColor
import hyper_param as hp

def getGridIndex(frenet_s, frenet_l, lane_widths, ego_speed):
    if -lane_widths[0] / 2 < frenet_l < lane_widths[0] / 2:
        lane_diff = 0
    else:
        if frenet_l > 0: # left lane
            if frenet_l < lane_widths[0] / 2 + lane_widths[1]:
                lane_diff = 1
            elif frenet_l < lane_widths[0] / 2 + lane_widths[1] + lane_widths[2]:
                lane_diff = 2
            else:
                return -1
        else: # right lane
            if frenet_l > -lane_widths[0] / 2 - lane_widths[-1]:
                lane_diff = -1
            elif frenet_l > -lane_widths[0] / 2 - lane_widths[-1] - lane_widths[-2]:
                lane_diff = -2
            else:
                return -1
        
    if frenet_s > hp.center_grid_length/2 + max(hp.first_grid_length, hp.first_grid_time_gap * ego_speed):
        if lane_diff == 2:
            return 0
        elif lane_diff == 1:
            return 1
        elif lane_diff == 0:
            return 2
        elif lane_diff == -1:
            return 3
        elif lane_diff == -2:
            return 4
        else:
            return -1
    elif frenet_s > hp.center_grid_length/2:
        if lane_diff == 2:
            return 5
        elif lane_diff == 1:
            return 6
        elif lane_diff == 0:
            return 7
        elif lane_diff == -1:
            return 8
        elif lane_diff == -2:
            return 9
        else:
            return -1
    elif -hp.center_grid_length/2 <= frenet_s <= hp.center_grid_length/2:
        if lane_diff == 2:
            return 10
        elif lane_diff == 1:
            return 11
        elif lane_diff == 0:
            return 12
        elif lane_diff == -1:
            return 13
        elif lane_diff == -2:
            return 14
        else:
            return -1
    elif -10-hp.center_grid_length/2 < frenet_s < -hp.center_grid_length/2:
        if lane_diff == 2:
            return 15
        elif lane_diff == 1:
            return 16
        elif lane_diff == 0:
            return 17
        elif lane_diff == -1:
            return 18
        elif lane_diff == -2:
            return 19
        else:
            return -1
    else:   
        return -1
    

def getLaneChangeGridIndex(lane_diff):
    if lane_diff == 2:
        return [0, 5, 10, 15]
    elif lane_diff == 1:
        return [1, 6, 11, 16]
    elif lane_diff == 0:
        return [2, 7, 12, 17]
    elif lane_diff == -1:
        return [3, 8, 13, 18]
    elif lane_diff == -2:
        return [4, 9, 14, 19]
    else:
        return []
    
def getDirectionGridIndex(direction):
    if direction == "front":
        return [2, 7, 12]
    if direction == "rear":
        return [12, 17]
    if direction == "left":
        return [6, 11, 16, 12]
    if direction == "right":
        return [8, 13, 18, 12]
    return []