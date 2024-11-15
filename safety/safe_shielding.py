import hyper_param as hp
from common.print_color import printColor
from safety.RSS import RSS

class ShieldingState:
    def __init__(self):
        self.is_lat_active = False
        self.is_lon_active = False


class FixedDistanceShielding:
    def __init__(self, emergent_acc=4, emergent_dec=-8, acc_dt=0.1) -> None:
        self.emergent_dec = emergent_dec
        self.dt = acc_dt
        self.RSS = RSS()

        self.state = ShieldingState()
        pass

    def getState(self):
        return self.state

    def safeLatAction(self, lane_change, state):
        self.state.is_lat_active = False
        current_frame_state = state[:, :, hp.history_frame_num-1]
        if lane_change == 1 or lane_change == -1: # 如果正前方有车，来不及完成换道动作的情况
            if current_frame_state[13, 4] and current_frame_state[13, 0] < 4:
                lane_change = 0
                self.state.is_lat_active = True

        if lane_change == 1: # steer left
            if current_frame_state[12, 4]: # 邻近左侧栅格有车
                lane_change = 0
                self.state.is_lat_active = True
            elif current_frame_state[17, 4]:  # 左后侧栅格有车
                rear_v = current_frame_state[17, 3]
                rear_s = current_frame_state[17, 0]
                ego_v = current_frame_state[0, 3]
                ego_s = current_frame_state[0, 0]
                relative_v = rear_v - ego_v
                if relative_v > 0:
                    ttc = (ego_s - rear_s) / relative_v
                    if ttc < 0.5:
                        lane_change = 0
                        self.state.is_lat_active = True
        elif lane_change == -1: # steer right
            if current_frame_state[14, 4]: # 邻近右侧栅格有车
                lane_change = 0
                self.state.is_lat_active = True
            elif current_frame_state[19, 4]: # 左右侧栅格有车
                rear_v = current_frame_state[19, 3]
                rear_s = current_frame_state[19, 0]
                ego_v = current_frame_state[0, 3]
                ego_s = current_frame_state[0, 0]
                relative_v = rear_v - ego_v
                if relative_v > 0:
                    ttc = (ego_s - rear_s) / relative_v
                    if ttc < 0.5:
                        lane_change = 0
                        self.state.is_lat_active = True

        return lane_change
    
    def safeLonAction(self, velocity, state):
        self.state.is_lon_active = False
        current_frame_state = state[:, :, hp.history_frame_num-1]
        ego_feature = current_frame_state[0, :]
        traffic_feature = current_frame_state[1:, :]

        there_is_front_vehicle = False
        # check if there is front vehicle
        front_grid_indexes = [12, 7, 2]
        for grid_index in front_grid_indexes:
            if traffic_feature[grid_index, 4] == 2 and traffic_feature[grid_index, 0] > 0:
                front_v = traffic_feature[grid_index, 3]
                front_s = traffic_feature[grid_index, 0]
                there_is_front_vehicle = True
                break

        # check if ttc to the front vehicle
        if there_is_front_vehicle:
            ego_s = ego_feature[0]
            ego_v = ego_feature[3]

            # get dtc and rss_threshold
            dtc = (front_s - ego_s) - 5 # m
            rss_distance = self.RSS.minLonDistance(front_v, ego_v)

            # get ttc
            relative_v = ego_v - front_v 
            if relative_v > 0:
                ttc = dtc / relative_v
            else:
                ttc = float('inf')

            emergency_velocity = ego_v + self.dt * self.emergent_dec
            if (ttc <= 0.5 or dtc < rss_distance) and velocity > emergency_velocity:
                self.state.is_lon_active = True
                velocity = emergency_velocity
        return velocity

    def isTriggered(self, state, action):
        self.safeLonAction(action[0], state)
        self.safeLatAction(action[1], state)
        self.is_active = self.state.is_lat_active or self.state.is_lon_active
        if self.is_active:
            return 1
        return 0
    
    def isActive(self):
        self.is_active = self.state.is_lat_active or self.state.is_lon_active
        if self.is_active:
            return 1
        return 0