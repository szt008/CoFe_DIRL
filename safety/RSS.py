class RSS:
    def __init__(self, lat_safety_margin=0.2, 
                        # reaction_time=0.01, 
                        reaction_time=0.1,
                        max_acc=4.0, min_dec=2.0, max_dec=8.0,
                        min_lat_dcc=4.0, max_lat_acc=4.0):
        self.reaction_time = reaction_time

        self.max_acc = max_acc
        self.min_dec = min_dec # 越小导致需要的安全距离越大
        self.max_dec = max_dec

        self.max_lat_acc = max_lat_acc
        self.min_lat_dec = min_lat_dcc # 越小导致需要的安全距离越大

    def minLonDistance(self, v_front, v_rear):
        rss_distance = max(0, 
                            v_rear * self.reaction_time \
                            + self.max_acc * self.reaction_time**2 / 2 \
                            + (v_rear + self.max_acc * self.reaction_time)**2 / (2 * self.min_dec) \
                            - v_front**2 / (2 * self.max_dec) \
                            )
        return rss_distance

    def reviseAction(self, action, ego_vehicle_state):
        min_lon_distance = self.minLonDistance()

        return action