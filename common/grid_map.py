import numpy as np
import math

class FrenetGridMap:
    def __init__(self) -> None:
        self.resolution_s = 3.2 / 3
        self.resolution_l = 2
        self.grid_num_s = 9
        self.grid_num_l = 30
        self.map_min_s = -10
        self.map_min_l = -3.2 * 1.5

        self.map_max_s = self.grid_num_s * self.resolution_s + self.map_min_s
        self.map_max_l = self.grid_num_l * self.resolution_l + self.map_min_l

        self.obstacles = []
        
    def generateOccupancyMap(self):
        map = np.array.zeros((self.grid_num_l, self.grid_num_s))
        for obstacle in self.obstacles:
            if obstacle.has_frenet_info:
                occupancy_points = obstacle.sampleInFrenet()
                for occupancy_point in occupancy_points:
                    if occupancy_point.s >= self.map_min_s and occupancy_point.s <= self.map_max_s \
                            and occupancy_point.l >= self.map_min_l and occupancy_point.l <= self.map_max_l:
                        index_s = int((occupancy_point.s - self.map_min_s) / self.resolution_s)
                        index_l = int((occupancy_point.l - self.map_min_l) / self.resolution_l)
                        map[index_l, index_s] = 1
            else:
                print("Obstacle dosen't have frenet info!")
        return map
