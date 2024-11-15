import math
import copy 
from common.math import linearInterpolation
from common.trajectory import TrajPoint

class ObstacleSampling:
    def __init__(self, size, shape) -> None:
        self.points = []
        if shape == "rectangle":
            self.length = size[0]
            self.width = size[1]
            self.lat_sampling_num = 3
            self.lon_sampling_num = 5

            lon_resolution = self.length / (self.lon_sampling_num - 1)
            min_lon = -self.length / 2
            lon_sampling = []
            for i in range(self.lon_sampling_num):
                lon_sampling.append(min_lon + i * lon_resolution)

            lat_resolution = self.width  / (self.lat_sampling_num - 1)
            min_lat = -self.width / 2
            lat_sampling = []
            for i in range(self.lat_sampling_num):
                lat_sampling.append(min_lat + i * lat_resolution)

            for i in range(self.lon_sampling_num):
                for j in range(self.lat_sampling_num):
                    p = TrajPoint(x=lon_sampling[i], y=lat_sampling[j])
                    self.points.append(p)
        # elif self.shape == "circle":
        #     self.radius = size[0]
        #     self.sampling_nums = [3, 6]
        else:
            print("The obstacle shape is not modeled yet!")


    def transformTo(self, x, y, yaw, is_in_frenet=False):
        result = []
        for p in self.points:
            new_x = p.x * math.cos(yaw) - p.y * math.sin(yaw) + x
            new_y = p.x * math.sin(yaw) + p.y * math.cos(yaw) + y
            if is_in_frenet:
                new_l = new_y
                new_s = new_x
            else:
                new_l = 0
                new_s = 0
            result.append(TrajPoint(x=new_x, y=new_y, frenet_l=new_l, frenet_s=new_s))
        return result


class Obstacle:
    def __init__(self, x=0, y=0, yaw=0, size=[0, 0], shape = "rectangle") -> None:
        self.x = x
        self.y = y
        self.yaw = yaw

        self.sampling = ObstacleSampling(size, shape)
        self.has_frenet_info = False

    def calculateFrenetInfo(self, reference_line, resolution = 0.1, back_extension_distance = 10):
        min_distance = math.inf
        
        # 检测处于本车后方的障碍物
        last_ref_yaw = reference_line.points[0].yaw
        last_ref_x = reference_line.points[0].x
        last_ref_y = reference_line.points[0].y
        s = -resolution
        while s > -back_extension_distance:
            ref_x = last_ref_x - math.cos(last_ref_yaw) * resolution
            ref_y = last_ref_y - math.sin(last_ref_yaw) * resolution
            ref_yaw = last_ref_yaw - resolution * reference_line.points[0].cur
            last_ref_yaw = ref_yaw
            last_ref_x = ref_x
            last_ref_y = ref_y
            s -= resolution
            distance =  math.sqrt((ref_x - self.x)**2 + (ref_y - self.y)**2)
            if distance < min_distance:
                min_distance = distance
                result_s = s
                reference_yaw = ref_yaw
        
        # 检测处于本车前方的障碍物
        for i in range(len(reference_line.points)-1):
            s = reference_line.points[i].s
            heading_to_next_point = math.atan2(reference_line.points[i+1].y - reference_line.points[i].y, \
                                                reference_line.points[i+1].x - reference_line.points[i].x)
            while s < reference_line.points[i + 1].s:
                rate = (reference_line.points[i+1].s - 1) / reference_line.points[i].ds_to_next
                ref_x = linearInterpolation(reference_line.points[i].x, reference_line.points[i+1].x, rate) 
                ref_y = linearInterpolation(reference_line.points[i].y, reference_line.points[i+1].y, rate) 
                ref_yaw = linearInterpolation(reference_line.points[i].yaw, reference_line.points[i+1].yaw, rate) 
                distance =  math.sqrt((ref_x - self.x)**2 + (ref_y - self.y)**2)
                if distance < min_distance:
                    min_distance = distance
                    result_s = s
                    reference_yaw = ref_yaw
                s += resolution
            
        if min_distance < 10:
            self.has_frenet_info = True
            self.frenet_s = result_s
            self.frenet_l = min_distance
            self.frenet_yaw = self.yaw - reference_yaw
        else:
            print("Obstacle is too far from reference line!")
        return self.has_frenet_info


    def sampleInFrenet(self) -> list:
        if self.has_frenet_info:
            return self.sampling.transformTo(self.frenet_s, self.frenet_l, self.frenet_yaw, is_in_frenet=True)
        else:
            print("Obstacle dosen't have Frenet Info!")
            return []