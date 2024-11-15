import numpy as np
import csv
import os
import math

import queue
from collections import deque 

from common.math import linearInterpolation
from common.math import signedDistToLine

class TrajPoint:
    def __init__(self, x=0, y=0, yaw=0, cur=0, 
                       frenet_l=0, frenet_s=0, frenet_yaw=0,
                       vx=0, t=0):
        # Cartesian Info 笛卡尔坐标系下的路径信息
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cur = cur
        self.s = 0
        self.ds_to_next = 0

        # Temporal Info 时间信息
        self.vx = 0
        self.ax = 0
        self.t = 0

        # Frenet Info 道路坐标系下的路径信息
        self.frenet_l = frenet_l
        self.frenet_s = frenet_s
        self.frenet_yaw = frenet_yaw
        
        # Differentiation of Distance 时间信息
        self.ax_dot = 0 #关于路程的微分
        self.cur_dot = 0 #关于路程的微分

        self.l_min = 0
        self.l_max = 0
        self.vx_min = 0
        self.vx_max = 100
        self.ax_min = -10
        self.ax_max = 10

    def setFrenetLCorridor(self, center, half_width=0): # Spatio-temporal Corridor Boundary 时空安全走廊
        self.l_min = center - half_width
        self.l_max = center + half_width

    def setVxCorridor(self, center, half_width=0): # Spatio-temporal Corridor Boundary 时空安全走廊
        self.vx_min = center - half_width
        self.vx_max = center + half_width

    def setVxMinMax(self, input_min, input_max):
        self.vx_min = input_min
        self.vx_max = input_max
        
    def setAxMinMax(self, input_min, input_max):
        self.ax_min = input_min
        self.ax_max = input_max
    
    def distanceTo(self, other_point):
        return math.sqrt((self.x - other_point.x)**2 + (self.y - other_point.y)**2)

    def forward(self, step, direction):
        return TrajPoint(x=self.x+math.cos(direction)*step, y=self.y+math.sin(direction)*step)
    # def setSafeCorridor(self, t_min=0, t_max=0, l_min=0, l_max=0): # Spatio-temporal Corridor Boundary 时空安全走廊
    #     self.t_min = t_min
    #     self.t_max = t_max
    #     self.l_min = l_min
    #     self.l_max = l_max

class Trajectory:
    def __init__(self):
        self.points = []
        self.back_points = []
        # self.points = queue.Queue()


    def rotate(self, theta): # 旋转
        for i in range(0, len(self.points)):
            delta_x = self.points[i].x - self.points[0].x
            delta_y = self.points[i].y - self.points[0].y
            self.points[i].x = self.points[0].x + delta_x * math.cos(theta) - delta_y * math.sin(theta)
            self.points[i].y = self.points[0].y + delta_x * math.sin(theta) + delta_y * math.cos(theta)
            self.points[i].yaw = self.points[i].yaw + theta


    def translate(self, x = 0, y = 0): # 平移
        for i in range(0, len(self.points)):
            self.points[i].x += x
            self.points[i].y += y


    def calculateS(self):
        self.points[0].s = 0
        self.points[0].ds_to_next = math.sqrt((self.points[1].x - self.points[0].x)**2 + (self.points[1].y - self.points[0].y)**2)
        for i in range(1,len(self.points) - 1):
            self.points[i].s = self.points[i - 1].s + self.points[i - 1].ds_to_next
            self.points[i].ds_to_next = math.sqrt((self.points[i + 1].x - self.points[i].x)**2 + (self.points[i + 1].y - self.points[i].y)**2)
        self.points[-1].s = self.points[-2].s + self.points[-2].ds_to_next
        self.points[-1].ds_to_next = 0

    def calculateCartesianInfo(self, reference_line): # 要求一一对应
        for i in range(len(self.points)):
            self.points[i].x = float(reference_line.points[i].x + self.points[i].frenet_l * math.cos(reference_line.points[i].yaw + math.pi / 2))
            self.points[i].y = float(reference_line.points[i].y + self.points[i].frenet_l * math.sin(reference_line.points[i].yaw + math.pi / 2))
            self.points[i].yaw = float(reference_line.points[i].yaw + self.points[i].frenet_yaw)

            self.points[i].frenet_s = float(self.points[i].s)

            self.points[i].l_max = float(reference_line.points[i].l_max - self.points[i].frenet_l)
            self.points[i].l_min = float(reference_line.points[i].l_min - self.points[i].frenet_l)
        self.calculateS()


    def sToYaw(self, s):
        if s > self.points[-1].s:
            print("Inputted s should smaller or equal to the final s of reference line.")
            print("s is "+ str(s))
            print("ref max s is "+ str(self.points[-1].s))
            return
        elif s < 0:
            print("Inputted s should bigger than 0.")
            print("s is "+ str(s))
            return
            
        for i in range(len(self.points)):
            if self.points[i].s == s:
                return self.points[i].yaw
            elif self.points[i].s > s:
                rate = (self.points[i].s - s) / self.points[i - 1].ds_to_next
                return linearInterpolation(self.points[i - 1].yaw, self.points[i].yaw,rate)
            

    def sToCur(self, s):
        if s > self.points[-1].s:
            print("Inputted s should smaller or equal to the final s of reference line.")
            return
        for i in range(len(self.points)):
            if self.points[i].s == s:
                return self.points[i].cur
            elif self.points[i].s > s:
                rate = (self.points[i].s - s) / self.points[i - 1].ds_to_next
                return linearInterpolation(self.points[i - 1].cur, self.points[i].cur,rate)
            

    def xyToS(self, x, y, resolution = 0.1, back_extension_distance = 10):
        min_distance = 999
        result_s = 0
        
        s = -resolution
        while s > -back_extension_distance:
            ref_x = self.points[0].x + math.cos(self.points[0].yaw) * s
            ref_y = self.points[0].y + math.sin(self.points[0].yaw) * s
            s -= resolution
            distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
            if distance < min_distance:
                min_distance = distance
                result_s = s
            
        for i in range(len(self.points)-1):
            s = self.points[i].s
            while s < self.points[i + 1].s:
                ref_x = self.points[i].x + math.cos(self.points[i].yaw) * (s - self.points[i].s)
                ref_y = self.points[i].y + math.sin(self.points[i].yaw) * (s - self.points[i].s)
                distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
                if distance < min_distance:
                    min_distance = distance
                    result_s = s
                s += resolution
            
        is_too_far = False
        if min_distance > 10:
            is_too_far = True
        return (is_too_far, result_s)
    

    # def toFrenet(self, x, y, yaw, resolution = 0.1):
        # min_distance = 20 # 如果最短距离远于20， 那么返回None
        # frenet_info = None

        # back_points_num = len(self.back_points)
        # points_num = len(self.points)
        # # Traverse back_points and points
        # for i in range(points_num + back_points_num -1):
        #     if i < back_points_num: # in back points
        #         current_point = self.back_points[i]
        #         if i != back_points_num-1:
        #             next_point = self.back_points[i+1]
        #         else:
        #             next_point = self.points[0]
        #     else:
        #         current_point = self.points[i - back_points_num]
        #         next_point = self.points[i - back_points_num + 1]

        #     s = current_point.s
        #     while s < next_point.s:
        #         ref_yaw = current_point.yaw + math.cos(current_point.cur) * (s - current_point.s)
        #         ref_x = current_point.x + math.cos(ref_yaw) * (s - current_point.s)
        #         ref_y = current_point.y + math.sin(ref_yaw) * (s - current_point.s)
        #         distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        #         if distance < min_distance:
        #             min_distance = distance
        #             frenet_s = s
        #             frenet_yaw = yaw - ref_yaw
        #             nearest_p = TrajPoint(x=ref_x, y=ref_y, yaw=ref_yaw)
        #         s += resolution

        # frenet_info = (frenet_s, frenet_yaw)
        # return frenet_info
    
    def toFrenet(self, x, y, yaw, resolution = 0.1):
        frenet_info = None
        input_p = TrajPoint(x=x, y=y, yaw=yaw)
        (min_distance, neighbour_points) = self.distanceAndNeighbourPointsTo(input_p)
        for i in range(2):
            current_point = neighbour_points[i]
            next_point = neighbour_points[i+1]
            if current_point == None or next_point == None:
                continue
            s = current_point.s
            direction = math.atan2(next_point.y - current_point.y, next_point.x - current_point.x)
            while s < next_point.s:
                ref_yaw = current_point.yaw + current_point.cur * (s - current_point.s)
                ref_x = current_point.x + math.cos(direction) * (s - current_point.s)
                ref_y = current_point.y + math.sin(direction) * (s - current_point.s)
                tmp_p = TrajPoint(x=ref_x, y=ref_y, yaw=direction)
                unsigned_distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
                if unsigned_distance <= min_distance:
                    min_distance = unsigned_distance
                    left_p = tmp_p.forward(step=1, direction=tmp_p.yaw+math.pi/2)
                    right_p = tmp_p.forward(step=1, direction=tmp_p.yaw-math.pi/2)
                    if left_p.distanceTo(input_p) < right_p.distanceTo(input_p):
                        signed_distance = unsigned_distance
                    else:
                        signed_distance = -unsigned_distance
                    frenet_info = (s, signed_distance, yaw - ref_yaw)
                s += resolution
        return frenet_info

    def sumPointsNum(self):
        return len(self.points) + len(self.back_points)

    def saveInCsvSmartly(self, file_name, attribute_names, angle_mode="Radian"):
        with open(file_name, mode='w', newline='', encoding='utf-8') as f:     
            writer = csv.writer(f)
            writer.writerow(attribute_names)
            for point in self.points:
                content = []
                for attribute in attribute_names:
                    if attribute == "x":
                        content.append(str(point.x))
                    elif attribute == "y":
                        content.append(str(point.y))
                    elif attribute == "yaw" or attribute == "theta":
                        if angle_mode == "Radian":
                            content.append(str(point.yaw))
                        else:
                            content.append(str(point.yaw / math.pi * 180))
                    elif attribute == "cur" or attribute == "kappa":
                        content.append(str(point.cur))
                    elif attribute == "s":
                        content.append(str(point.s))
                    elif attribute == "v" or attribute == "vx":
                        content.append(str(point.vx))
                    elif attribute == "a" or attribute == "ax":
                        content.append(str(point.ax))
                    elif attribute == "t":
                        content.append(str(point.t))
                    elif attribute == "left_width" or attribute == "lw":
                        content.append(str(point.left_width))
                    elif attribute == "right_width" or attribute == "rw":
                        content.append(str(point.right_width))
                    elif attribute == "frenet_s" or attribute == "fs":
                        content.append(str(point.frenet_s))
                    elif attribute == "frenet_l" or attribute == "fl":
                        content.append(str(point.frenet_l))
                    elif attribute == "ax_dot":
                        content.append(str(point.ax_dot))
                    else:
                        content.append(str(0))
                writer.writerow(content)
        print('saveInCsv successfully')


    def offset(self, distance):
        for p in self.points:
            angle = p.yaw + math.pi / 2
            p.x = p.x + distance * math.cos(angle)
            p.y = p.y + distance * math.sin(angle)


    def unsignedDistanceTo(self, point):
        min_distance = float('inf')
        for p in self.points:
            tmp_distance = p.distanceTo(point)
            if tmp_distance < min_distance:
                min_distance = tmp_distance
        for p in self.back_points:
            tmp_distance = p.distanceTo(point)
            if tmp_distance < min_distance:
                min_distance = tmp_distance
        return min_distance
    
    def distanceAndNeighbourPointsTo(self, point):
        min_distance = float('inf')
        last_point = None

        for i in range(len(self.back_points)):
            tmp_distance = self.back_points[i].distanceTo(point)
            if tmp_distance < min_distance:
                min_distance = tmp_distance
                nearest_point = self.back_points[i]
                nearest_last_point = last_point
                if i == len(self.back_points)-1:
                    nearest_next_point = self.points[0]
                else:
                    nearest_next_point = self.back_points[i+1]
            last_point = self.back_points[i]

        for i in range(len(self.points)):
            tmp_distance = self.points[i].distanceTo(point)
            if tmp_distance < min_distance:
                min_distance = tmp_distance
                nearest_point = self.points[i]
                nearest_last_point = last_point
                if i == len(self.points)-1:
                    nearest_next_point = None
                else:
                    nearest_next_point = self.points[i+1]
            last_point = self.points[i]
        return (min_distance, [nearest_last_point, nearest_point, nearest_next_point])