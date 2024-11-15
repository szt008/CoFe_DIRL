#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gym
import sys
import time
import math
import yaml
import copy
import shutil
import torch
import argparse
import warnings
import statistics
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import count
from collections import deque
import matplotlib.pyplot as plt
from cpprb import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter   

from smarts.core.agent import AgentSpec
from smarts.core.utils.episodes import episodes
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints

from agent.BC import BC
from agent.TD3_BC import TD3BC
from agent.CoFeDIRL import CoFeDIRL
from agent.IDM_MOBIL import IDM_MOBIL
import hyper_param as hp
from common.counter import Counter
from common.print_color import printColor
from common.best_model_saver import BestModelSaver

from evaluate_multi_scenario import Evaluator

from utils import performanceToReward

DATA_FOLDER = './data/NGSIM_i80_0500-0515_202312161319'
TB_COMPARE_FOLDER = './data/compare/compare_26' # None 
# METHOD_LABEL = 'TD3BC_1DR_NULL_NoRL'
# METHOD_LABEL = 'BC_1DR_NULL_6'
# METHOD_LABEL = 'TD3BC_100DR_NULL_OnRL_SS_LR25_03'
METHOD_LABEL = 'CoFe_newhq_street'
AGENT_ID = "Elegance"
SUMO_GUI = False # False or True
ENVISION_GUI = False # False or True
LOAD_FOLDER = None # None
LOAD_FROM_DAGGER_START = True
INITIAL_ONLINE_DATA_PATH = './data/online_data_for_initial_policy/INITIAL_500' # None

def getAgrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--POLICY', default='CoFeDIRL') #BC TD3BC CoFeDIRL
    parser.add_argument('-db', '--DATA_BALANCING_MODE', default='NULL') # EPA==EP-Aware OVER==OverSampling NULL
    parser.add_argument('-s', '--SEED', type=int, default=101)
    parser.add_argument('-d', '--DAGGER', type=bool, default=True) #True or False
    parser.add_argument('-e', '--EPISTEMIC', type=bool, default=False)
    parser.add_argument('-oe', '--ONLINE_EVAL', type=bool, default=True)
    parser.add_argument('-um', '--UPDATE_MODE', default='ILRL') # IL or RL or ILRL

    parser.add_argument('-dil', '--DIL', type=bool, default=True) # True or False
    parser.add_argument('-drl', '--DRL', type=bool, default=True) 
    parser.add_argument('-iil', '--IIL', type=bool, default=True) 
    parser.add_argument('-irl', '--IRL', type=bool, default=True) 
    parser.add_argument('-cf', '--CoFe', type=bool, default=False) 
    return parser.parse_args()

class DataReader:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        self.traj_names = os.listdir(data_folder + '/' + 'trajectories')
        self.traj_num = len(self.traj_names)

    def loadFrame(self, frame_index): # 对前后帧的rgb图像进行叠加，一共三帧，能看到前0.3秒的内容
        state = np.zeros(shape=(21, 5,  hp.history_frame_num))
        next_state = np.zeros(shape=(21, 5,  hp.history_frame_num))
        
        for i in range(hp.history_frame_num):
            index = frame_index - hp.history_frame_num + 1 + i
            
            if index >= 0:
                state[:, :, i:i+1] = self.stateAt(index)
            next_index = index + 1
            if next_index >= 0 and next_index < self.frame_num:
                next_state[:, :, i:i+1] = self.stateAt(next_index)

        action = self.trajectory['action'][frame_index]
        performance = self.trajectory['performance'][frame_index]
        performance['off_lane'] = False
        return state, action, performance, next_state
    
    def stateAt(self, index):
        current_state = np.zeros(shape=(21, 5, 1))
        current_state[0, :, 0] = self.trajectory['state'][index]['ego_traffic_feature']
        current_state[1:21, :, 0] = self.trajectory['state'][index]['traffic_feature_g']
        return current_state
    
    def actionAt(self, index):
        printColor(self.trajectory['action'][index])
    
    def loadNewTrajectory(self, traj_name):
        self.trajectory = np.load(self.data_folder + '/trajectories/' + traj_name, 
                                  allow_pickle=True).item()
        self.frame_num = len(self.trajectory['state'])

class DataSaver:
    def __init__(self, data_name) -> None:
        self.data_name = data_name
        self.fresh()

        t = time.gmtime()
        self.save_dir =  './data/' + str(t.tm_year).zfill(4) + str(t.tm_mon).zfill(2) + str(t.tm_mday).zfill(2) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + '_' + self.data_name 
        if os.path.exists(self.save_dir):
            os.system("rm -rf {}".format(self.save_dir))
        os.makedirs(self.save_dir)
        os.makedirs(self.save_dir + '/' + 'tb_log')
        os.makedirs(self.save_dir + '/' + 'model')
        os.makedirs(self.save_dir + '/' + 'replay_buffer')

        shutil.copyfile("./utils.py", self.save_dir + "/utils.py")
        shutil.copyfile("./hyper_param.py", self.save_dir + "/hyper_param.py")

        self.tensor_board = SummaryWriter(self.save_dir + '/' + 'tb_log')

        self.frame_counter = Counter()
        self.epoc_counter = Counter()

        self.distribution_variables = {'confict_DI':[], 'confict_RI':[]}


    def collect(self, train_info):
        current_frame_index = self.frame_counter.count()
        for key in train_info.keys():
            if isinstance(train_info[key], dict):
                self.tensor_board.add_scalars(key, train_info[key], current_frame_index)
            else:
                self.tensor_board.add_scalar(key, train_info[key], current_frame_index)
        self.data["loss"].append(train_info['loss'])
        self.total_loss += train_info['loss']

        # 查看分布的变量
        if not current_frame_index % 1000 == 0:
            self.distribution_variables['confict_DI'].append(train_info['conflict_intensity']['demonstration_vs_interaction'])
            self.distribution_variables['confict_RI'].append(train_info['conflict_intensity']['reinforcement_vs_imitation'])
        else:
            for name in self.distribution_variables:
                self.tensor_board.add_histogram(name, torch.tensor(self.distribution_variables[name]), self.epoc_counter.value())
                self.distribution_variables[name] = []
        
    def fresh(self):
        self.total_loss = 0
        self.data = {"loss":[]}

    def endEpoc(self):
        print('\n|Epoc:', self.epoc_counter.count(), '\n|R:', self.total_loss)
        self.tensor_board.add_scalar('total_loss', self.total_loss, self.epoc_counter.value())
        self.fresh()

    def closeTensorBoard(self):
        self.tensor_board.close()

def main(agrs):
    evaluator = Evaluator(AGENT_ID, agrs.SEED, SUMO_GUI, ENVISION_GUI)

    if agrs.POLICY == "BC":
        policy = BC(epistemic_estimation=agrs.EPISTEMIC, sampling_mode=agrs.DATA_BALANCING_MODE)
    elif agrs.POLICY == "TD3BC":
        policy = TD3BC(epistemic_estimation=agrs.EPISTEMIC, sampling_mode=agrs.DATA_BALANCING_MODE, update_mode=agrs.UPDATE_MODE)
    elif agrs.POLICY == "CoFeDIRL":
        combined_update_mode = {'demonstration':{'imitation':agrs.DIL, 'reinforcement':agrs.DRL},
                                'interaction':{'imitation':agrs.IIL, 'reinforcement':agrs.IRL},
                                }
        policy = CoFeDIRL(epistemic_estimation=agrs.EPISTEMIC, sampling_mode=agrs.DATA_BALANCING_MODE, combined_update_mode=combined_update_mode, conflict_free=agrs.CoFe)
        
    if not INITIAL_ONLINE_DATA_PATH == None:
        policy.loadInitialOnlineData(INITIAL_ONLINE_DATA_PATH)
        
    if not LOAD_FOLDER == None:
        if LOAD_FROM_DAGGER_START:
            policy.loadModel(LOAD_FOLDER + '/dagger_start_model')
            policy.loadReplayBuffer(LOAD_FOLDER + '/dagger_start_replay_buffer')
            load_start_epoc = hp.dagger_start_epoc

    expert_policy = IDM_MOBIL(target_v=20, dt=0.1) if agrs.DAGGER else None
    
    data_saver = DataSaver(METHOD_LABEL)
    # data_saver.tensor_board.add_graph(policy.net, input_to_model=torch.FloatTensor(np.zeros(shape=(21, 5, hp.history_frame_num))).permute(2,0,1).cuda())
    data_reader = DataReader(DATA_FOLDER)
    best_model_saver = BestModelSaver(10, data_saver.save_dir)
    
    lateral_action_num = [0, 0, 0]
    for epoc in tqdm(range(hp.max_num_epoc), ascii=True):
        if not LOAD_FOLDER == None and epoc <= load_start_epoc:
            continue
        data_saver.fresh()
        data_reader.loadNewTrajectory(data_reader.traj_names[epoc % data_reader.traj_num])
        not_done = 1
        for frame_index in range(data_reader.frame_num):
            state, action, performance, next_state = data_reader.loadFrame(frame_index)
            reward, reward_components = performanceToReward(performance)
            if frame_index == data_reader.frame_num - 1:
                not_done = 0

            # lon_action = action['acceleration']
            lon_action = action['velocity']
            lat_action = action['lane_change']

            data_times = 1
            lat_action_index = np.argmax(lat_action)
            
            for i in range(data_times):
                policy.addToReplayBuffer(state=state, 
                                         lon_action=lon_action, 
                                         lat_action=lat_action, 
                                         next_state=next_state, 
                                         reward=reward,
                                         not_done=not_done)
            lateral_action_num[lat_action_index] += data_times # for print
            
            # 以上为数据进入replay buffer的过程，后续为进行训练的过程
        if  policy.getBufferStoredSize() < hp.buffer_size * 0.8 or epoc < 50:
            continue # current replay buffer is too small
        for _ in range(hp.train_time_per_epoc):
            train_info = policy.train(epoc, DAGGER=(agrs.DAGGER and epoc > hp.dagger_start_epoc))
            data_saver.collect(train_info)

        data_saver.endEpoc()
            
        if policy.has_been_trained and epoc > hp.evaluation_start_epoc and epoc % hp.evaluation_interval == 0 and (agrs.ONLINE_EVAL or agrs.DAGGER):
            eval_info = evaluator.evaluate(policy, hp.evaluation_epoc, is_with_exploration=True, expert_policy=expert_policy, evaluation_epoc=epoc)
            data_saver.tensor_board.add_scalar('eval_mean_reward', eval_info['reward'], epoc)
            data_saver.tensor_board.add_scalar('eval_success_rate', eval_info['success_rate'], epoc)
            data_saver.tensor_board.add_scalars('eval_success_rate_by_label', eval_info['success_rate_by_label'], epoc)
            data_saver.tensor_board.add_scalars('eval_reward_components', eval_info['mean_reward_components'], epoc)
            data_saver.tensor_board.add_scalar('eval_safe_shielding_rate', eval_info['safe_shielding'], epoc)
            best_model_saver.process(epoc, policy, eval_info['success_rate'])
        
        if policy.has_been_trained and epoc > hp.evaluation_start_epoc and epoc % hp.static_evaluation_interval == 0:
            static_eval_info = evaluator.evaluateStatic(policy)
            data_saver.tensor_board.add_scalar('static_eval_acc', static_eval_info['AC'], epoc)
            data_saver.tensor_board.add_scalars('static_eval_acc_by_label', static_eval_info['AC_lat_by_label'], epoc)

        if policy.has_been_trained and epoc % hp.save_interval  == 0 and epoc >= hp.save_start_epoc:
            if not TB_COMPARE_FOLDER == None:
                if not os.path.exists(TB_COMPARE_FOLDER):
                    os.makedirs(TB_COMPARE_FOLDER) 
                if os.path.exists(TB_COMPARE_FOLDER + '/' + METHOD_LABEL):
                    shutil.rmtree(TB_COMPARE_FOLDER + '/' + METHOD_LABEL)
                shutil.copytree(data_saver.save_dir + '/' + 'tb_log', TB_COMPARE_FOLDER + '/' + METHOD_LABEL)
                # os.rename(TB_COMPARE_FOLDER+'tb_log', TB_COMPARE_FOLDER+METHOD_LABEL)
            
            if epoc == hp.dagger_start_epoc: # 需要确保 hp.dagger_start_epoc % hp.save_interval == 0
                os.makedirs(data_saver.save_dir + '/' + 'dagger_start_model')
                os.makedirs(data_saver.save_dir + '/' + 'dagger_start_replay_buffer')
                policy.saveModel(data_saver.save_dir + '/' + 'dagger_start_model')
                policy.saveReplayBuffer(data_saver.save_dir + '/' + 'dagger_start_replay_buffer')
            else:
                policy.saveModel(data_saver.save_dir + '/' + 'model')
                policy.saveReplayBuffer(data_saver.save_dir + '/' + 'replay_buffer')

    data_saver.closeTensorBoard()
    print('Complete')

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)    
    agrs = getAgrs()
    main(agrs)