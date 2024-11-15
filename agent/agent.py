import numpy as np
import hyper_param as hp
import torch
import math
import copy
import random
import pickle
from cpprb import ReplayBuffer
from safety.safe_shielding import FixedDistanceShielding
from common.print_color import printColor
from utils import observationAdapter, actionAdapter, actionAdapterLat, toOneHot
from for_exp.utils_for_exp import performanceToReward
# from utils import performanceToReward

if torch.cuda.is_available():
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

printColor("The using device is " + str(device), 'p')

class Agent: # contain common operations of replay buffer
    def __init__(self, epistemic_estimation=False, sampling_mode='NULL') -> None: 
        self.epistemic_estimation = epistemic_estimation
        if self.epistemic_estimation:
            self.integrated_model_num = hp.integrated_model_num
        else:
            self.integrated_model_num = 1

        self.sampling_mode = sampling_mode
        self.generateReplayBuffers()

        self.has_been_trained = False
        self.device = device

        self.last_eu_lat_by_label = {-1:0, 0:0, 1:0} 
        self.eu_lat_by_label = {-1:0, 0:0, 1:0} # for reference of the first sampling 
        self.epistemic_uncertainty_lon = 0
        self.epistemic_uncertainty_lat = 0
        self.epistemic_uncertainty = 0
        self.policy_uncertainty_lat = 0
        
        self.total_it = 0 # for train

        # 打开文件
        self.data_features = np.load(hp.data_feature_file, allow_pickle=True).item()
        self.state_mean = torch.FloatTensor(self.data_features['feature_mean']).to(device)
        self.state_var = torch.FloatTensor(self.data_features['feature_var']).to(device)
        for i in range(self.state_var.shape[0]):
            for j in range(self.state_var.shape[1]):
                if self.state_var[i,j] == 0.0:
                    self.state_var[i,j] = 1
                    self.data_features['feature_var'][i,j] = 1

        states_mean_list = []
        states_var_list = []
        for i in range(hp.batch_size):
            states_mean_list.append(self.state_mean)
            states_var_list.append(self.state_var)
        self.states_mean = torch.stack(states_mean_list)
        self.states_var = torch.stack(states_var_list)

        self.actor = None
        self.critic = None
        self.shielding = FixedDistanceShielding()

    def act(self, obs, is_with_exploration=False, safe_lane_num=0):
        # self.last_state = self.state
        self.state = observationAdapter(self.state, obs)
        
        # normalize the state as input
        normalized_state = torch.FloatTensor(self.state).to(self.device)    
        for j in range(hp.history_frame_num): # process every frame
            normalized_state[:,:,j] = normalized_state[:,:,j] - self.state_mean
            normalized_state[:,:,j] = torch.div(normalized_state[:,:,j], self.state_var)

        self.raw_action = self._act(normalized_state) # velocity and lane_change (one hot)
        
        
        # (velocity, lane_change) = actionAdapter(self.raw_action, obs.ego_vehicle_state, is_deterministic=True)

        lane_change_decision = self.raw_action[1].cpu().detach().numpy().squeeze()
        self.lat_action_one_hot = lane_change_decision
        for tmp_lane_change in [-1, 1]:
            target_lane_index = obs.ego_vehicle_state.lane_index + tmp_lane_change
            if target_lane_index < (0+safe_lane_num) or target_lane_index >= (len(obs.waypoint_paths)-safe_lane_num):
                lane_change_decision[tmp_lane_change+1] = 0
        (velocity, lane_change) = (self.raw_action[0].item(), actionAdapterLat(lane_change_decision, is_deterministic=not is_with_exploration))

        # safety shielding section
        self.correct_lane_change = self.shielding.safeLatAction(lane_change, self.state) # 进行侧向的安全判断
        self.correct_velocity = self.shielding.safeLonAction(velocity, self.state)
        if self.shielding.state.is_lat_active:
            printColor("self.shielding.is_lat_active", 'y')
        if self.shielding.state.is_lon_active:
            printColor("self.shielding.is_lon_active", 'r')

        # # output section
        # if is_with_exploration: # exploration process
        #     # lon_random_num = random.random()
        #     lat_random_num = random.random()
        #     self.epsilon = 0.1
        #     # if lon_random_num < self.epsilon:
        #     #     velocity = random.random() * 20 # 0~20
        #     if lat_random_num < self.epsilon:
        #         lane_change = random.randint(-1,1)
        # self.action = (correct_velocity, correct_lane_change)
        # self.action = (velocity, lane_change)
        self.action = (velocity, lane_change)
        return self.action

    def _act(self, 
             state, 
             execute_dropout=False): # for not estimation every time
        # 将数据维度进行变换
        if len(state.shape) == 4: # batch
            state = state.permute(0,3,1,2)
        else:
            state = state.permute(2,0,1)
        
        lon_out, lane_change_cmd = self.actor.forward(state, execute_dropout=execute_dropout)
        lon_out = lon_out.unsqueeze(1)
        lat_out = lane_change_cmd.unsqueeze(1)
        return (lon_out, lat_out)
    
    def generateReplayBuffers(self):
        # replay buffer
        buffer_structure = {"state": {"shape": hp.state_shape},
                            "lon_action": {},
                            "lat_action": {"shape": (3) },
                            "next_state": {"shape": hp.state_shape},
                            "reward": {},
                            "not_done": {},
                            }
        if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
            self.replay_buffer_dict = {1:ReplayBuffer(hp.buffer_size, buffer_structure), # left lane change
                                       0:ReplayBuffer(hp.buffer_size, buffer_structure), # keep lane
                                      -1:ReplayBuffer(hp.buffer_size, buffer_structure),} # right lane change
        self.replay_buffer = ReplayBuffer(hp.buffer_size, buffer_structure)
        # dagger replay buffer
        buffer_structure = {"state": {"shape": hp.state_shape},
                            "lon_action": {},
                            "lat_action": {"shape": (3) },
                            "next_state": {"shape": hp.state_shape},
                            "reward": {},
                            "expert_lon_action": {}, # for dagger
                            "expert_lat_action": {"shape": (3) }, # for dagger
                            "not_done": {},
                            }
        if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
            self.dagger_replay_buffer_dict = {1:ReplayBuffer(hp.buffer_size, buffer_structure),# left lane change
                                              0:ReplayBuffer(hp.buffer_size, buffer_structure), # keep lane
                                             -1:ReplayBuffer(hp.buffer_size, buffer_structure),}# right lane change
        self.dagger_replay_buffer = ReplayBuffer(hp.buffer_size, buffer_structure)

    def getBufferStoredSize(self, buffer_name='Normal'):
        if buffer_name == 'DAGGER':
            return self.dagger_replay_buffer.get_stored_size()
        return self.replay_buffer.get_stored_size()
        
    def addToReplayBuffer(self, state=None, lon_action=None, lat_action=None, next_state=None, reward=None, not_done=1):
        if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
            self.replay_buffer_dict[np.argmax(lat_action)-1].add(state=state, lon_action=lon_action, lat_action=lat_action, next_state=next_state, reward=reward, not_done=not_done)
        self.replay_buffer.add(state=state, 
                               lon_action=lon_action, 
                               lat_action=lat_action, 
                               next_state=next_state, 
                               reward=reward, not_done=not_done)
            
    def addToDaggerReplayBuffer(self, dagger_frames):        
        for (state, action, reward, next_state, expert_action, not_done) in dagger_frames:
            lon_action = action[0]
            if isinstance(action[1], list) or isinstance(action[1], np.ndarray):
                lat_action = action[1]
            else:
                printColor(type(action[1]))
                lat_action = toOneHot(action[1])

            expert_lon_action = expert_action[0]
            if isinstance(expert_action[1], list) or isinstance(expert_action[1], np.ndarray):
                expert_lat_action = expert_action[1]
            else:
                expert_lat_action = toOneHot(expert_action[1])

            if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
                self.dagger_replay_buffer_dict[np.argmax(lat_action)-1].add(state=state, 
                                                                            lon_action=lon_action, lat_action=lat_action, 
                                                                            next_state=next_state, reward=reward,
                                                                            expert_lat_action=expert_lat_action, expert_lon_action=expert_lon_action,
                                                                            not_done=not_done)
            self.dagger_replay_buffer.add(state=state, 
                                            lon_action=lon_action, lat_action=lat_action, 
                                            next_state=next_state, reward=reward,
                                            expert_lat_action=expert_lat_action, expert_lon_action=expert_lon_action,
                                            not_done=not_done)

        # start_time = time.time()
        # end_time = time.time()
        # printColor("time check is " + str(time.time() - start_time))

    def normalSampling(self, batch_size, is_from_dagger=False):
        if is_from_dagger:
            replay_buffer = self.dagger_replay_buffer
        else:
            replay_buffer = self.replay_buffer

        data = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(data['state']).to(device)
        actions = (torch.FloatTensor(data['lon_action']).to(device), 
                   torch.FloatTensor(data['lat_action']).to(device))
        next_states = torch.FloatTensor(data['next_state']).to(device)
        rewards = torch.FloatTensor(data['reward']).to(device)
        not_dones = torch.FloatTensor(data['not_done']).to(device)

        if is_from_dagger:
            expert_actions = (torch.FloatTensor(data['expert_lon_action']).to(device), torch.FloatTensor(data['expert_lat_action']).to(device))
        else:
            expert_actions = copy.deepcopy(actions)
            
        return states, actions, rewards, next_states, expert_actions, not_dones
    
    def customizedSampling(self, batch_size, sampling_ratios, is_from_dagger=False):
        if is_from_dagger:
            replay_buffer = self.dagger_replay_buffer
            replay_buffer_dict = self.dagger_replay_buffer_dict
        else:
            replay_buffer = self.replay_buffer
            replay_buffer_dict = self.replay_buffer_dict

        states_list = []
        lon_actions_list = []
        lat_actions_list = []
        rewards_list = []
        next_states_list = []
        expert_lon_actions_list = []
        expert_lat_actions_list = []
        not_dones_list = []

        total_sampling_size = 0
        for key in replay_buffer_dict.keys():
            sampling_num = int(batch_size * sampling_ratios[key])

            data = replay_buffer_dict[key].sample(sampling_num)
            states_list.append(torch.FloatTensor(data['state']).to(device))
            lon_actions_list.append(torch.FloatTensor(data['lon_action']).to(device))
            lat_actions_list.append(torch.FloatTensor(data['lat_action']).to(device))
            rewards_list.append(torch.FloatTensor(data['reward']).to(device))
            next_states_list.append(torch.FloatTensor(data['next_state']).to(device))
            not_dones_list.append(torch.FloatTensor(data['not_done']).to(device))
            if is_from_dagger:
                expert_lon_actions_list.append(torch.FloatTensor(data['expert_lon_action']).to(device))
                expert_lat_actions_list.append(torch.FloatTensor(data['expert_lat_action']).to(device))
            total_sampling_size += sampling_num

        if batch_size - total_sampling_size > 0: # sampling from the normal buffer to complete batch size 
            data = replay_buffer.sample(batch_size - total_sampling_size)
            states_list.append(torch.FloatTensor(data['state']).to(device))
            lon_actions_list.append(torch.FloatTensor(data['lon_action']).to(device))
            lat_actions_list.append(torch.FloatTensor(data['lat_action']).to(device))
            rewards_list.append(torch.FloatTensor(data['reward']).to(device))
            next_states_list.append(torch.FloatTensor(data['next_state']).to(device))
            not_dones_list.append(torch.FloatTensor(data['not_done']).to(device))
            if is_from_dagger:
                expert_lon_actions_list.append(torch.FloatTensor(data['expert_lon_action']).to(device))
                expert_lat_actions_list.append(torch.FloatTensor(data['expert_lat_action']).to(device))

        states = torch.cat(states_list, dim=0)
        actions = (torch.cat(lon_actions_list, dim=0), torch.cat(lat_actions_list, dim=0))
        rewards = torch.cat(rewards_list, dim=0)
        next_states = torch.cat(next_states_list, dim=0)
        not_dones = torch.cat(not_dones_list, dim=0)

        if is_from_dagger:
            expert_actions = (torch.cat(expert_lon_actions_list, dim=0), torch.cat(expert_lat_actions_list, dim=0))
        else:
            expert_actions = copy.deepcopy(actions)

        return states, actions, rewards, next_states, expert_actions, not_dones

    
    # def samplingProcessIL(self, DAGGER):
        # is_dagger_sampling = False
        # if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
        #     # 确定采样比例
        #     if self.sampling_mode == 'OVER':
        #         sampling_ratios = {-1:0.333, 0:0.333, 1:0.333}
        #     if self.sampling_mode == 'EPA':
        #         tau = 5 # 调节温度系数，越大越均匀
        #         sampling_ratios = softmaxForDict(self.eu_lat_by_label, tau=tau)
        #     if not DAGGER or self.getBufferStoredSize(buffer_name='DAGGER') < hp.batch_size * hp.dagger_ratio:
        #         states, demonstrative_actions, _, _ = self.customizedSampling(hp.batch_size, 
        #                                                   sampling_ratios)
        #     else:
        #         dagger_batch_size = int(hp.batch_size * hp.dagger_ratio)
        #         states, demonstrative_actions, _, _ = self.customizedSampling(hp.batch_size - dagger_batch_size, 
        #                                                         sampling_ratios, 
        #                                                         is_from_dagger=False)
        #         dagger_states, _, _, _, dagger_demonstrative_actions = self.customizedSampling(dagger_batch_size, 
        #                                                                       sampling_ratios, 
        #                                                                       is_from_dagger=True)
        #         is_dagger_sampling = True
        # else:
        #     if not DAGGER or self.getBufferStoredSize(buffer_name='DAGGER') < hp.batch_size * hp.dagger_ratio:
        #         states, demonstrative_actions, _, _ = self.normalSampling(hp.batch_size)
        #     else:
        #         dagger_batch_size = int(hp.batch_size * hp.dagger_ratio)
        #         states, demonstrative_actions, _, _ = self.normalSampling(hp.batch_size - dagger_batch_size, is_from_dagger=False)
        #         dagger_states, _, _, _, dagger_demonstrative_actions = self.normalSampling(dagger_batch_size, is_from_dagger=True)
        #         is_dagger_sampling = True

        # if is_dagger_sampling:
        #     states = torch.cat([states, dagger_states], dim=0)
        #     demonstrative_actions = (torch.cat([demonstrative_actions[0], dagger_demonstrative_actions[0]], dim=0), 
        #                              torch.cat([demonstrative_actions[1], dagger_demonstrative_actions[1]], dim=0))

        # return states, demonstrative_actions
    
    def samplingProcessRL(self, DAGGER):
        is_dagger_sampling = False
        if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
            # 确定采样比例
            if self.sampling_mode == 'OVER':
                sampling_ratios = {-1:0.25, 0:0.5, 1:0.25}
            if self.sampling_mode == 'EPA':
                tau = 5 # 调节温度系数，越大越均匀
                sampling_ratios = softmaxForDict(self.eu_lat_by_label, tau=tau)
            if not DAGGER or self.getBufferStoredSize(buffer_name='DAGGER') < hp.batch_size * hp.dagger_ratio:
                states, actions, rewards, next_states, demonstrative_actions, not_dones = self.customizedSampling(hp.batch_size, sampling_ratios)
            else:
                dagger_batch_size = int(hp.batch_size * hp.dagger_ratio)
                states, actions, rewards, next_states, demonstrative_actions, not_dones = self.customizedSampling(hp.batch_size - dagger_batch_size, 
                                                                sampling_ratios, 
                                                                is_from_dagger=False)
                dagger_states, dagger_actions, dagger_rewards, dagger_next_states, dagger_demonstrative_actions, dagger_not_dones = self.customizedSampling(dagger_batch_size, 
                                                                              sampling_ratios, 
                                                                              is_from_dagger=True)
                is_dagger_sampling = True
        else:
            if not DAGGER or self.getBufferStoredSize(buffer_name='DAGGER') < hp.batch_size * hp.dagger_ratio:
                states, actions, rewards, next_states, demonstrative_actions, not_dones = self.normalSampling(hp.batch_size)
            else:
                dagger_batch_size = int(hp.batch_size * hp.dagger_ratio)
                states, actions, rewards, next_states, demonstrative_actions, not_dones = self.normalSampling(hp.batch_size - dagger_batch_size, is_from_dagger=False)
                dagger_states, dagger_actions, dagger_rewards, dagger_next_states, dagger_demonstrative_actions, dagger_not_dones = self.normalSampling(dagger_batch_size, is_from_dagger=True)
                is_dagger_sampling = True

        if is_dagger_sampling:
            states = torch.cat([states, dagger_states], dim=0)
            actions = (torch.cat([actions[0], dagger_actions[0]], dim=0), 
                       torch.cat([actions[1], dagger_actions[1]], dim=0))
            rewards = torch.cat([rewards, dagger_rewards], dim=0)
            next_states = torch.cat([next_states, dagger_next_states], dim=0)
            not_dones = torch.cat([not_dones, dagger_not_dones], dim=0)
            demonstrative_actions = (torch.cat([demonstrative_actions[0], dagger_demonstrative_actions[0]], dim=0), 
                                     torch.cat([demonstrative_actions[1], dagger_demonstrative_actions[1]], dim=0))

        return states, actions, rewards, next_states, demonstrative_actions, not_dones

    def extractDataFeature(self, actions):
        self.lat_label_num = {-1:0, 0:0, 1:0}
        for i in range(len(actions[1])):
            index = torch.argmax(actions[1][i]).item()
            self.lat_label_num[index - 1] += 1

    def epistemicEstimate(self, state, action):
        # 将数据维度进行变换
        if len(state.shape) == 4: # batch
            state = state.permute(0,3,1,2)
        else:
            state = state.permute(2,0,1)
        lon_out_list = []
        lat_out_list = []
        with torch.no_grad(): # 测试表示可以节省一半的时间
            for i in range(self.integrated_model_num):
                lon_out, lane_change_cmd = self.actor.forward(state, execute_dropout=True)
                lon_out_list.append(lon_out.unsqueeze(1))
                lat_out_list.append(lane_change_cmd.unsqueeze(1))
        
            lon_outs = torch.cat(lon_out_list, dim=1) #  100 10 1
            lat_outs = torch.cat(lat_out_list, dim=1) #  100 10 3

            # longitudinal epistemic uncertainty
            lon_var_sum = 0
            for i in range(hp.batch_size):
                lon_var_sum += torch.var(lon_outs[i][:][:])
            lon_var_average = lon_var_sum / hp.batch_size
            self.epistemic_uncertainty_lon = torch.exp(-lon_var_average) #0->1

            # lateral policy uncertainty
            lat_policy_var_sum = 0 
            for i in range(hp.batch_size):
                for j in range(self.integrated_model_num):
                    lat_policy_var_sum += torch.var(lat_outs[i,j,:])
            self.policy_uncertainty_lat = lat_policy_var_sum / (hp.batch_size * self.integrated_model_num) / 4 * 9

            # lateral epistemic uncertainty
            lat_var_sum = 0 
            self.eu_lat_by_label = {-1:0, 0:0, 1:0}
            base_num = self.integrated_model_num / 3
            for i in range(hp.batch_size):
                action_num = {-1:0, 0:0, 1:0} # num for each action
                for j in range(self.integrated_model_num):
                    action_num[torch.argmax(lat_outs[i,j,:]).item()-1] += 1

                action_label = torch.argmax(action[1][i]).item()-1
                self.eu_lat_by_label 
                lat_var_sum += max(action_num)

                self.eu_lat_by_label[action_label] += (action_num[action_label]-base_num)/ (self.integrated_model_num-base_num)
                
            average_action_num = lat_var_sum / hp.batch_size
            self.epistemic_uncertainty_lat = (average_action_num-base_num) / (self.integrated_model_num-base_num) # self.policy_uncertainty_lat
            self.epistemic_uncertainty = (self.epistemic_uncertainty_lon**2 + self.epistemic_uncertainty_lat**2)**0.5 / 2**0.5

            for k in range(-1, 2): # action -1 0 1
                if self.lat_label_num[k] == 0:
                    self.eu_lat_by_label[k] = self.last_eu_lat_by_label[k]
                else:
                    self.eu_lat_by_label[k] = self.eu_lat_by_label[k] / self.lat_label_num[k]

        self.last_eu_lat_by_label = self.eu_lat_by_label
        # self.has_estimated_eu = True
    
    # def getBasicTrainInfo(self, loss):
    #     train_info = {'loss': loss.item(),
    #                   'replay_buffer_size': self.replay_buffer.get_stored_size(),
    #                   'dagger_replay_buffer_size': self.dagger_replay_buffer.get_stored_size()}
    #     return train_info

    def getTrainInfo(self, actions, policy_actions, loss):
        train_info = {'loss': loss.item(),
                      'replay_buffer_size': self.replay_buffer.get_stored_size(),
                      'dagger_replay_buffer_size': self.dagger_replay_buffer.get_stored_size()}
        

        if policy_actions != None:        
            total_num = len(actions[0])
            self.extractDataFeature(actions)
            # calculate accuracy
            lat_label_accuracy_num = {-1:0, 0:0, 1:0} # calculate accuracy by label
            accuracy_num = 0; accuracy_num_lat = 0 ;accuracy_num_lon = 0
            for i in range(total_num):
                if torch.argmax(actions[1][i]) == torch.argmax(policy_actions[1][i]):
                    accuracy_num_lat += 1
                    lat_action_index = torch.argmax(actions[1][i]).item()
                    lat_label_accuracy_num[lat_action_index-1] += 1
                    if abs(actions[0][i] - policy_actions[0][i]) < hp.lon_accuracy_threshold:
                        accuracy_num_lon += 1
                        accuracy_num += 1
                else: 
                    if abs(actions[0][i] - policy_actions[0][i]) < hp.lon_accuracy_threshold:
                        accuracy_num_lon += 1

            accuracy_lat_by_label = {}
            for key in self.lat_label_num.keys():
                if self.lat_label_num[key] == 0: # 如果没有该样本的数据
                    accuracy_lat_by_label[str(key)] = -0.1
                else:
                    accuracy_lat_by_label[str(key)] = float(lat_label_accuracy_num[key] / self.lat_label_num[key])

            train_info['AC_lat'] = float(accuracy_num_lat / hp.batch_size)
            train_info['AC_lat_by_label'] = accuracy_lat_by_label
            train_info['AC_lon'] = float(accuracy_num_lon / hp.batch_size)
            train_info['AC'] = float(accuracy_num / hp.batch_size)
        
        if self.epistemic_estimation:
            train_info['EU_lon'] = self.epistemic_uncertainty_lon
            train_info['EU_lat'] = self.epistemic_uncertainty_lat
            train_info['EU'] = self.epistemic_uncertainty
            train_info['EU_lat_by_label'] = toStrKeyDict(self.eu_lat_by_label)
            train_info['PU_lat'] = self.policy_uncertainty_lat
        # if self.sampling_mode == 'EPA':
        #     train_info['SamplingRatio'] = toStrKeyDict(epa_sampling_ratios)
        return train_info
    
    def saveModel(self, filename):
        if self.critic != None:
            torch.save(self.critic.state_dict(), filename + "/critic")
            # torch.save(self.critic_optimizer.state_dict(), filename + "/critic_optimizer")
            
        if self.actor != None:
            torch.save(self.actor.state_dict(), filename + "/actor")
            # torch.save(self.actor_optimizer.state_dict(), filename + "/actor_optimizer")

    def saveOptimizer(self, filename):
        if self.critic != None:
            torch.save(self.critic_optimizer.state_dict(), filename + "/critic_optimizer")
            
        if self.actor != None:
            torch.save(self.actor_optimizer.state_dict(), filename + "/actor_optimizer")

    def loadModel(self, filename):
        # if self.critic != None:
        #     self.critic.load_state_dict(torch.load(filename + "/critic"))
        #     self.critic_target = copy.deepcopy(self.critic)

        if self.actor != None:
            self.actor.load_state_dict(torch.load(filename + "/actor"))
            self.actor_target = copy.deepcopy(self.actor)

    def loadOptimizer(self, filename):
        if self.critic != None:
            self.critic_optimizer.load_state_dict(torch.load(filename + "/critic_optimizer"))

        if self.actor != None:
            self.actor_optimizer.load_state_dict(torch.load(filename + "/actor_optimizer"))

    def saveReplayBuffer(self, filename):
        # self.replay_buffer.save(filename + '/replay_buffer')
        # self.dagger_replay_buffer.save(filename + '/dagger_replay_buffer')
        replay_buffer_data = self.replay_buffer.sample(self.replay_buffer.get_stored_size())
        with open(filename + '/replay_buffer', 'wb') as f:
            pickle.dump(replay_buffer_data, f)

        dagger_replay_buffer_data = self.dagger_replay_buffer.sample(self.dagger_replay_buffer.get_stored_size())
        with open(filename + '/dagger_replay_buffer', 'wb') as f:
            pickle.dump(dagger_replay_buffer_data, f)

    # def loadReplayBuffer(self, filename):
    #     experiences = self.replay_buffer.sample(self.replay_buffer.current_size)
    #     for state, action, reward, next_state in experiences:
    #         if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
    #             self.replay_buffer_dict[np.argmax(action[1])-1].add(state=state, 
    #                 lon_action=lon_action, lat_action=lat_action, 
    #                 next_state=next_state, reward=reward,
    #                 expert_lat_action=expert_lat_action, expert_lon_action=expert_lon_action)
            
    def loadInitialOnlineData(self, folder_path):
        printColor('Load initial online data!', 'p')
        with open(folder_path + "/data.pkl", "rb") as file:
            data = pickle.load(file)

        with open(folder_path + "/performances.pkl", "rb") as file:
            performances = pickle.load(file)

        not_done = 1
        for i in range(data['state'].shape[0]):
            state = data['state'][i]
            lon_action = data['lon_action'][i]
            lat_action = data['lat_action'][i]
            if i == data['state'].shape[0] - 1:
                not_done = 0

            reward, _ = performanceToReward(performances[i])

            next_state = data['next_state'][i]
            expert_lon_action = data['expert_lon_action'][i]
            expert_lat_action = data['expert_lat_action'][i]

            if self.sampling_mode == 'OVER' or self.sampling_mode == 'EPA':
                self.dagger_replay_buffer_dict[np.argmax(lat_action)-1].add(state=state, 
                                                                            lon_action=lon_action, lat_action=lat_action, 
                                                                            next_state=next_state, reward=reward,
                                                                            expert_lat_action=expert_lat_action, expert_lon_action=expert_lon_action,
                                                                            not_done=not_done)
            self.dagger_replay_buffer.add(state=state, 
                                            lon_action=lon_action, lat_action=lat_action, 
                                            next_state=next_state, reward=reward,
                                            expert_lat_action=expert_lat_action, expert_lon_action=expert_lon_action,
                                            not_done=not_done)

    # def observeF1Score(self):


def softmaxForDict(x, tau=1):
    v_sum = 0
    res_dict = {}
    for k, v in x.items():
        res_dict[k] = math.exp((1-v)/tau)
        v_sum += res_dict[k]

    for key in res_dict.keys():
        res_dict[key] = res_dict[key] / v_sum
    return res_dict

def toStrKeyDict(a):
    return {str(k): v for k, v in a.items()}