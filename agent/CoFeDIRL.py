import time
import math
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import hyper_param as hp
from cpprb import ReplayBuffer
from common.print_color import printColor
from agent.agent import Agent, softmaxForDict, toStrKeyDict
from agent.net.actor_net import ActorNet
from agent.net.critic_net import CriticNet
from torch.nn.utils import clip_grad_norm_

def getAnealingCoefficient(epoc):
    if epoc <= hp.gradient_projection_anealing_start_epoc:
        k_anealing = 0.0
    elif epoc < hp.gradient_projection_anealing_end_epoc:
        k_anealing = float((epoc - hp.gradient_projection_anealing_start_epoc) / \
            (hp.gradient_projection_anealing_end_epoc - hp.gradient_projection_anealing_start_epoc))
    else:
        k_anealing = 1.0
    return k_anealing

def getGradProjection(g1, g2, k_anealing=1):
    # get the projection of gradient1 on the direction of gradient2
    gradient_dot = torch.dot(g1.view(-1), g2.view(-1))
    # g1_norm = torch.norm(g1)
    g2_norm = torch.norm(g2)

    if gradient_dot < 0:
        g1_projection = (gradient_dot / g2_norm**2) * g2
        g1_projection_normal_direction = g1 - g1_projection
        return g1_projection_normal_direction * k_anealing + g1 * (1 - k_anealing)
    else:
        return g1

class CoFeDIRL(Agent):
    def __init__(self, max_velocity=20, load_file=None, epistemic_estimation=False, sampling_mode='NULL', combined_update_mode={}, conflict_free=False) -> None:
        super().__init__(epistemic_estimation=epistemic_estimation, sampling_mode=sampling_mode) # 允许子类法访问父类
        self.name = 'CoFeDIRL'
        self.actor = ActorNet(self.device, max_velocity=max_velocity)

        self.loss_function = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.learning_rate, weight_decay=0.001)

        self.critic = CriticNet(self.device, max_velocity=max_velocity)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.learning_rate, weight_decay=0.001)

        self.actor_target = copy.deepcopy(self.actor) # 用于产生next_action，根据actor进行延迟更新
        self.critic_target = copy.deepcopy(self.critic) # 用于产生target q，根据critic进行延迟更新

        self.state = np.zeros(shape=(21, 5, hp.history_frame_num)) # initial state
        
        if load_file != None:
            self.loadModel(load_file)

        self.discount = 0.99
        self.policy_freq = 2
        self.alpha = 2.5
        self.tau = 0.005

        self.combined_update_mode = combined_update_mode
        self.actor_gradients = {'demonstration':{'imitation':{},'reinforcement':{}}, 'interaction':{'imitation':{},'reinforcement':{}}} # 检测冲突
        self.critic_gradients = {'demonstration':{}, 'interaction':{}} # 检测冲突
        self.is_initialized = {}
        for data_source in combined_update_mode:
            self.is_initialized[data_source] = {}
            for update_method in combined_update_mode[data_source]:
                self.is_initialized[data_source][update_method] = 0

        self.conflict_free = conflict_free
                
        # self.existing_update_mode_labels = []
        # for data_source in combined_update_mode:
        #     for update_method in combined_update_mode[data_source]:
        #         if combined_update_mode[data_source][update_method]:
        #             self.existing_update_mode_labels.append(data_source + '_' + update_method)
        # for data_source in combined_update_mode:
        #     for update_method in combined_update_mode[data_source]:
        #         for name, param in self.actor.named_parameters():
        #                 if param.requires_grad:
        #                     self.actor_gradients[data_source][update_method][name] = torch.Tensor(0).to(self.device)

    def train(self, epoc, DAGGER=False):
        self.has_been_trained = True
        if epoc < hp.dagger_start_epoc: # to test with only rl // start trainning from 51th epoc 
            return {'loss': 0,
                    'AC_lat': 0,
                    'AC_lat_by_label': 0,
                    'AC_lon': 0,
                    'AC': 0,
                    'replay_buffer_size':0,
                    'dagger_replay_buffer_size':0,
                    'EU_lon':0,
                    'EU_lat':0,
                    'EU':0,
                    'EU_lat_by_label':0,
                    'PU_lat':0,}
        # printColor('start trainning', 'g')
        self.total_it += 1 

        # Sampling Section
        # demonstrative_states, demonstrative_actions = self.samplingProcessIL(DAGGER)
        # states, actions, rewards, next_states, demonstrative_actions, not_dones = self.samplingProcessRL(DAGGER)
        data_sources = ['interaction','demonstration']
        actor_loss = {'demonstration':{}, 'interaction':{}}
        critic_loss = {'demonstration':{}, 'interaction':{}}
        
        dagger_batch_size = int(hp.batch_size * hp.dagger_ratio)
        demonstrative_data_info = {'batch_size':hp.batch_size-dagger_batch_size, 
                                   'is_from_dagger':False, }
        interactive_data_info = {'batch_size':dagger_batch_size, 
                                 'is_from_dagger':True, }
        data_infos = {'interaction':interactive_data_info, 
                      'demonstration':demonstrative_data_info}
        demonstrative_actions_list = [] # for get train info
        policy_actions_list = [] # for get train info

        is_actor_update_epoc = (self.total_it % self.policy_freq == 0)
        k_anealing = getAnealingCoefficient(epoc) # for gradient projection

        for data_source in data_sources:
            # states, actions, rewards, next_states, demonstrative_actions, not_dones = self.normalSampling(data_infos[data_source]['batch_size'], is_from_dagger=data_infos[data_source]['is_from_dagger'])
            states, actions, rewards, next_states, demonstrative_actions, not_dones = self.customizedSampling(data_infos[data_source]['batch_size'], 
                                                                                                              {-1:0.3, 0:0.4, 1:0.3},
                                                                                                              is_from_dagger=data_infos[data_source]['is_from_dagger'])
            # Normalize Section
            for j in range(hp.history_frame_num): # process every frame
                states[:,:,:,j] = states[:,:,:,j] - self.states_mean[:data_infos[data_source]['batch_size'],:,:]
                states[:,:,:,j] = torch.div(states[:,:,:,j], self.states_var[:data_infos[data_source]['batch_size'],:,:])
            for j in range(hp.history_frame_num): # process every frame
                next_states[:,:,:,j] = next_states[:,:,j] - self.states_mean[:data_infos[data_source]['batch_size'],:,:]
                next_states[:,:,:,j] = torch.div(next_states[:,:,j], self.states_var[:data_infos[data_source]['batch_size'],:,:])
            # the normalization process of action is in the critic net

            demonstrative_actions_list.append(demonstrative_actions)

            # printColor(data_source)
            # printColor(actions[1])##to do当前不是概率的 

            # Critic Section
            with torch.no_grad(): # with next action noise version
                noise_lon = torch.randn(data_infos[data_source]['batch_size'], 1).to(self.device)
                noise_lat = (torch.randint(0, 3, (data_infos[data_source]['batch_size'],1)) - 1).to(self.device) #torch.random

                next_actions = self.actor_target.forward(next_states)
                next_actions_with_noise = ((next_actions[0] + noise_lon).clamp(0.0, 20.0), (next_actions[1] + noise_lat).clamp(-1, 1))
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions_with_noise)
                target_Q = rewards + self.discount * torch.min(target_Q1, target_Q2) * not_dones

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic.forward(states, actions) # 使用两个q网络计算，

            # Compute critic loss
            critic_loss[data_source] = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) # 对两个Q网络都更新

            if self.combined_update_mode[data_source]['reinforcement']:
                self.critic_optimizer.zero_grad()
                critic_loss[data_source].backward()
                clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # 梯度裁减，防止梯度爆炸
                for name, param in self.critic.named_parameters():
                    if param.requires_grad:
                        if data_source == 'demonstration' and self.combined_update_mode['interaction']['reinforcement'] and self.conflict_free:
                            param.grad = getGradProjection(param.grad, self.critic_gradients['interaction'][name], k_anealing=k_anealing)
                            # param.grad = grad_projection * k_anealing + param.grad * (1-k_anealing)
                        self.critic_gradients[data_source][name] = param.grad
                self.critic_optimizer.step()

            if is_actor_update_epoc:
                policy_actions = self._act(states, execute_dropout=self.epistemic_estimation) # -> Tuple
                # if self.total_it % hp.epistemic_estimation_freq == 1:
                #     self.epistemicEstimate(demonstrative_states, demonstrative_actions)
                policy_actions = (policy_actions[0].squeeze().unsqueeze(1), policy_actions[1].squeeze())
                policy_actions_list.append(policy_actions)
                
                # RL Section
                if self.combined_update_mode[data_source]['reinforcement']:
                    Q = self.critic.Q1(states, policy_actions)
                    lmbda = self.alpha / Q.abs().mean().detach() # 系数
                    actor_loss[data_source]['reinforcement'] = -lmbda * Q.mean()
                
                    self.actor_optimizer.zero_grad()
                    actor_loss[data_source]['reinforcement'].backward()
                    clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 梯度裁减，防止梯度爆炸

                    for name, param in self.actor.named_parameters():
                        if param.requires_grad:
                            if data_source == 'demonstration' and self.combined_update_mode['interaction']['reinforcement'] and self.conflict_free:
                                param.grad = getGradProjection(param.grad, self.actor_gradients['interaction']['reinforcement'][name], k_anealing=k_anealing)
                            self.actor_gradients[data_source]['reinforcement'][name] = param.grad
                    self.actor_optimizer.step()
                    self.is_initialized[data_source]['reinforcement'] = 1

                # IL Section
                if self.combined_update_mode[data_source]['imitation']:
                    policy_actions = self._act(states, execute_dropout=self.epistemic_estimation) # -> Tuple
                    # if self.total_it % hp.epistemic_estimation_freq == 1:
                    #     self.epistemicEstimate(demonstrative_states, demonstrative_actions)
                    policy_actions = (policy_actions[0].squeeze().unsqueeze(1), policy_actions[1].squeeze())
                    self.actor_optimizer.zero_grad()
                    loss_lon = self.loss_function(policy_actions[0], demonstrative_actions[0])
                    loss_lat = self.loss_function(policy_actions[1], demonstrative_actions[1])
                    actor_loss[data_source]['imitation'] = loss_lon + loss_lat * 2
                
                    self.actor_optimizer.zero_grad()
                    actor_loss[data_source]['imitation'].backward()
                    clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 梯度裁减，防止梯度爆炸
                    for name, param in self.actor.named_parameters():
                        if param.requires_grad: # save current gradient
                            if data_source == 'demonstration' and self.combined_update_mode['interaction']['imitation'] and self.conflict_free:
                                param.grad = getGradProjection(param.grad, self.actor_gradients['interaction']['imitation'][name], k_anealing=k_anealing)
                            self.actor_gradients[data_source]['imitation'][name] = param.grad
                    self.actor_optimizer.step()
                    self.is_initialized[data_source]['imitation'] = 1

        # 计算冲突强度，冲突强度的变化是否可以和振动强度的变化相关联，找一下振动强度的关键公式
        conflict_intensity = {}
        # calculate conflict_intensity of actor update 
        completed_conflict_dual_label = []
        for data_source_1 in ['demonstration', 'interaction']:
            for update_method_1 in ['imitation', 'reinforcement']:
                gradient_1 = self.actor_gradients[data_source_1][update_method_1] # loop to get first gradient
                label_1 = data_source_1 + '_' + update_method_1
                if not self.is_initialized[data_source_1][update_method_1]:
                    continue

                for data_source_2 in ['demonstration', 'interaction']:
                    for update_method_2 in ['imitation', 'reinforcement']:
                        gradient_2 = self.actor_gradients[data_source_2][update_method_2] # loop to get second gradient
                        label_2 = data_source_2 + '_' + update_method_2
                        if not self.is_initialized[data_source_2][update_method_2] or label_1 == label_2:
                            continue

                        conflict_label =  label_1 + '_vs_' + label_2
                        if not conflict_label in completed_conflict_dual_label:
                            completed_conflict_dual_label.append(label_2 + '_vs_' + label_1)
                            conflict_intensity[conflict_label] = 0
                            for name, param in self.actor.named_parameters():
                                if param.requires_grad:
                                    dot_product = torch.dot(gradient_1[name].view(-1), gradient_2[name].view(-1))
                                    gradient_norm_1 = torch.norm(gradient_1[name]).item()
                                    gradient_norm_2 = torch.norm(gradient_2[name]).item()
                                    conflict_intensity[conflict_label] += dot_product / gradient_norm_1 / gradient_norm_2

        # calculate conflict_intensity of critic update 
        if self.combined_update_mode['demonstration']['reinforcement'] and self.combined_update_mode['interaction']['reinforcement']:
            conflict_intensity['critic_rl'] = 0
            param_set_num = 0
            gradient_1 = self.critic_gradients['demonstration']
            gradient_2 = self.critic_gradients['interaction']
            for name, param in self.critic.named_parameters():
                if param.requires_grad:
                    dot_product = torch.dot(gradient_1[name].view(-1), gradient_2[name].view(-1))
                    gradient_norm_1 = torch.norm(gradient_1[name]).item()
                    gradient_norm_2 = torch.norm(gradient_2[name]).item()
                    conflict_intensity['critic_rl'] += dot_product / gradient_norm_1 / gradient_norm_2
                    param_set_num += 1
            conflict_intensity['critic_rl'] /= param_set_num

        # calculate conflict_intensity between Demonstration and Interaction
        # inpt:combined_update_mode actor_gradients output:
        conflict_intensity['demonstration_vs_interaction'] = 0
        if ((self.combined_update_mode['demonstration']['reinforcement'] and self.is_initialized['demonstration']['reinforcement']) \
            or (self.combined_update_mode['demonstration']['imitation'] and self.is_initialized['demonstration']['imitation'])) \
            and ((self.combined_update_mode['interaction']['reinforcement'] and self.is_initialized['interaction']['reinforcement']) \
            or (self.combined_update_mode['interaction']['imitation'])  and self.is_initialized['interaction']['imitation']):
            param_set_num = 0
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    gradient_demonstration = None
                    if self.combined_update_mode['demonstration']['reinforcement'] \
                        and self.is_initialized['demonstration']['reinforcement']:
                        if gradient_demonstration == None:
                            gradient_demonstration = self.actor_gradients['demonstration']['reinforcement'][name]
                        else:
                            gradient_demonstration += self.actor_gradients['demonstration']['reinforcement'][name]
                    if self.combined_update_mode['demonstration']['imitation'] \
                        and self.is_initialized['demonstration']['imitation']:
                        if gradient_demonstration == None:
                            gradient_demonstration = self.actor_gradients['demonstration']['imitation'][name]
                        else:
                            gradient_demonstration += self.actor_gradients['demonstration']['imitation'][name]
                    gradient_interaction = None
                    if self.combined_update_mode['interaction']['reinforcement'] \
                        and self.is_initialized['interaction']['reinforcement']:
                        if gradient_interaction == None:
                            gradient_interaction = self.actor_gradients['interaction']['reinforcement'][name]
                        else:
                            gradient_interaction += self.actor_gradients['interaction']['reinforcement'][name]
                    if self.combined_update_mode['interaction']['imitation'] \
                        and self.is_initialized['interaction']['imitation']:
                        if gradient_interaction == None:
                            gradient_interaction = self.actor_gradients['interaction']['imitation'][name]
                        else:
                            gradient_interaction += self.actor_gradients['interaction']['imitation'][name]
                    
                    dot_product = torch.dot(gradient_demonstration.view(-1), gradient_interaction.view(-1))
                    gradient_demonstration_norm = torch.norm(gradient_demonstration).item()
                    gradient_interaction_norm = torch.norm(gradient_interaction).item()
                    conflict_intensity['demonstration_vs_interaction'] += dot_product / gradient_demonstration_norm / gradient_interaction_norm
                    param_set_num += 1
            conflict_intensity['demonstration_vs_interaction'] /= param_set_num
        
        # calculate conflict_intensity between imitation and reinforcement
        conflict_intensity['reinforcement_vs_imitation'] = 0
        if ((self.combined_update_mode['demonstration']['reinforcement'] and self.is_initialized['demonstration']['reinforcement']) 
            or (self.combined_update_mode['interaction']['reinforcement'] and self.is_initialized['interaction']['reinforcement'])) \
            and ((self.combined_update_mode['demonstration']['imitation'] and self.is_initialized['demonstration']['imitation'])
            or (self.combined_update_mode['interaction']['imitation'] and self.is_initialized['interaction']['imitation'])):
            param_set_num = 0
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    gradient_reinforcement = None
                    if self.combined_update_mode['demonstration']['reinforcement'] \
                        and self.is_initialized['demonstration']['reinforcement']:
                        if gradient_reinforcement == None:
                            gradient_reinforcement = self.actor_gradients['demonstration']['reinforcement'][name]
                        else:
                            gradient_reinforcement += self.actor_gradients['demonstration']['reinforcement'][name]
                    if self.combined_update_mode['interaction']['reinforcement'] \
                        and self.is_initialized['interaction']['reinforcement']:
                        if gradient_reinforcement == None:
                            gradient_reinforcement = self.actor_gradients['interaction']['reinforcement'][name]
                        else:
                            gradient_reinforcement += self.actor_gradients['interaction']['reinforcement'][name]
                    gradient_imitation = None
                    if self.combined_update_mode['demonstration']['imitation']:
                        if gradient_imitation == None:
                            gradient_imitation = self.actor_gradients['demonstration']['imitation'][name]
                        else:
                            gradient_imitation += self.actor_gradients['demonstration']['imitation'][name]
                    if self.combined_update_mode['interaction']['imitation']:
                        if gradient_imitation == None:
                            gradient_imitation = self.actor_gradients['interaction']['imitation'][name]
                        else:
                            gradient_imitation += self.actor_gradients['interaction']['imitation'][name]
                    
                    dot_product = torch.dot(gradient_reinforcement.view(-1), gradient_imitation.view(-1))
                    gradient_reinforcement_norm = torch.norm(gradient_reinforcement).item()
                    gradient_imitation_norm = torch.norm(gradient_imitation).item()
                    conflict_intensity['reinforcement_vs_imitation'] += dot_product / gradient_reinforcement_norm / gradient_imitation_norm
                    param_set_num += 1
            conflict_intensity['reinforcement_vs_imitation'] /= param_set_num
    
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if is_actor_update_epoc:
            train_info = self.getTrainInfoRevision(demonstrative_actions_list, policy_actions_list, critic_loss)
        else:
            train_info = self.getTrainInfoRevision(None, None, critic_loss)
        train_info['conflict_intensity'] = conflict_intensity
        return train_info 
    
    def getTrainInfoRevision(self, demonstrative_actions_list, policy_actions_list, critic_loss):
        critic_loss_total = 0
        for _, val in critic_loss.items():
            critic_loss_total += val
        if policy_actions_list == None:
            return self.getTrainInfo(None, None, critic_loss_total)
        
        demonstrative_actions = (torch.cat([demonstrative_actions_list[0][0], demonstrative_actions_list[1][0]], dim=0), 
                                 torch.cat([demonstrative_actions_list[0][1], demonstrative_actions_list[1][1]], dim=0))
        policy_actions = (torch.cat([policy_actions_list[0][0], policy_actions_list[1][0]], dim=0), 
                            torch.cat([policy_actions_list[0][1], policy_actions_list[1][1]], dim=0))
        return self.getTrainInfo(demonstrative_actions, policy_actions, critic_loss_total)
