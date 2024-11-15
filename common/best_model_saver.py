import os
import math
import copy
import torch
from common.fixed_size_queue import FixedSizeQueue

class BestModelSaver:
    def __init__(self, num_for_average, save_dir) -> None:
        self.num_for_average = num_for_average
        self.num_for_average = num_for_average

        self.best_model_performance = -math.inf # for save the best model
        self.model_performances = [] # for save the best model

        self.critic_queue = FixedSizeQueue(int(num_for_average / 2))
        self.actor_queue = FixedSizeQueue(int(num_for_average / 2))
        self.epoc_queue = FixedSizeQueue(int(num_for_average / 2))

        self.save_dir = save_dir + '/' + 'best_model'
        os.makedirs(self.save_dir)

    def addModelToQueue(self, epoc, policy):
        self.critic_queue.add(copy.deepcopy(policy.critic.state_dict()))
        self.actor_queue.add(copy.deepcopy(policy.actor.state_dict()))
        self.epoc_queue.add(epoc)

    def process(self, epoc, policy, performance_score):
        self.model_performances.append(performance_score)
        self.addModelToQueue(epoc, policy)

        if not self.actor_queue.queue.full():
            return
        current_model_performance = sum(self.model_performances[-self.num_for_average:]) / len(self.model_performances[-self.num_for_average:])
        if current_model_performance > self.best_model_performance:
            self.best_model_performance = current_model_performance

            torch.save(self.critic_queue.get(), self.save_dir + "/critic")
            torch.save(self.actor_queue.get(), self.save_dir + "/actor")

            with open(self.save_dir + '/info.txt', "w") as file:
                file.write(str(self.epoc_queue.get()) + '_' + str(int(current_model_performance * 100)))