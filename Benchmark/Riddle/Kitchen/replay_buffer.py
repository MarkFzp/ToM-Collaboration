import numpy as np

class ReplayBuffer:

    def __init__(self, config):

        self.max_batch = config.max_batch
        self.size = 0
        assert self.size <= self.max_batch

        self.t_storage = []
        self.s_storage = []
        self.target_menu = []
        self.success = []

    def store(self, t_tranjectory, s_tranjectory, target_menu, success):

        if self.size >= self.max_batch:
            self.t_storage = self.t_storage[1:]
            self.s_storage = self.s_storage[1:]
            self.target_menu = self.target_menu[1:]
            self.success = self.success[1:]
            self.size -= 1

        self.t_storage.append(t_tranjectory)
        self.s_storage.append(s_tranjectory)
        self.target_menu.append(target_menu)
        self.success.append(success)
        self.size += 1

    def get_target(self, ind):
        return self.target_menu[ind]

    def get_data(self, ind, agent):

        if agent == 0:
            return self.t_storage[ind]
        elif agent == 1:
            return self.s_storage[ind]

    def accuracy(self):
        acc_count = 0
        wrong_count = 0
        for i in range(self.size):
            if self.success[i]:
                acc_count += 1
            else:
                wrong_count += 1
        acc_rate = acc_count / self.size
        return acc_rate
