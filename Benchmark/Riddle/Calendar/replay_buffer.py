import numpy as np

class ReplayBuffer:

    def __init__(self, config):

        self.max_batch = config.max_batch
        self.size = 0
        assert self.size <= self.max_batch

        self.t_storage = []
        self.s_storage = []
        self.meetable = []
        self.success = []
        self.starter = []
        self.sar_sequence = []

    def store(self, t_tranjectory, s_tranjectory, meetable, success, starter, sar_sequence):

        if self.size >= self.max_batch:
            self.t_storage = self.t_storage[1:]
            self.s_storage = self.s_storage[1:]
            self.meetable = self.meetable[1:]
            self.success = self.success[1:]
            self.starter = self.starter[1:]
            self.sar_sequence = self.sar_sequence[1:]
            self.size -= 1

        self.t_storage.append(t_tranjectory)
        self.s_storage.append(s_tranjectory)
        self.meetable.append(meetable)
        self.success.append(success)
        self.starter.append(starter)
        self.sar_sequence.append(sar_sequence)
        self.size += 1

    def get_meetable(self, ind):
        return self.meetable[ind]

    def get_starter(self, ind):
        return self.starter[ind]

    def get_sar_sequence(self, ind):
        return self.sar_sequence[ind]

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
