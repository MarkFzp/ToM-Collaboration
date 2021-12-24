import numpy as np

class ReplayBuffer:
    def __init__(self, config):
        self.buffer_size = config.buffer_size
        self.idx = 0
        self.trajs = np.array([None] * self.buffer_size)
    
    def push(self, traj):
        self.trajs[self.idx] = traj
        self.idx += 1
        if self.idx == self.buffer_size:
            self.idx = 0

    def get(self, count):
        sampled_idx = np.random.choice(self.buffer_size, size = count, replace = False)
        return self.trajs[sampled_idx].tolist()

    def accuracy(self):
        acc_count = 0
        wrong_count = 0
        p2_wrong = 0
        for traj in self.trajs:
            if traj['success']:
                acc_count += 1
            else:
                wrong_count += 1
                if len(traj['sarsa']) % 2 == 0:
                    p2_wrong += 1
        acc_rate = acc_count / self.buffer_size
        p2_wrong_rate = p2_wrong / wrong_count
        return acc_rate, p2_wrong_rate
    
    def mean_traj_len(self):
        total_traj_len = 0
        for traj in self.trajs:
            total_traj_len += len(traj['sarsa'])
        mean_traj_len = total_traj_len / self.buffer_size
        return mean_traj_len
