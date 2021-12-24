import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, num_agents, max_traj_len):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """

        self._maxsize = int(size)
        self.max_traj_len = max_traj_len
        self._next_idx = 0
        assert num_agents > 0
        self.num_agents = num_agents

        self._storage = [{
            'private': [],
            'last_actions': [],
            'actions':[],
            'rewards': [],
            'terminal': []
        } for _ in range(num_agents)]


    def __len__(self):
        return len(self._storage[0]['rewards'])

    def clear(self):
        for i in range(self.num_agents):
            for k in self._storage[i]:
                self._storage[i][k] = []
        self._next_idx = 0


    def add(self, private, last_action, action, rewards, terminal):

        for i in range(self.num_agents):
            data = (private[i], last_action[i], action[i], rewards[i], terminal[i])
            data = self._process_traj(*data)

            if self._next_idx >= len(self._storage[i]['private']):
                for k,d in zip(self._storage[i], data):
                    self._storage[i][k].append(d)
            else:
                for k,d in zip(self._storage[i], data):
                    self._storage[i][k][self._next_idx] = d
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _process_traj(self, private, last_action, action, rewards, terminal):
        ## each is a list, a traj
        fill = self.max_traj_len - len(private)
        private = np.pad(np.stack(private, axis=1), ((0,0),(0,fill),(0,0)), 'constant', constant_values=0)
        last_action = np.pad(np.stack(last_action, axis=1), ((0,0),(0,fill)), 'constant',constant_values=-1)
        action = np.pad(np.stack(action, axis=1), ((0,0),(0,fill)), 'constant',constant_values=-1)
        rewards = np.pad(np.stack(rewards, axis=1), ((0,0),(0,fill)), 'constant',constant_values=0)
        # not goal
        terminal = np.pad(np.stack(terminal, axis=1), ((0,0),(0,fill)), 'constant',constant_values=True)

        return private, last_action, action, rewards, terminal

    def _encode_sample(self, ind):

        private = [np.concatenate(self._storage[i]['private'], axis=0)[ind] for i in range(self.num_agents)]
        last_actions = [np.concatenate(self._storage[i]['last_actions'], axis=0)[ind] for i in range(self.num_agents)]
        actions = [np.concatenate(self._storage[i]['actions'], axis=0)[ind] for i in range(self.num_agents)]
        rewards = [np.concatenate(self._storage[i]['rewards'], axis=0)[ind] for i in range(self.num_agents)]
        terminal = [np.concatenate(self._storage[i]['terminal'], axis=0)[ind] for i in range(self.num_agents)]


        return private, last_actions, actions, rewards, terminal

    def make_index(self, batch_size):
        return np.random.choice(len(self._storage[0]['private']), min(batch_size, len(self._storage[0]['private'])), replace=False)

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)


    def is_available(self):
        return len(self._storage[0]['private'])>=self._maxsize