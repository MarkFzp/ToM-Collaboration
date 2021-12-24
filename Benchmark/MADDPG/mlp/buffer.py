import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, num_agents):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """

        self._maxsize = int(size)
        self._next_idx = 0
        assert num_agents > 0
        self.num_agents = num_agents

        self._storage = [{
            'menu': [],
            'workplace_embed': [],
            'actions':[],
            'rewards': [],
            'next_workplace_embed': [],
            'goal': [],
            'terminal': []
        } for _ in range(num_agents)]

    def __len__(self):
        return len(self._storage[0]['rewards'])

    def clear(self):
        for i in range(self.num_agents):
            for k in self._storage[i]:
                self._storage[i][k] = []
        self._next_idx = 0

    def add(self, menu, workplace_embed, action, rewards, next_workplace_embed,  goal, terminal):

        for i in range(self.num_agents):
            data = (menu[i], workplace_embed[i], action[i], rewards[i], next_workplace_embed[i],  goal[i], terminal[i])

            if self._next_idx >= len(self._storage[i]['menu']):
                for k,d in zip(self._storage[i], data):
                    self._storage[i][k].append(d)
            else:
                for k,d in zip(self._storage[i], data):
                    self._storage[i][k][self._next_idx] = d
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, ind):


        menu = [np.concatenate(self._storage[i]['menu'], axis=0)[ind] for i in range(self.num_agents)]
        workplace_embed = [np.concatenate(self._storage[i]['workplace_embed'], axis=0)[ind] for i in range(self.num_agents)]
        actions = [np.concatenate(self._storage[i]['actions'], axis=0)[ind] for i in range(self.num_agents)]
        rewards = [np.concatenate(self._storage[i]['rewards'], axis=0)[ind] for i in range(self.num_agents)]
        next_menu = menu
        next_workplace_embed = [np.concatenate(self._storage[i]['next_workplace_embed'], axis=0)[ind] for i in range(self.num_agents)]
        goal = [np.concatenate(self._storage[i]['goal'], axis=0)[ind] for i in range(self.num_agents)]
        terminal = [np.concatenate(self._storage[i]['terminal'], axis=0)[ind] for i in range(self.num_agents)]


        return menu, workplace_embed, actions, rewards, next_menu, next_workplace_embed, goal, terminal

    def make_index(self, batch_size):
        return np.random.choice(len(self._storage[0]['menu']), min(batch_size, len(self._storage[0]['menu'])), replace=False)

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)


    def is_available(self):
        return len(self._storage[0]['menu'])>=self._maxsize