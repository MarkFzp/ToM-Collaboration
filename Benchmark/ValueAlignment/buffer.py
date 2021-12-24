import numpy as np
from collections import OrderedDict

class Buffer:

    def __init__(self, num_dishes, num_ingredients, num_haction, num_raction, max_batch, bs):

        assert bs <= max_batch
        self.max_batch = max_batch
        self.bs = bs

        self.count = 0

        self.storage = OrderedDict()

        self.storage['menu'] = np.zeros((max_batch, num_dishes, num_ingredients))
        self.storage['workplace_embed'] = np.zeros((max_batch, num_ingredients))
        self.storage['belief'] = np.zeros((max_batch, num_dishes))
        self.storage['human_actions'] = np.zeros((max_batch, num_haction))
        self.storage['robot_actions'] = np.zeros((max_batch, num_raction))
        self.storage['rewards'] = np.zeros((max_batch,))
        self.storage['next_menu'] = np.zeros((max_batch, num_dishes, num_ingredients))
        self.storage['next_workplace_embed'] = np.zeros((max_batch, num_ingredients))
        self.storage['goal'] = np.zeros((max_batch,))
        self.storage['terminal'] = np.zeros((max_batch,))


    def store(self, menu, workplace_embed, belief, haction, raction, rewards,
              next_menu, next_workplace_embed, goal, terminal):

        if self.count >= self.max_batch:
            for k in self.storage:
                self.storage[k] = np.roll(self.storage[k], -1, axis=0)

            s = 0
        else:
            s = self.count
            # if s == 0:
            # for k in self.storage:
            #     self.storage[k].pop(0)
            # else:
            #     ind = np.concatenate(self.storage['rewards'], axis=0) > 0
            #     if ind.sum() > self.max_batch:
            #         ind[np.random.choice(ind.sum(), ind.sum() - self.max_batch, replace=False)] = False
            #     else:
            #         ind[np.random.choice(self.max_batch, self.max_batch-ind.sum(), replace=False)] = True
            #     for k in self.storage:
            #         if k == 'rewards' or k == 'terminal' or 'goal':
            #             self.storage[k] = [np.array([x]) for x in np.concatenate(self.storage[k], axis=0)[ind]]
            #         else:
            #             self.storage[k] = [x[np.newaxis,:] for x in np.concatenate(self.storage[k], axis=0)[ind]]
        self.storage['menu'][s] = menu
        self.storage['workplace_embed'][s] = workplace_embed
        self.storage['belief'][s] = belief
        self.storage['human_actions'][s] = haction
        self.storage['robot_actions'][s] = raction
        self.storage['rewards'][s] = rewards
        self.storage['next_menu'][s] = next_menu
        self.storage['next_workplace_embed'][s] = next_workplace_embed
        self.storage['goal'][s] = goal
        self.storage['terminal'][s] = terminal

        if self.count < self.max_batch:
            self.count += 1

    def get_data(self):
        ind = np.random.permutation(self.max_batch)
        menu = self.storage['menu'][ind]
        workplace_embed = self.storage['workplace_embed'][ind]
        belief = self.storage['belief'][ind]
        haction = self.storage['human_actions'][ind]
        raction = self.storage['robot_actions'][ind]
        rewards = self.storage['rewards'][ind]
        next_menu = self.storage['next_menu'][ind]
        next_workplace_embed = self.storage['next_workplace_embed'][ind]
        goal = self.storage['goal'][ind]
        terminal = self.storage['terminal'][ind]

        for count in range(0, self.max_batch, self.bs):
            menu_b = menu[count:min(self.max_batch, count + self.bs)]
            workplace_embed_b = workplace_embed[count:min(self.max_batch, count + self.bs)]
            belief_b = belief[count:min(self.max_batch, count + self.bs)]
            haction_b = haction[count:min(self.max_batch, count + self.bs)]
            raction_b = raction[count:min(self.max_batch, count + self.bs)]
            rewards_b = rewards[count:min(self.max_batch, count + self.bs)]
            next_menu_b = next_menu[count:min(self.max_batch, count + self.bs)]
            next_workplace_embed_b = next_workplace_embed[count:min(self.max_batch, count + self.bs)]
            goal_b = goal[count:min(self.max_batch, count + self.bs)]
            terminal_b = terminal[count:min(self.max_batch, count + self.bs)]

            yield menu_b, workplace_embed_b, belief_b, haction_b, raction_b, rewards_b, next_menu_b, next_workplace_embed_b, goal_b, terminal_b


    def is_available(self):
        return self.count >= self.max_batch


class Bufferv2:

    def __init__(self, max_batch, bs):

        assert bs <= max_batch
        self.max_batch = max_batch
        self.bs = bs

        self.storage = {
            'menu': [],
            'workplace_embed':[],
            'belief':[],
            'human_actions': [],
            'robot_actions':[],
            'rewards':[],
            'next_menu': [],
            'next_workplace_embed':[],
            'goal':[],
            'terminal': []
        }


    def store(self, menu, workplace_embed, belief, haction, raction, rewards,
              next_menu, next_workplace_embed, goal, terminal):

        if len(self.storage['menu']) > self.max_batch:
            # s = np.random.choice([0, 1], p=[0.7, 0.3])
            # if s == 0:
            for k in self.storage:
                self.storage[k].pop(0)
            # else:
            #     ind = np.concatenate(self.storage['rewards'], axis=0) > 0
            #     if ind.sum() > self.max_batch:
            #         ind[np.random.choice(ind.sum(), ind.sum() - self.max_batch, replace=False)] = False
            #     else:
            #         ind[np.random.choice(self.max_batch, self.max_batch-ind.sum(), replace=False)] = True
            #     for k in self.storage:
            #         if k == 'rewards' or k == 'terminal' or 'goal':
            #             self.storage[k] = [np.array([x]) for x in np.concatenate(self.storage[k], axis=0)[ind]]
            #         else:
            #             self.storage[k] = [x[np.newaxis,:] for x in np.concatenate(self.storage[k], axis=0)[ind]]
        self.storage['menu'].append(menu)
        self.storage['workplace_embed'].append(workplace_embed)
        self.storage['belief'].append(belief)
        self.storage['human_actions'].append(haction)
        self.storage['robot_actions'].append(raction)
        self.storage['rewards'].append(np.array([rewards]))
        self.storage['next_menu'].append(next_menu)
        self.storage['next_workplace_embed'].append(next_workplace_embed)
        self.storage['goal'].append(np.array([goal]))
        self.storage['terminal'].append(np.array([terminal]))

    def get_data(self):
        ind = np.random.choice(len(self.storage['menu']), min(self.bs, len(self.storage['menu'])), replace=False)

        menu = np.concatenate(self.storage['menu'], axis=0)[ind]
        workplace_embed = np.concatenate(self.storage['workplace_embed'], axis=0)[ind]
        belief = np.concatenate(self.storage['belief'], axis=0)[ind]
        haction = np.concatenate(self.storage['human_actions'], axis=0)[ind]
        raction = np.concatenate(self.storage['robot_actions'], axis=0)[ind]
        rewards = np.concatenate(self.storage['rewards'], axis=0)[ind]
        next_menu = np.concatenate(self.storage['next_menu'], axis=0)[ind]
        next_workplace_embed = np.concatenate(self.storage['next_workplace_embed'], axis=0)[ind]
        goal = np.concatenate(self.storage['goal'], axis=0)[ind]
        terminal = np.concatenate(self.storage['terminal'], axis=0)[ind]


        return menu, workplace_embed, belief, haction, raction, rewards, next_menu, next_workplace_embed, goal, terminal


    def is_available(self):
        return len(self.storage['menu'])>self.bs