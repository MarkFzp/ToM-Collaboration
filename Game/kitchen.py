import copy
import numpy as np
from easydict import EasyDict as edict

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")
from game import Game
from utils.dataset_kitchen import load_data
from pprint import pprint as brint

class Kitchen(Game):
    def __init__(self, player1, player2,
                 step_reward, fail_reward,
                 succeed_reward, ingredients,
                 num_dishes, max_dish_ingredients, dataset_path=None):
        
        super().__init__(player1, player2, step_reward, fail_reward, succeed_reward)
        
        self.ingredients_ = ingredients
        self.num_ingredients_ = len(self.ingredients_)
        self.num_dishes_ = num_dishes
        self.max_dish_ingredients_ = max_dish_ingredients
        self.dataset = load_data(dataset_path) if dataset_path is not None else None
        if self.dataset is not None:
            self.train_idx = 0
            self.test_idx = 0
            self.train_size = self.dataset['train']['menu'].shape[0]
            self.test_size = self.dataset['test']['menu'].shape[0]
            print('load date from %s' % dataset_path)

    def reset(self, menu = None, order = None, train = True):
        self.state_ = edict()
        self.observation_ = edict()
        if self.dataset is None:
            if menu is None:
                menu = set()
                included_ingredients = set()
                while len(menu) < self.num_dishes_:
                    concept_idx = np.sort(np.random.choice(self.num_ingredients_ + 1, self.max_dish_ingredients_))
                    dish = tuple([self.ingredients_[ci] for ci in concept_idx if ci < self.num_ingredients_])
                    if len(dish) == 0:
                        continue
                    menu.add(dish)
                    for ingredient in dish:
                        included_ingredients.add(ingredient)
                self.state_.menu = list(menu)
                self.state_.included_ingredients = list(included_ingredients)

            else:
                self.state_.menu = list(menu)
                assert(len(self.state_.menu) == self.num_dishes_)
                self.state_.included_ingredients = set()
                for dish in self.state_.menu:
                    for i in dish:
                        self.state_.included_ingredients.add(i)

            self.state_.menu_embed = np.zeros((self.num_dishes_, self.num_ingredients_))
            for idx, dish in enumerate(menu):
                for ing in dish:
                    self.state_.menu_embed[idx, ing] += 1
            
            if order is None:
                self.state_.order = np.random.choice(self.num_dishes_)
            else:
                self.state_.order = order
            
            self.state_.order_embed = np.zeros(self.num_dishes_)
            self.state_.order_embed[self.state_.order] = 1

        else:
            if train:
                mode = 'train'
                idx = self.train_idx
            else:
                mode = 'test'
                idx = self.test_idx
            
            if menu is None:
                menu_embed = self.dataset[mode]['menu'][idx]
                menu = []
                included_ingredients = set()
                for dish_embed in menu_embed:
                    dish = []
                    for i in range(self.num_ingredients_):
                        for _ in range(int(dish_embed[i])):
                            dish.append(i)
                    included_ingredients.update(dish)
                    menu.append(tuple(dish))
                self.state_.menu = menu
                self.state_.included_ingredients = list(included_ingredients)
                self.state_.menu_embed = menu_embed
                
                if train:
                    self.train_idx += 1
                    if self.train_idx >= self.train_size:
                        self.train_idx = 0
                else:
                    self.test_idx += 1
                    if self.test_idx >= self.test_size:
                        self.test_idx = 0
            
            else:
                self.state_.menu = list(menu)
                # assert(len(self.state_.menu) == self.num_dishes_)
                self.state_.included_ingredients = set()
                for dish in self.state_.menu:
                    self.state_.included_ingredients.update(dish)
                
                self.state_.menu_embed = np.zeros((self.num_dishes_, self.num_ingredients_))
                for idx, dish in enumerate(menu):
                    for ing in dish:
                        self.state_.menu_embed[idx, ing] += 1

            if order is None:
                self.state_.order = self.dataset[mode]['order'][idx]
            else:
                self.state_.order = order

            self.state_.order_embed = np.zeros(self.num_dishes_)
            self.state_.order_embed[self.state_.order] = 1

        self.observation_.menu_embed = self.state_.menu_embed
        self.state_.workplace = []
        self.state_.workplace_embed = np.zeros(self.num_ingredients_)
        self.observation_.workplace_embed = self.state_.workplace_embed

    def start(self):
        return {self.player1_name_: edict({'observation': copy.deepcopy(self.observation_), 'private': self.state_.order_embed, 'terminate': False}),
                self.player2_name_: edict({'observation': copy.deepcopy(self.observation_), 'private': None, 'terminate': False})}
    
    def proceed(self, action):
        assert(np.sum(self.state_.menu_embed[self.state_.order, :] < self.state_.workplace_embed) == 0)
        action = action[self.player1_name_] if action.get(self.player1_name_) is not None else action[self.player2_name_]
        self.state_.workplace.append(action)
        self.state_.workplace_embed[action] += 1
        self.observation_.workplace_embed = self.state_.workplace_embed
        
        success = None
        if action not in self.state_.menu[self.state_.order]:
            reward = self.fail_reward_# - self.step_reward_
            terminate = True
            success = False
        elif np.sum(self.state_.menu_embed[self.state_.order, :] != self.state_.workplace_embed) == 0:
            reward = self.succeed_reward_ + self.step_reward_
            terminate = True
            success = True
         # prepared unnecessary ingredients
        elif np.sum(self.state_.menu_embed[self.state_.order, :] < self.state_.workplace_embed) > 0:
            reward = self.fail_reward_# - self.step_reward_
            terminate = True
            success = False
        else:
            reward = self.step_reward_
            terminate = False
        
        package = edict({'observation': copy.deepcopy(self.observation_), 'reward': reward, 'terminate': terminate})
        if success is not None:
            package.success = success
        return {self.player1_name_: package, self.player2_name_: package}
            

    def observe(self, player_name):
        assert(self.state_)
        if player_name == self.player1_name_ or player_name == self.player2_name_:
            return self.observation_
        else:
            assert(0)


def main():
    kitchen = Kitchen('A', 'B', 0.15, 0, 1, range(10), 4, 4)#, dataset_path='/home/Datasets/Kitchen/dataset1_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz')
    kitchen.reset()
    brint(kitchen.state_)
    brint(kitchen.start())
    brint(kitchen.proceed({'A': 1}))
    brint(kitchen.observation_)

if __name__ == '__main__':
    main()
