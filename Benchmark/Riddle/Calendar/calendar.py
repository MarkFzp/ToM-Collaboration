import copy
from functools import reduce

import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from game import Game

from pprint import pprint as brint

import pdb


class Calendar(Game):
    def __init__(self, player1, player2,
                 step_reward, fail_reward,
                 succeed_reward, num_slots,
                 dataset_path_prefix=None):
        '''
        dataset_path_prefix example: "calendar_8_1"
        then training set file is "calendar_8_1_train.txt"
             test set file is "calendar_8_1_test.txt"
        '''

        super().__init__(player1, player2, step_reward, fail_reward, succeed_reward)
        self.num_slots_ = num_slots
        self.num_calendars_ = 2 ** self.num_slots_

        self.dataset_path_prefix_ = dataset_path_prefix
        if self.dataset_path_prefix_ is not None:
            self.train_path_ = self.dataset_path_prefix_ + '_train.txt'
            self.test_path_ = self.dataset_path_prefix_ + '_test.txt'
            self.train_cids_ = []
            self.test_cids_ = []
            if os.path.exists(self.train_path_):
                with open(self.train_path_, 'r') as train_f:
                    for cid in train_f:
                        cid = int(cid.rstrip('\n'))
                        if cid != 0:
                            self.train_cids_.append(cid)
            else:
                raise Exception('training set file {} not found'.format(self.train_path_))
            if os.path.exists(self.test_path_):
                with open(self.test_path_, 'r') as test_f:
                    for cid in test_f:
                        cid = int(cid.rstrip('\n'))
                        if cid != 0:
                            self.test_cids_.append(cid)
            else:
                raise Exception('test set file {} not found'.format(self.test_path_))

            self.train_cid_count_ = len(self.train_cids_)
            self.test_cid_count_ = len(self.test_cids_)
            print('Load data from %s or %s' % (self.train_path_, self.test_path_))
            assert (self.train_cid_count_ + self.test_cid_count_ == self.num_calendars_ - 1)

        binaries = [list(('{0:0%db}' % self.num_slots_).format(i)) for i in range(self.num_calendars_)]
        numbers = []
        for b in binaries:
            numbers.append([int(bb) for bb in b])
        self.tensor_ = np.array(numbers)
        self.calendar2msg_ = {}  # 0 means empty time slots, 1 means occupied time slots
        self.calendar2intervals_ = {}
        self.calendar_num_valid_msg_ = np.ones(self.num_calendars_)
        self.all_msgs_, self.all_msgs_tensor_ = self.get_msg(self.num_calendars_ - 1)
        self.num_msgs_ = len(self.all_msgs_)
        # msgs, proposals, reject
        self.action_tensor_ = np.concatenate([self.all_msgs_tensor_, np.zeros([self.num_msgs_, self.num_slots_])],
                                             axis=1)
        self.action_tensor_ = np.concatenate([self.action_tensor_,
                                              np.concatenate([np.zeros([self.num_slots_, self.num_slots_]),
                                                              1 - np.identity(self.num_slots_)], 1)], axis=0)
        self.action_tensor_ = np.concatenate([self.action_tensor_,
                                              np.concatenate([np.zeros([1, self.num_slots_]),
                                                              np.ones([1, self.num_slots_])], axis=1)], axis=0)
        self.num_actions_ = self.action_tensor_.shape[0]
        # initialize caches about msgs
        for i in range(self.num_calendars_):
            self.get_msg(i)

    def get_msg(self, idx):
        if idx == 0:
            self.calendar2msg_[idx] = ([], np.zeros((0, self.num_slots_)))
        if self.calendar2msg_.get(idx) is None:
            basic_msgs = list(np.nonzero(self.tensor_[idx, :])[0])
            msgs = []
            intervals = []
            for first in range(len(basic_msgs)):
                second = first + 1
                while second < len(basic_msgs):
                    if basic_msgs[second] == basic_msgs[second - 1] + 1:
                        msgs.append(tuple(range(basic_msgs[first], basic_msgs[second] + 1)))
                        second += 1
                    else:
                        break

            first = 0
            while first < len(basic_msgs):
                second = first
                while second < len(basic_msgs):
                    if second < len(basic_msgs) - 1 and basic_msgs[second] == basic_msgs[second + 1] - 1:
                        second += 1
                    else:
                        intervals.append(tuple(range(basic_msgs[first], basic_msgs[second] + 1)))
                        first = second + 1
                        break
            msgs += [tuple([m]) for m in basic_msgs]
            msg_tensor = np.zeros((len(msgs), self.num_slots_))
            for im, msg in enumerate(msgs):
                msg_tensor[im, list(msg)] = 1
            interval_tensor = np.zeros((len(intervals), self.num_slots_))
            for it, interval in enumerate(intervals):
                interval_tensor[it, list(interval)] = 1
            self.calendar2msg_[idx] = (msgs, msg_tensor)
            self.calendar_num_valid_msg_[idx] = len(msgs)
            self.calendar2intervals_[idx] = (intervals, interval_tensor)

        return self.calendar2msg_[idx]

    def reset(self, player1_cid=None, player2_cid=None, train=True):
        if self.dataset_path_prefix_ is None:
            self.observation_ = None
            self.player1_cid_ = np.random.randint(1, self.num_calendars_) if player1_cid is None else player1_cid
            self.player2_cid_ = np.random.randint(1, self.num_calendars_) if player2_cid is None else player2_cid
            self.meetable_slots_ = (self.tensor_[self.player1_cid_] + self.tensor_[self.player2_cid_] == 0)

        else:
            self.observation_ = None
            if train:
                self.player1_cid_ = np.random.choice(self.train_cids_) if player1_cid is None else player1_cid
                self.player2_cid_ = np.random.choice(self.train_cids_) if player2_cid is None else player2_cid
            else:
                self.player1_cid_ = np.random.choice(self.test_cids_) if player1_cid is None else player1_cid
                self.player2_cid_ = np.random.choice(self.test_cids_) if player2_cid is None else player2_cid
            self.meetable_slots_ = (self.tensor_[self.player1_cid_] + self.tensor_[self.player2_cid_] == 0)

        return

    def start(self):
        return {self.player1_name_: edict(
            {'observation': None, 'private': self.tensor_[self.player1_cid_], 'terminate': False}),
                self.player2_name_: edict(
                    {'observation': None, 'private': self.tensor_[self.player2_cid_], 'terminate': False})}

    def proceed(self, action):
        if action.get(self.player1_name_) is not None:
            actioner = self.player1_name_
            actioner_cid = self.player1_cid_
        else:
            actioner = self.player2_name_
            actioner_cid = self.player2_cid_

        action = action[actioner]

        action_success = None
        msg_success = None
        # reject
        if action == self.num_actions_ - 1:
            terminate = True
            if np.sum(self.meetable_slots_) == 0:
                reward = self.succeed_reward_  # +  self.step_reward_
                action_success = True
            else:
                reward = self.fail_reward_ / 2
                action_success = False
        # propose
        elif action >= self.num_msgs_:
            terminate = True
            if self.meetable_slots_[action - self.num_msgs_]:
                reward = self.succeed_reward_  # + self.step_reward_
                action_success = True
            else:
                reward = self.fail_reward_
                action_success = False
        # send msg
        else:
            if self.all_msgs_[action] not in self.calendar2msg_[actioner_cid][0]:
                reward = self.fail_reward_
                terminate = True
                msg_success = False
            else:
                terminate = False
                reward = self.step_reward_
                msg_success = True

        package = edict({'observation': copy.deepcopy(self.observation_), 'reward': reward, 'terminate': terminate})
        if action_success is not None:
            package.action_success = action_success
        elif msg_success is not None:
            package.msg_success = msg_success
        return {self.player1_name_: package, self.player2_name_: package}

    def observe(self, player_name):
        assert (self.state_)
        if player_name == self.player1_name_ or player_name == self.player2_name_:
            return self.observation_
        else:
            assert (0)

    def msg_overlap(self):
        assert (self.dataset_path_prefix_ is not None)
        train_msg_set = set()
        total_test_msg_count = 0
        test_msg_in_train_cout = 0

        for train_cid in self.train_cids_:
            train_msg_set.update(self.calendar2msg_[train_cid][0])

        for test_cid in self.test_cids_:
            for msg in self.calendar2msg_[test_cid][0]:
                if msg in train_msg_set:
                    # print(msg)
                    test_msg_in_train_cout += 1
                # else:
                #     print('no')
                total_test_msg_count += 1

        return test_msg_in_train_cout / total_test_msg_count


def test_overlap():
    num_slots = 8
    base_path = 'Game'
    msg_overlap_l = []
    for idx in range(1, 4):
        dataset_idx = idx
        calendar = Calendar('A', 'B', 0, -1, 1, num_slots,
                            os.path.join(base_path, 'calendar_{}_{}'.format(num_slots, dataset_idx)))
        msg_overlap_l.append(calendar.msg_overlap())
    print('msg overlap for all splits: {}'.format(msg_overlap_l))
    print('average overlap: {}'.format(np.mean(msg_overlap_l)))


def main():
    num_slots = 10
    dataset_idx = 1
    base_path = 'Game'
    calendar = Calendar('A', 'B', 0, -1, 1, num_slots,
                        os.path.join(base_path, 'calendar_{}_{}'.format(num_slots, dataset_idx)))
    print(calendar.tensor_.shape)
    print(calendar.tensor_[122, :])
    print(calendar.calendar2msg_[122])
    print(calendar.all_msgs_tensor_.shape)
    meetable = np.zeros(num_slots)
    max_cont = np.zeros(10)
    not_meetable = 0
    for _ in range(10000):
        calendar.reset()
        initial_calendars = calendar.start()
        ca = initial_calendars['A']
        cb = initial_calendars['B']
        str_ca = reduce((lambda x, y: str(x) + str(y)), ca.private)
        max_cont[max([len(str_) for str_ in str_ca.split('0')]) - 1] += 1
        str_cb = reduce((lambda x, y: str(x) + str(y)), cb.private)
        max_cont[max([len(str_) for str_ in str_cb.split('0')]) - 1] += 1
        meetable += calendar.meetable_slots_
        not_meetable += np.sum(calendar.meetable_slots_) == 0
    print(meetable / 1e4)
    print(not_meetable / 1e4)
    print(max_cont / 20000)


if __name__ == '__main__':
    test_overlap()
