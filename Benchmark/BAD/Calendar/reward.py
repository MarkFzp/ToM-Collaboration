import tensorflow as tf
import numpy as np

class Reward:
    def __init__(self, config, calendar):
        self.calendar = calendar
        self.p1 = 0
        self.p2 = 1

    def get_reward(self, p1_cids, p2_cids, actions, action_p_idx):
        rewards = []
        if action_p_idx == self.p1:
            p2_cid = p2_cids + 1
            for p1_cid, action in zip(p1_cids, actions):
                self.calendar.reset(player1_cid = p1_cid + 1, player2_cid = p2_cid)
                ret = self.calendar.proceed({action_p_idx: action})[self.p1]
                rewards.append(ret.reward)
        else:
            p1_cid = p1_cids + 1
            for p2_cid, action in zip(p2_cids, actions):
                self.calendar.reset(player1_cid = p1_cid, player2_cid = p2_cid + 1)
                ret = self.calendar.proceed({action_p_idx: action})[self.p2]
                rewards.append(ret.reward)

        return np.array(rewards)
