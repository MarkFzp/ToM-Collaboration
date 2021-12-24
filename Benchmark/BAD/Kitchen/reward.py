import tensorflow as tf
import numpy as np

class Reward:
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config
        self.identity = np.eye(config.num_action)

        if config.reward_architecture != 'adhoc':

            with tf.variable_scope('Reward'):
                # state
                self.menu = tf.placeholder(tf.float32, [None, config.num_candidate, config.num_ingredient])
                self.workplace = tf.placeholder(tf.float32, [None, config.num_ingredient])
                self.target = tf.placeholder(tf.float32, [None, config.num_candidate])

                # action
                self.action = tf.placeholder(tf.int32, [None])
                self.action_oh = tf.one_hot(self.action, depth = config.num_action)
                
                if config.reward_architecture in ['mlp']:
                    self.menu_flat = tf.reshape(self.menu, [-1, config.num_candidate * config.num_ingredient])
                    self.in_x = tf.concat([
                        self.menu_flat, 
                        self.workplace, 
                        self.target, 
                        self.action_oh
                    ], axis = -1)
                    self.fc_layers = [self.in_x]
                    for i, dim in enumerate(config.reward_mlp_fc_dim):
                        if i == len(config.reward_mlp_fc_dim) - 1:
                            act = None
                        else:
                            act = tf.nn.leaky_relu
                        self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                    self.reward = self.fc_layers[-1]
                
                elif config.reward_architecture == 'target':
                    self.target_idx = tf.where(self.target)
                    self.target_dish = tf.gather_nd(self.menu, self.target_idx)
                    self.diff = self.target_dish - (self.workplace + self.action_oh)
                    self.fc_layers = [self.diff]
                    for i, dim in enumerate(config.reward_target_fc_dim):
                        if i == len(config.reward_target_fc_dim) - 1:
                            act = None
                        else:
                            act = tf.nn.leaky_relu
                        self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                    self.reward = self.fc_layers[-1]

                else:
                    raise Exception('Wrong reward_architecture type')
                
                assert(self.reward.get_shape()[-1] == 1)

                self.reward = tf.reshape(self.reward, [-1])
                self.reward_spvs = tf.placeholder(tf.float32, [None])
                self.succeed_imbalance_mask = tf.where(
                    tf.equal(self.reward_spvs, tf.constant(config.succeed_reward, dtype = tf.float32)), 
                    tf.fill([tf.shape(self.reward_spvs)[0]], value = float(config.succeed_reward_bias)), 
                    tf.ones([tf.shape(self.reward_spvs)[0]])
                )
                self.fail_imbalance_mask = tf.where(
                    tf.equal(self.reward_spvs, tf.constant(config.fail_reward, dtype = tf.float32)), 
                    tf.fill([tf.shape(self.reward_spvs)[0]], value = float(config.fail_reward_bias)), 
                    tf.ones([tf.shape(self.reward_spvs)[0]])
                )
                self.reward_imbalance_mask = self.succeed_imbalance_mask + self.fail_imbalance_mask 
                self.l1_diff = tf.abs(self.reward - self.reward_spvs)
                self.loss = tf.reduce_mean(self.reward_imbalance_mask * tf.square(self.l1_diff))
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)


    def get_reward(self, menu, workplace, target, action):
        if self.config.reward_architecture == 'adhoc':
            action_oh = self.identity[action]
            target_idx = np.where(target)
            target_dish = menu[target_idx[0], target_idx[1]]
            diff = target_dish - (workplace + action_oh)
            fail_idx = np.where(diff < 0)[0]
            succeed_idx = np.where(np.mean(diff == 0, axis = 1) == 1)[0]
            assert((np.intersect1d(fail_idx, succeed_idx)).size == 0)
            reward = np.full(len(menu), self.config.step_reward)
            reward[fail_idx] = self.config.fail_reward
            reward[succeed_idx] = self.config.succeed_reward

        else:
            reward = self.sess.run(
                self.reward,
                feed_dict = {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.target: target, 
                    self.action: action
                }
            )

        return reward


    def train_reward(self, menu, workplace, target, action, reward_spvs):
        if self.config.reward_architecture == 'adhoc':
            action_oh = self.identity[action]
            target_idx = np.where(target)
            target_dish = menu[target_idx[0], target_idx[1]]
            diff = target_dish - (workplace + action_oh)
            fail_idx = np.where(diff < 0)[0]
            succeed_idx = np.where(np.mean(diff == 0, axis = 1) == 1)[0]
            assert((np.intersect1d(fail_idx, succeed_idx)).size == 0)
            reward = np.full(len(menu), self.config.step_reward)
            reward[fail_idx] = self.config.fail_reward
            reward[succeed_idx] = self.config.succeed_reward

            loss = np.nan
            l1_diff = np.abs(reward - reward_spvs)
        
        else:
            loss, l1_diff, reward, _ = self.sess.run(
                [self.loss, self.l1_diff, self.reward, self.train_op],
                feed_dict = {
                    self.menu: menu, 
                    self.workplace: workplace, 
                    self.target: target, 
                    self.action: action, 
                    self.reward_spvs: reward_spvs
                }
            )

        return loss, l1_diff, reward

