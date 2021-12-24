import tensorflow as tf
import numpy as np

class RobotBelief:
    '''
    bayesian belief
    '''
    def __init__(self, num_goals, num_hactions, num_ractions):

        self.num_goals = num_goals
        self.num_hactions = num_hactions
        self.num_ractions = num_ractions

        self.new_belief_ph = tf.placeholder(tf.float32, [1, num_goals], 'new_belief')

        self._belief = tf.get_variable('goals', initializer=
            tf.nn.softmax(tf.ones([1, num_goals])))

        self.init_opt = tf.initializers.variables([self._belief])

        self.update_opt = self._update_opt()

    def _update_opt(self):
        return self._belief.assign(self.new_belief_ph)

    def init_belief(self, sess):
        sess.run(self.init_opt)

    def update_belief(self, sess, belief):
        sess.run(self.update_opt, {self.new_belief_ph: belief})

    def fixed_point_belief(self, belief, hprobs_per_ha_ra):

        def update_belief(b, h):
            b = b * h
            s = np.sum(b, axis=-1, keepdims=True)
            s[s==0] = 1
            b = b/s
            return b

        belief_ = update_belief(belief, hprobs_per_ha_ra)
        s = 0

        while np.max(np.abs(belief_-belief)) > 1e-3 and s<1000:
            belief = belief_

            belief_ = update_belief(belief, hprobs_per_ha_ra)
            s+=1

        return belief_

    def get_belief(self,sess):

        return sess.run(self._belief)
