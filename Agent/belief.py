import tensorflow as tf
import numpy as np

class BeliefNetwork:
    def __init__(self, config, background, state, history_input, initial_belief):
        self.config_ = config
        self.background_ = background # Batch_size X 1 X belief_dim X dim
        self.state_ = state # Batch_Size X Sequence_len X dim
        self.history_input_ = history_input # Batch_Size X Sequence_len X dim_1
        self.initial_belief_ = initial_belief # Batch_Size X 1 X belief_dim
        # Batch_Size X dim_1
        self.history_state_input_ = tf.placeholder(tf.float32, shape = [self.history_input_.shape[0], self.config_.history_dim])
        self.belief_dim_ = int(self.initial_belief_.shape[2])

        self.history_encoder_ = tf.keras.layers.CuDNNGRU(self.config_.history_dim, stateful = False,
                                                    return_sequences = True, return_state = True)
        # Batch_Size X Sequence_len X dim, Batch_Size X dim
        # self.history_encoding_, self.history_states_ = self.history_encoder_(tf.concat([self.history_input_, self.state_], axis = 2),
        #                                                                      self.history_state_input_)
        self.history_encoding_, self.history_states_ = self.history_encoder_(self.history_input_, self.history_state_input_)
        self.components_ = [self.background_ * tf.ones_like(tf.expand_dims(self.state_, 2))[:, :, :, 0: 1]]
        self.components_.append(tf.tile(tf.expand_dims(self.history_encoding_, 2), [1, 1, self.belief_dim_, 1]))
        #self.components_.append(tf.tile(tf.expand_dims(self.state_, 2), [1, 1, self.belief_dim_, 1]))
        
        self.components_concat_ = tf.concat(self.components_, axis = 3) # Batch_Size X Sequence_len X belief_dim X dim
        
        self.feat_tensors_ = [self.components_concat_]
        for out_dim in self.config_.fc_layer_info:
            fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size = 1, strides = 1,
                                padding = 'VALID', activation = tf.nn.leaky_relu,
                                kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
            self.feat_tensors_.append(fc)
        # Batch_Size X Sequence_len X belief_dim X num_actions
        self.likelihood_pre_ = tf.layers.conv2d(self.feat_tensors_[-1], self.config_.num_actions, kernel_size = 1,
                                            strides = 1, padding = 'VALID', activation = None,
                                            kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
        self.likelihood_ = tf.nn.softmax(self.likelihood_pre_) + 1e-9
        # Batch_Size X Sequence_len X belief_dim X num_actions
        self.new_belief_pre_ = self.likelihood_ * tf.expand_dims(self.initial_belief_, -1)
        # Batch_Size X Sequence_len X num_actions X belief_dim
        self.new_belief_ = tf.transpose(self.new_belief_pre_ / tf.reduce_sum(self.new_belief_pre_, axis = 2, keepdims = True), [0, 1, 3, 2])

def main():
    from easydict import EasyDict as edict
    config = edict({'history_dim': 20, 'fc_layer_info': [30, 20], 'num_actions': 10})
    B_net = BeliefNetwork(config, tf.random.uniform(shape = [4, 1, 4, 10]), tf.random.uniform(shape = [4, 5, 10]),
                          tf.random.uniform(shape = [4, 5, 10]), tf.random.uniform(shape = [4, 1, 4]))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    [ib, likelihood, nb] = sess.run([B_net.initial_belief_, B_net.likelihood_, B_net.new_belief_],
                             feed_dict = {B_net.history_state_input_: np.zeros((4, 20))})
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()