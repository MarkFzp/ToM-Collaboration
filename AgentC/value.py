import tensorflow as tf
import numpy as np

class ValueNetwork:
    def __init__(self, config, state_actions, history_input,
                 agent1_tensor = None, agent2_tensor = None, other_agent_belief_tensor = None):
        self.config_ = config
        self.state_actions_ = state_actions # Batch_Size X Sequence_len X num_actions X dim
        self.history_input_ = history_input # Batch_Size X Sequence_len X dim_1
        # Batch_Size X dim_1
        self.history_state_input_ = tf.placeholder(tf.float32, shape = [self.history_input_.shape[0], self.config_.history_dim])
        self.agent1_tensor_ = agent1_tensor # Batch_Size X Sequence_len X dim
        self.agent2_tensor_ = agent2_tensor # Batch_Size X Sequence_len X dim
        self.other_agent_belief_tensor_ = other_agent_belief_tensor # Batch_Size X Sequence_len X num_action X dim
        self.num_actions_ = int(self.state_actions_.shape[2])

        self.history_encoder_ = tf.keras.layers.CuDNNGRU(self.config_.history_dim, stateful = False,
                                                    return_sequences = True, return_state = True)
        # Batch_Size X Sequence_len X dim, Batch_Size X dim
        self.history_encoding_, self.history_states_ = self.history_encoder_(self.history_input_, self.history_state_input_)
        
        self.components_ = [self.state_actions_]
        #self.components_.append(tf.tile(tf.expand_dims(self.history_encoding_, 2), [1, 1, self.num_actions_, 1]))
        if self.agent1_tensor_ is not None:
            self.components_.append(tf.tile(tf.expand_dims(self.agent1_tensor_, 2), [1, 1, self.num_actions_, 1]))
        if self.agent2_tensor_ is not None:
            self.components_.append(tf.tile(tf.expand_dims(self.agent2_tensor_, 2), [1, 1, self.num_actions_, 1]))
        if self.other_agent_belief_tensor_ is not None:
            self.components_.append(self.other_agent_belief_tensor_)
        self.components_concat_ = tf.concat(self.components_, axis = 3) # Batch_Size X Sequence_len X num_actions X dim

        self.feat_tensors_ = [self.components_concat_]
        for out_dim in self.config_.fc_layer_info:
            fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size = 1, strides = 1,
                                  padding = 'VALID', activation = tf.nn.leaky_relu,
                                  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
            self.feat_tensors_.append(fc)
        
        # Batch_Size X Sequence_len X num_actions
        self.values_ = tf.squeeze(tf.layers.conv2d(self.feat_tensors_[-1], 1, kernel_size = 1, strides = 1,
                                        padding = 'VALID', activation = None,
                                        kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2)))
        

def main():
    from easydict import EasyDict as edict
    config = edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10]})
    Q_net = ValueNetwork(config, tf.random.uniform(shape = [4, 5, 9, 7]), tf.random.uniform(shape = [4, 5, 20]))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    #Q_net.history_encoder_.reset_states(np.zeros((4, 10)))
    [he, hs, values] = sess.run([Q_net.history_encoding_, Q_net.history_states_, Q_net.values_],
                                feed_dict = {Q_net.history_state_input_: np.zeros((4, 10))})
    print(he.shape) # outputs
    print(hs.shape) # last states
    print(values.shape)

if __name__ == '__main__':
    main()