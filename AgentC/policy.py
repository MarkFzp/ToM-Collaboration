import tensorflow as tf
import numpy as np

class PolicyNetwork:
    def __init__(self, config, num_actions, agent_tensor, history_input):
        self.config_ = config
        self.num_actions_ = num_actions
        self.agent_tensor_ = agent_tensor # Batch_Size X Sequence_len X dim
        self.history_input_ = history_input # Batch_Size X Sequence_len X dim_1
        # Batch_Size X dim_1
        self.history_state_input_ = tf.placeholder(tf.float32, shape = [self.history_input_.shape[0], self.config_.history_dim])

        self.history_encoder_ = tf.keras.layers.CuDNNGRU(self.config_.history_dim, stateful = False,
                                                    return_sequences = True, return_state = True)
        # Batch_Size X Sequence_len X dim, Batch_Size X dim
        self.history_encoding_, self.history_states_ = self.history_encoder_(self.history_input_, self.history_state_input_)

        # Batch_Size X Sequence_len X dim
        self.components_concat_ = tf.concat([self.agent_tensor_, self.history_encoding_], axis = 2)

        self.feat_tensors_ = [tf.expand_dims(self.components_concat_, 2)]
        for out_dim in self.config_.fc_layer_info:
            fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size = 1, strides = 1,
                                  padding = 'VALID', activation = tf.nn.leaky_relu,
                                  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
            self.feat_tensors_.append(fc)
        
        # Batch_Size X Sequence_len X num_actions
        self.action_prob_unscale_ = tf.squeeze(tf.layers.conv2d(self.feat_tensors_[-1], self.num_actions_, kernel_size = 1,
                                                                strides = 1, padding = 'VALID', activation = None,# softmax done in loss layer
                                                                kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2)), 2)
        self.action_prob_ = tf.nn.softmax(self.action_prob_unscale_)

def main():
    from easydict import EasyDict as edict
    config = edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10], 'num_actions': 9})
    P_net = PolicyNetwork(config, tf.random.uniform(shape = [4, 1, 7]),
                          tf.random.uniform(shape = [4, 1, 9]),
                          tf.placeholder(tf.float32, shape = [4, None, 11]))#tf.random.uniform(shape = [4, 5, 11]))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    P_net.history_encoder_.reset_states(np.zeros((4, 10)))
    test_input1 = np.random.uniform(size = [4, 1, 11])
    test_input2 = np.random.uniform(size = [4, 10])
    [he, hs, action_prob] = sess.run([P_net.history_encoding_, P_net.history_states_, P_net.action_prob_],
                                     feed_dict = {P_net.history_input_: test_input1, P_net.history_state_input_: test_input2})
    print(he.shape) # outputs
    print(hs.shape) # last states
    print(action_prob.shape)
    print(he, hs)
    print('###################3')
    #P_net.history_encoder_.reset_states(hs)
    [he, hs, action_prob] = sess.run([P_net.history_encoding_, P_net.history_states_, P_net.action_prob_],
                                     feed_dict = {P_net.history_input_: test_input1, P_net.history_state_input_: hs})
    print(he, hs)
if __name__ == '__main__':
    main()