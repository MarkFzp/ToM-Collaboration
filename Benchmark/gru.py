import tensorflow as tf
import numpy as np

class GRU:
    def __init__(self, hidden_dim: int, in_x_dim: int, out_y_dim: int or None, gpu: bool, sess: "tf.Session()"):
        self.hidden_dim = hidden_dim
        self.in_x_dim = in_x_dim
        self.out_y_dim = out_y_dim
        self.gpu = gpu
        self.sess = sess

        with tf.variable_scope('gru'):
            # batch_size * seq_len * in_x_dim
            self.in_x = tf.placeholder(tf.float32, [None, None, self.in_x_dim]) 
            self.in_state = tf.placeholder_with_default(tf.zeros([tf.shape(self.in_x)[0], self.hidden_dim]), [None, self.hidden_dim])
            if gpu:
                self.cell = tf.keras.layers.CuDNNGRU(self.hidden_dim, return_sequences = True, return_state = True)
            else:
                self.cell = tf.keras.layers.GRU(self.hidden_dim, return_sequences = True, return_state = True)
            self.out_seq, self.out_state = self.cell(self.in_x, initial_state = self.in_state)
            if self.out_y_dim is not None:
                self.out_seq = tf.layers.dense(self.out_seq, self.out_y_dim)
    
    
    def __call__(self, in_x: "np.array", in_state: "np.array" = None) -> "pair of np.array":
        if len(in_x.shape) != 3 or in_x.shape[-1] != self.in_x_dim:
            print('in_x.shape: ', in_x.shape)
            raise Exception('in_x dimension: batch_size * seq_len * in_x_dim')

        if in_state is None:
            out_seq, out_state = self.sess.run([self.out_seq, self.out_state], \
                feed_dict = {self.in_x: in_x})
        else:
            out_seq, out_state = self.sess.run([self.out_seq, self.out_state], \
                feed_dict = {self.in_x: in_x, self.in_state: in_state})
        
        return out_seq, out_state


if __name__ == "__main__":
    sess = tf.Session()
    gru = GRU(10, 20, None, True, sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    print(gru(np.random.randint(10, size=[5, 1, 20])))
