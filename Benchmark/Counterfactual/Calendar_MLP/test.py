import tensorflow as tf
import numpy as np

def f(a, b, c):
    print(a + b + c)

f(
    1, 
    2, 
    3
)

init_state = tf.placeholder(tf.float32, shape=[None, 10])
input1 = tf.placeholder(tf.float32, shape=[None, None, 20])
# input2 = tf.placeholder(tf.float32, shape=[10, 10, 20])
cell = tf.keras.layers.GRU(10, stateful = False, return_sequences = True, return_state = True)
# out_seq, out_state = cell(input, initial_state=init_state)
out_seq1, out_state1 = cell(input1)
# out_seq2, out_state2 = cell(input2)


rand_in = np.random.randint(10, size=[10, 1, 20])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(rand_in)
print('step1')
out_state1_, out_seq1_ = sess.run([out_state1, out_seq1], feed_dict={input1: rand_in})
print(np.mean(out_state1_ == out_seq1_[:, -1, :]))
# print('step2')
# out_state2_ = sess.run(out_state2, feed_dict={input2: rand_in})
# print(out_state2_)
# print(np.mean(out_state1_ == out_state2_))
# # cell.reset_states()
# print(tf.trainable_variables())