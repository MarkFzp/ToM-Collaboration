import tensorflow as tf 
import numpy as np 

A = tf.placeholder(tf.complex64, [None, None])
B = tf.placeholder(tf.complex64, [None, None])
input = tf.linalg.matmul(tf.linalg.inv(tf.transpose(A)), tf.transpose(B))
e, v = tf.linalg.eigh(input)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

A_np = None
B_np = None
A_np_64 = A_np.astype(np.complex64)
B_np_64 = B_np.astype(np.complex64)
del(A_np)
del(B_np)
e_np, v_np = sess.run([e, v], feed_dict = {A: A_np_64, B: B_np_64})
np.save('eigenvalues', e_np)
np.save('eigenvectors', v_np)
