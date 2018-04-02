import tensorflow as tf
import numpy as np

matrix = tf.placeholder(tf.float32, shape = (3, 3))

norms = tf.reduce_sum(matrix*matrix, 1)

norms_column = tf.reshape(norms, [-1, 1])

euclid_distance = norms_column - 2*tf.matmul(matrix, tf.transpose(matrix)) + norms


input_matrix = np.array(
    [[ 1, 1, 1 ],
     [ 0, 1, 1 ],
     [ 0, 0, 1 ],
     ],
    dtype = 'float32')


sess = tf.Session()
print(sess.run(euclid_distance, feed_dict = {matrix : input_matrix}))