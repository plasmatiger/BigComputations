import tensorflow as tf
import numpy as np

m = tf.placeholder(tf.int32)

f = tf.placeholder(tf.int32)
#matrix = tf.placeholder(tf.float32, shape = (m, f))

matrix = tf.placeholder(tf.float32, shape = (3, 3))

normalized_matrix = tf.nn.l2_normalize(matrix, dim =1) #row wise normalization

mat_dot = tf.matmul(normalized_matrix, tf.transpose(normalized_matrix))

cosine_distance = 1 - mat_dot


input_matrix = np.array(
    [[ 1, 1, 1 ],
     [ 1, 1, 1 ],
     [ 1, 1, 1 ],
     ],
    dtype = 'float32')

rows = 3

columns =3
 
print("input_mat:")
print( input_matrix)

sess = tf.Session()

print(sess.run(cosine_distance, feed_dict = {matrix : input_matrix, m : rows, f : columns}))