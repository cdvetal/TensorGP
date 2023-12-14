import tensorflow as tf
import sys

w1 = tf.Variable(1.0, trainable = True)
w2 = tf.Variable(1.0, trainable = True)
w3 = tf.Variable(1.0, trainable = True)
W = [w1, w2, w3]

size = sys.getsizeof(W)
print("Size of list of vars: ", size)

W = tf.Variable(tf.random.uniform([3], minval=0.0, maxval=1.0), trainable = True)
size = sys.getsizeof(W)
print("Size of var that is list: ", size)

print("Change third var: ", W.assign_add())
