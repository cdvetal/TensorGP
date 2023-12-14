import tensorflow as tf
from tensorflow.keras import optimizers
from contextlib import ExitStack

diff = True
length = 10
target = tf.constant(0.0, shape=(length), dtype=tf.float32)

x = tf.constant([i for i in range(length)], dtype=tf.float32)
x1 = tf.constant(1.0, shape=(length), dtype=tf.float32)

print("Target: ", target)
print("X: ", x)
print("X1: ", x1, "\n")

#W = tf.Variable([1.0, 1.0, 1.0])
w1 = tf.Variable(1.0, trainable = True)
w2 = tf.Variable(1.0, trainable = True)
w3 = tf.Variable(1.0, trainable = True)
W = [w1, w2, w3]
v1 = tf.Variable(0.5, trainable = True)
v2 = tf.Variable(0.5, trainable = True)
v3 = tf.Variable(0.5, trainable = True)
V = [v1, v2, v3]
print("Weights: ", W)
print("Weights: ", V)

with ExitStack() as stack:
    if diff:
        tape = stack.enter_context(tf.GradientTape())
    y_out = tf.multiply(tf.add(tf.multiply(x, W[1]), tf.multiply(x1, W[2])), W[0])
    loss = tf.reduce_sum(tf.square(target - y_out))
    grads = tape.gradient(loss, W)

with tf.GradientTape() as tape:
    y_out1 = tf.multiply(tf.add(tf.multiply(x, V[1]), tf.multiply(x1, V[2])), V[0])
    loss1 = tf.reduce_sum(tf.square(target - y_out1))
    grads1 = tape.gradient(loss1, V)

print("grads: ", grads)

steps = 3
for i in range(steps):

    train_step = optimizers.Adam(learning_rate=0.1).apply_gradients(zip(grads, W))
    #train_step = optimizers.Adam(learning_rate=0.1).apply_gradients(zip(grads1, V))

    print("step ", i, "- W:", W)

















