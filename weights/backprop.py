import tensorflow as tf
from tensorflow.keras import optimizers

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
print("Weights: ", W)

def loss():
    y_out = tf.multiply(tf.add(tf.multiply(x, W[1]), tf.multiply(x1, W[2])), W[0])
    return tf.reduce_sum(tf.square(target - y_out))

l = loss()
print("Loss: ", l)

steps = 10
for i in range(steps):
    train_step = optimizers.Adam(learning_rate=0.1).minimize(loss, W, tape=tf.GradientTape())
    print("step ", i, "- W:", W)
















