import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

#from tensorflow.examples.tutorials.mnist import input_data
#data = tf.keras.datasets.fashion_mnist.load_data() #= input_data.read_data_sets('data/fashion')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)

x_train = x_train/255.0      
x_test = x_test/255.0

x_train = x_train.reshape(x_train.shape[0],*(28,28,1))
x_test = x_test.reshape(x_test.shape[0],*(28,28,1))
x_validate = x_validate.reshape(x_validate.shape[0],*(28,28,1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,activation = 'softmax') 
])

model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001),metrics =['accuracy'])

import time
last_engine_time = time.time()

history = model.fit(
    x_train,
    y_train,
    batch_size=1000,
    epochs=10,
    verbose=2,
    validation_data=(x_validate,y_validate),
)
elapsed_engine_time=0
t_ = time.time()
elapsed_engine_time += t_ - last_engine_time
print("time",elapsed_engine_time)
last_engine_time = time.time()

score = model.evaluate(x_test,y_test,verbose=2)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy : {:.4f}'.format(score[1]))
t_ = time.time()
elapsed_engine_time += t_ - last_engine_time
print("time", elapsed_engine_time)

