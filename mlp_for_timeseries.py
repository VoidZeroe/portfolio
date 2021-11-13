# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:59:34 2021

@author: CHAINZ
"""


from keras.model import Sequential
from keras.model import Dense
from keras.optimizers import SGD
from tensorflow.examples.tutorials import input_data
import math

mnist_home = r'C:/Users/CHAINZ/APLPHASE1/mnist_home'
mnist = input_data.read_data_sets(mnist_home, one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels
num_outputs = 10 # 0-9 digits
num_inputs = 784 # total pixels

num_layers=2
num_neurons=[8,8]
n_epochs=50
batch_size=2
n_x=2

model=Sequential()
model.add(Dense(num_neurons[0], activation='relu', input_shape=(n_x,)))
model.add(Dense(num_neurons[1], activation='relu'))
model.add(Dense(unit=1))
model.summary()

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train, 
          batch_size= batch_size,
          epochs=n_epochs)
score = model.evaluate(X_test, Y_test)
print('\nTest mse:', score)
print('Test rmse:', math.sqrt(score))
