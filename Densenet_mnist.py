# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:31:00 2021

@author: hp
"""


from keras.datasets import cifar10
from keras.utils import to_categorical, plot_model
import tensorflow 

import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings


import numpy as np


(x_train, y_train),(x_test, y_test)=cifar10.load_data()
num_labels=len(np.unique(y_train))

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
batch_size=128
# network parameters
image_size = x_train.shape[1]
# resize and normalize
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
# network parameters
warnings.filterwarnings("ignore")

print("Tensorflow-version:", tensorflow.__version__)

model_d=DenseNet121(weights='imagenet',include_top=False, input_shape=(32, 32, 3)) 

x=model_d.output

x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu')(x) 
x= Dense(512,activation='relu')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)

preds=Dense(10,activation='softmax')(x) #FC-layer

model=Model(inputs=model_d.input,outputs=preds)
for layer in model.layers[:-8]:
    layer.trainable=False
    
for layer in model.layers[-8:]:
    layer.trainable=True
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
(x_train, y_train),(x_test, y_test)=cifar10.load_data()
num_labels=len(np.unique(y_train))

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# network parameters
image_size = x_train.shape[1]
# resize and normalize
"""x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255"""
model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))