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


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from the_Perceptron import plot_decision_regions
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def Augment(x):
    return([[[i,j,i*j] for i in x] for j in x])
    
mms = MinMaxScaler()




# SVC achieved a 1.0 accuracy score on my test set. 


# SVC achieved a 1.0 accuracy score on my test set. 
df=pd.read_csv(r'C:\Users\hp\deep learning\dsn\Train.csv')
y=df.iloc[ : ,26].values
df=df.drop(['ID',
            'Date_Customer',
            # 'Cmp1Accepted',
            # 'Cmp2Accepted',
            # 'Cmp3Accepted',
            # 'Cmp4Accepted',
            # 'Cmp5Accepted',
            # 'Any_Complain',
            # 'Recency',
            'Response'], axis=1)
# print(df.shape)
# print(np.unique(df.iloc[ : , 25].values))

# print(x,Y)
# df= df.dropna(axis=1)
# df=df[df.columns[1:]]
# print(df.isnull().sum())

df=pd.get_dummies(df)
x=df.iloc[ : , : ].values
imr = SimpleImputer(strategy='most_frequent')
imr = imr.fit(x)
x = imr.transform(x)

print(np.sum(y))
print('Number of class 1 samples before:',
      x[y == 1].shape[0])
x_up, y_up= resample(x[y == 1],
               y[y == 1],
               replace=True,
               n_samples=x[y == 0].shape[0],
               random_state=123)
x = np.vstack((x[y == 0], x_up))
y = np.hstack((y[y == 0], y_up))
mms=MinMaxScaler()
x=mms.fit_transform(x)
X=[]
for i in x:
    new=Augment(i)
    X.append(new)
x=np.array(X)
print(x.shape)

x_train, x_test, y_train, y_test=train_test_split(x, y,
                 test_size=0.3,
                 random_state=0,
                 stratify=y)


# (x_train, y_train),(x_test, y_test)=cifar10.load_data()
num_labels=len(np.unique(y_train))

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
batch_size=128
# network parameters
image_size = x_train.shape[1]
# resize and normalize
# x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
# x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') 
# network parameters
warnings.filterwarnings("ignore")

print("Tensorflow-version:", tensorflow.__version__)

print(x_train.shape)
model_d=DenseNet121(weights='imagenet',include_top=False, input_shape=(35, 35, 3)) 

x=model_d.output

x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu')(x) 
x= Dense(512,activation='relu')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)

preds=Dense(2,activation='softmax')(x) #FC-layer

model=Model(inputs=model_d.input,outputs=preds)
for layer in model.layers[:-8]:
    layer.trainable=False
    
for layer in model.layers[-8:]:
    layer.trainable=True
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=500, batch_size=batch_size)
_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))