# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:06:01 2021

@author: hp
"""



import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from the_Perceptron import plot_decision_regions
from sklearn.utils import resample
from sklearn.impute import SimpleImputer

df=pd.read_csv(r'C:\Users\hp\deep learning\dsn\Train.csv')
y=df.iloc[ : ,26].values
# df=df.drop(['Date_Customer',
#             'ID',
#             'Cmp1Accepted',
#             'Cmp2Accepted',
#             'Cmp3Accepted',
#             'Cmp4Accepted',
#             'Cmp5Accepted'], axis=1)
# print(df.head())
# print(np.unique(df.iloc[ : , 25].values))

# print(x,Y)
# df= df.dropna(axis=1)
# df=df[df.columns[1:]]
# print(df.isnull().sum())

df=pd.get_dummies(df)
# for i in (df.columns):
    # print(i)
# ohe=OneHotEncoder(categories='auto')
# df=ohe.fit_transform(df.iloc[ : ,2:3].values)


print(df.shape)
x=df.iloc[ : ,0:23].values
imr = SimpleImputer()
imr = imr.fit(x)
x = imr.transform(x)
lr = LogisticRegression()
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
print('Number of class 1 samples after:',
      x.shape[0])

x_train, x_test, y_train, y_test=train_test_split(x, y,
                                                  test_size=0.3,
                                                  random_state=0,
                                                  stratify=y)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


# (x_train, y_train),(x_test, y_test)=mnist.load_data()
num_label=len(np.unique(y_train))

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
# # image_size=x_train.shape[1]
input_size = x_train.shape[1]
# x_train=np.reshape(x_train,[-1,input_size])
# x_train = x_train.astype('float32') / 255
# x_test = np.reshape(x_test, [-1, input_size])
# x_test = x_test.astype('float32') / 255
# network parameters
batch_size = 128
hidden_units = 20
dropout = 0.7

model=Sequential()
model.add(Dense(hidden_units, input_dim= input_size))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout))
# model.add(Dense(hidden_units))
# model.add(Activation('relu'))
# model.add(Dropout(dropout))
model.add(Dense(num_label))
model.add(Activation('sigmoid'))
model.summary()

# plot_model(model, to_file='mlp-mnist.png', show_shapes=True)
# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the network
model.fit(x_train, y_train, epochs=5, batch_size=batch_size)
# validate the model on test dataset to determine generalization
_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size)

print(100.0 * acc)
pred=model.predict(x_test).T
print(pred)
print(np.sum(pred[1]))
