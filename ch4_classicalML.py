# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 05:29:48 2021

@author: CHAINZ
"""


from sklearn import datasets as skds
import sklearn.model_selection as skms
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder


X,y=skds.make_regression(n_samples=200, n_features=1, 
                         n_informative=1, n_targets=1,
                         noise=20)
if (y.ndim == 1):
    y = y.reshape(len(y),1)
X_train, X_test, y_train, y_test = skms.train_test_split(X, y,
                                                         test_size=.4,
                                                         random_state=123)
num_outputs=y_train.shape[1]
num_inputs=X_train.shape[1]
x_tensor=tf.placeholder(dtype=tf.float32, 
                        shape=[None,num_inputs],
                        name='x')
y_tensor=tf.placeholder(dtype=tf.float32,
                        shape=[None, num_outputs],
                        name='y')
w = tf.Variable(tf.zeros([num_inputs, num_outputs]),
                dtype=tf.float32, name='w')
b= tf.Variable(tf.zeros([num_outputs]),
               dtype=tf.float32,
               name='b')
model=tf.matmul(x_tensor,w)+b
lasso_param = tf.Variable(0.8, dtype=tf.float32)
lasso_loss = tf.reduce_mean(tf.abs(w)) * lasso_param
loss = tf.reduce_mean(tf.square(model - y_tensor)) + lasso_loss
# loss= tf.reduce_mean(tf.square(model-y_tensor))
mse = tf.reduce_mean(tf.square(model - y_tensor))
y_mean = tf.reduce_mean(y_tensor)
total_error = tf.reduce_sum(tf.square(y_tensor - y_mean))
unexplained_error = tf.reduce_sum(tf.square(y_tensor - model))
rs = 1 - tf.div(unexplained_error, total_error)
learning_rate=0.001
optimizer= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
num_epochs=1500
w_hat=0
b_hat=0
loss_epochs = np.empty(shape=[num_epochs],dtype=float)
mse_epochs = np.empty(shape=[num_epochs],dtype=float)
rs_epochs = np.empty(shape=[num_epochs],dtype=float)
mse_score = 0
rs_score = 0

with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    
    for epoch in range(num_epochs):
        tfs.run(optimizer, feed_dict={x_tensor: X_train, y_tensor: y_train})
        loss_val = tfs.run(loss,feed_dict={x_tensor: X_train, y_tensor:y_train})
        loss_epochs[epoch] = loss_val
        mse_score = tfs.run(mse,feed_dict={x_tensor: X_test, y_tensor: y_test})
        mse_epochs[epoch] = mse_score
        rs_score = tfs.run(rs,feed_dict={x_tensor: X_test, y_tensor: y_test})
        rs_epochs[epoch] = rs_score
    w_hat,b_hat = tfs.run([w,b])
    w_hat = w_hat.reshape(1)
print('model : Y = {0:.8f} X + {1:.8f}'.format(w_hat[0],b_hat[0]))
print('For test data : MSE = {0:.8f}, R2 = {1:.8f} '.format(mse_score,rs_score))


plt.figure(figsize=(14,8))
plt.title('Original Data and Trained Model')
x_plot = [np.min(X)-1,np.max(X)+1]
y_plot = w_hat*x_plot+b_hat
plt.axis([x_plot[0],x_plot[1],y_plot[0],y_plot[1]])
plt.plot(X,y,'b.',label='Original Data')
plt.plot(x_plot,y_plot,'r-',label='Trained Model')
plt.legend()
plt.show()


plt.figure(figsize=(14,8))
plt.axis([0,num_epochs,0,np.max(loss_epochs)])
plt.plot(loss_epochs, label='Loss on X_train')
plt.title('Loss in Iterations')
plt.xlabel('# Epoch')
plt.ylabel('MSE')
plt.axis([0,num_epochs,0,np.max(mse_epochs)])
plt.plot(mse_epochs, label='MSE on X_test')
plt.xlabel('# Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()



X, y = skds.make_classification(n_samples=200,
                                n_features=2,
                                n_informative=2,
                                n_redundant=0,
                                n_repeated=0,
                                n_classes=2,
                                n_clusters_per_class=1)
if (y.ndim == 1):
    y = y.reshape(-1,1)
# print(y)
ohe=OneHotEncoder()
y=ohe.fit_transform(y).toarray()
# y=np.eye(num_outputs)[y]
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, 
                                                         test_size=.4,
                                                         random_state=42)
num_outputs = y_train.shape[1]
num_inputs = X_train.shape[1]
learning_rate = 0.001
# input images
x = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs], name="x")
# output labels
y = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name="y")
# model paramteres
w = tf.Variable(tf.zeros([num_inputs,num_outputs]), name="w")
b = tf.Variable(tf.zeros([num_outputs]), name="b")
model = tf.nn.sigmoid(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum((y * tf.math.log(model)) + ((1 - y) * tf.math.log(1 - model)), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
num_epochs = 1

with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    
    for epoch in range(num_epochs):
        tfs.run(optimizer, feed_dict={x: X_train, y: y_train})
        y_pred = tfs.run(tf.argmax(model, 1), feed_dict={x: X_test})
        y_orig = tfs.run(tf.argmax(y, 1), feed_dict={y: y_test})
        preds_check = tf.equal(y_pred, y_orig)
        accuracy_op = tf.reduce_mean(tf.cast(preds_check, tf.float32))
        accuracy_score = tfs.run(accuracy_op)
        print("epoch {0:04d} accuracy={1:.8f}".format(
            epoch, accuracy_score))
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_orig)
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_pred)
        plt.title('Predicted')
        plt.show()
        
        

