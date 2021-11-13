# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 05:22:46 2021

@author: CHAINZ
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineSGD(object):
    
    
    def __init__(self,eta=0.01 , n_iter=10, random_state=None,shuffle=True):
    
        self.eta=eta
        self.n_iter= n_iter
        self.random_state=random_state
        self.w_initialized=False
        self.shuffle=shuffle
    
    def fit(self, X,Y):
        
        self._initialize_weights(X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            if self.shuffle:
                X,Y=self._shuffle(X,Y)
            cost=[]
            for xi, target in zip(X,Y):
                cost.append(self._update_weights(xi, target))
            avg_cost=sum(cost)/len(Y)
            self.cost_.append(avg_cost)
        
        return self
    
    def partial_fit(self,X,Y):
        
        if not self.w_initialized:
            self._initialize_weigths(X.shape[1])
        if Y.ravel().shape[0]>1:
            for xi,target in zip(X,Y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,Y)
        return self
    
    def _shuffle(self,X,Y):
        r=self.rgen.permutation(len(Y))
        return X[r],Y[r]
    
    def _initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=1 + m)
        self.w_initialized = True
  
    def _update_weights(self,xi,target):
        output=self.activation(self.net_input(xi))
        error=(target-output)
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+=self.eta*error
        cost=0.5*error**2
        return cost
        
    def net_input(self,X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return(np.where(self.activation(self.net_input(X))
                        >=0.0,1,-1))


class AdalineGD(object):
    
    
    def __init__(self,eta=0.01 , n_iter=50, random_state=1):
    
        self.eta=eta
        self.n_iter= n_iter
        self.random_state=random_state
    
    
    def fit(self, X,Y):
        
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_=[]
        self.cost_=[]
        
        
        for i in range(self.n_iter):
            net_input=self.net_input(X)
            output=self.activation(net_input)
            errors=Y-output
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2
            self.cost_.append(cost)
            
        return self
    
    def net_input(self,X):
        return(np.dot(X, self.w_[1: ])+self.w_[0])
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return(np.where(self.activation(self.net_input(X))>=0.0,1,-1))



class Perceptron:
    
    
    def __init__(self,eta=0.01 , n_iter=500, random_state=1):
    
        self.eta=eta
        self.n_iter= n_iter
        self.random_state=random_state
    
    
    def fit(self, X,Y):
        
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_=[]
        # print(self)
        
        for i in range(self.n_iter):
            errors=0
            
            for xi, target in zip(X,Y):
                update = self.eta *(target - self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                # print(self.w_,21)
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self,X):
        return(np.dot(X, self.w_[1: ])+self.w_[0])
    
    def predict(self,X):
        return(np.where(self.net_input(X)>=0.0,1,-1))
# df=pd.read_csv('https://archive.ics.uci.edu/ml/'
            # 'machine-learning-databases/iris/iris.data', 
            # header=None)


def load_data(url, output_file):

    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading data')

    except FileNotFoundError:
        print('File not found...downloading data')
        df = pd.read_csv(url, header=None)
        df.to_pickle(output_file)

    return df


def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers=('s', 'x','^','v','o')
    colors=('red','blue','green','lightgreen', 'cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max=X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max=X[:,1].min()-1, X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min,x2_max,resolution))
    # print([xx1,xx2],2)
    z= classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    
    Z=z.reshape(xx1.shape)
    # print(Z,3)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')



"""
df=load_data('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', 'iris_data')    
# print(df.tail(50))
Y=df.iloc[0:100,4].values
Y=np.where(Y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values
# print(X)
# plt.scatter(X[:50,0] , X[:50,1])
# plt.scatter(X[50:100,0],X[50:100,1])
# p lt.scatter(X[100:150,0],X[100:150,1])

# plt.show()
X_std=np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada = AdalineSGD(n_iter=15, eta=0.01,random_state=1)
ada.fit(X_std, Y)

# ppn=Perceptron(eta=0.1,n_iter=10)
# ppn.fit(X, Y)
# plot_decision_regions(X, Y, classifier=ppn)
# plt.show()
plot_decision_regions(X_std, Y, classifier=ada)
plt.show()
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
# plt.show()
plt.plot(range(1, len(ada.cost_) + 1),ada.cost_,marker='o')
plt.show()"""