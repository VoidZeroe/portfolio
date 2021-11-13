# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:55:30 2021

@author: CHAINZ
"""

import numpy as np
from matplotlib import pyplot as plt
def countour_function(x,y):
    return np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
x=np.linspace(0,5,50)
y=np.linspace(0,5,50)
X,Y=np.meshgrid(x,y)
print(y,Y[1])
z=countour_function(X,Y)
plt.contour(X,Y,z,20,cmap='RdGy')