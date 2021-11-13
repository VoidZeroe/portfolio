# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 03:53:32 2021

@author: CHAINZ
"""
import tensorflow as tf



def RNN_timeseries(n_x,n_y,n_x_vars,n_y_vars):
    state_size=4
    n_epochs=100
    n_timesteps=n_x
    learning_rate=0.1    