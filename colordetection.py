# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:13:29 2021

@author: hp
"""


import cv2
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES,hex_to_rgb


img_path=r'C:\Users\hp\Pictures\Screenshots\test.png'

def convert_rgb_to_names(rgb_tuple):
    css3_db=CSS3_HEX_TO_NAMES
    names=[]
    rgb_values=[]
    for color_hex,color_name in css3_db.items():
        names.append(color_name)
        
        rgb_values.append(hex_to_rgb(color_hex))
    kdt_db= KDTree(rgb_values)
    distance, index= kdt_db.query(rgb_tuple)
    return(names[index])
print(convert_rgb_to_names((0,900,0))) 
