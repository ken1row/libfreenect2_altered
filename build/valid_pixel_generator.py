# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:31:29 2016

@author: tanaka
"""

import cv2
import numpy as np

def generate():
    img = np.zeros((424, 512, 3))
    for y in range(101, 210, 2):
        for x in range(1, 512, 1):
            img[y, x] = [0, 0, 255]
    for y in range(214, 396, 2):
        for x in range(1, 512, 1):
            img[y, x] = [0, 0, 255]
    cv2.imwrite('valid_pixels.png', img)
    
generate()