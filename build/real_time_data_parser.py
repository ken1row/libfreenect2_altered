#!/usr/bin/env python

    
import sys
import os
import numpy as np
import time

def generator():
    x = np.linspace(1, 10, 100)

    np.save('x.npy', x)    
    for i in np.linspace(1, 3.14, 10):
        y = np.sin(x+i) * i
        np.save('y.npy', y)
        time.sleep(1)
        
        
if __name__ == '__main__':
    generator()