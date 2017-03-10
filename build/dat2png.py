#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from array import array
import struct

def read_float_file(filename='test.dat', kinect=False):
    '''
    Read binary file of double as a numpy array.
    
    Parameters
    ----------
    filename : str
        input binary file's path.
    kinect : bool, optional
        If True, returns grayscale image (OpenCV compatible) regarding the data is captured by Kinect v2, otherwise returns 1d array.
    '''
    in_file = open(filename, 'rb')
    float_array = array('d')
    float_array.fromstring(in_file.read())
    np_array = np.array(float_array)
    if kinect:
        return np_array.reshape((512,424)).T[::-1,::-1]
    else:
        return np_array
        
def read_depth_wise_data(filename='phase_depth_0.dat', load_depth=False, depth_file='depth_bins.dat'):
    data = read_float_file(filename)
    if load_depth:
        depth = read_float_file(depth_file)
        return depth, data
    return data
    
def read_int_file(filename='test.dat'):
    '''
    Read binary file of int as numpy array.
    '''
    in_file = open(filename, 'r')
    float_array = array('i')
    float_array.fromstring(in_file.read())
    np_array = np.array(float_array)
    return np_array
    
def convert_image_i(fin='raw0', extin='.dat', extout='.png'):
    data = read_int_file(fin+extin)
    data = data.reshape((512,424)).T[::-1,::-1]
    cv2.imwrite(fin+extout, np.uint16(data))
    
def convert_image(fin='raw0', extin='.dat', extout='.png', auto_range=False, max_rate=100., fixed_range=False, max_value=65520, print_minmax=False):
    try:
        data = read_float_file(fin+extin)
        if print_minmax:
            print 'Min and max:', np.amin(data), np.amax(data)
        data = data.reshape((512,424)).T[::-1,::-1]
        if auto_range:
            data = data - np.amin(data)
            data = np.minimum(data * 65535. / np.percentile(data, max_rate), 65535)
        if fixed_range:
            data = data - np.amin(data)
            data = np.minimum(data * 65535 / max_value, 65535)
        cv2.imwrite(fin+extout, np.uint16(data))
    except:
        pass
    
def convert_image_b(fin='raw0', extin='.dat', extout='.png', auto_range=False):
    try:
        data = read_float_file(fin+extin)
        data = data.reshape((1080,1920,3))[:,::-1, :]
        if auto_range:
            data = data - np.amin(data)
            data = data * 65535. / np.amax(data)
        cv2.imwrite(fin+extout, np.uint8(data))
    except:
        pass
    
if __name__ == '__main__':
    import subprocess
#    for i in range(9):
#        convert_image('raw' + str(i))#, auto_range=True)
    for i in range(3):
#        convert_image('phase_a' + str(i), auto_range=True)
#        convert_image('phase_b' + str(i), auto_range=True)
#        convert_image('amplitude' + str(i))
        convert_image('phase' + str(i), auto_range=True)
        convert_image('f_amplitude' + str(i), fixed_range=True, max_value=1600)
    convert_image('phase_full', auto_range=True)
    convert_image('f_amplitude_full', auto_range=True, max_rate=99)
#    convert_image('depth_raw', auto_range=True)
    convert_image('depth_out', auto_range=True, print_minmax=True)
    convert_image_b('RGB_camera')
    subprocess.call(['cp', 'f_amplitude_full.png', 'mask.png'])
