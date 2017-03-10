#!/usr/bin/env python

import numpy as np
from array import array
#import struct
try:
    import cv2
    def imwrite(f, d):
        cv2.imwrite(f, d)
except ImportError:
    try:
        from PIL import Image
        def imwrite(f, d):
            pil_image = Image.fromarray(d)
            pil_image.save(f)
    except ImportError:
        try:
            import matplotlib.image as mpimg
            def imwrite(f, d):
                print 'Info: The image will become pseudo-color.'
                mpimg.imsave(f, d)
        except ImportError:
            print 'Warning: Image output is not supported.'
            def imwrite(f, d):
                print 'Ignored: Saving image to', f
                pass

def read_float_file(filename):
    ''' Read a binary (double) file as numpy array.
    
    Parameters
    ----------
    filename : str
        File name of the input.
        
    Returns
    -------
    out : ndarray
        Array of all float values with the 1d shape.
    '''
    in_file = open(filename, 'r')
    float_array = array('d')
    float_array.fromstring(in_file.read())
    np_array = np.array(float_array)
    return np_array
    
def convert_image(fin='phase0', extin='.dat', extout='.png', auto_range=False, max_rate=100.):
    ''' Convert a binary file to a 16-bit grayscale image.
    
    Parameters
    ----------
    fin : str
        File name without extension.
    extin : str
        Extension of the input file.
    extout : str
        Extension of the output file.
    auto_range : bool
        If True, the dinamic range is automatically adjusted.
    max_rate : float
        If auto_range is True, max_rate is used for determining the maximam value.
    '''
    data = read_float_file(fin+extin)
    data = data.reshape((512,424)).T[::-1,::-1]
    if auto_range:
        data = data - np.amin(data)
        data = np.minimum(data * 65535. / np.percentile(data, max_rate), 65535)
    imwrite(fin+extout, np.uint16(data))
    
def convert_image_rgb(fin='RGB_camera', extin='.dat', extout='.png', auto_range=False):
    data = read_float_file(fin+extin)
    data = data.reshape((1080,1920,3))[:,::-1, :]
    if auto_range:
        data = data - np.amin(data)
        data = data * 65535. / np.amax(data)
    imwrite(fin+extout, np.uint8(data))
    
if __name__ == '__main__':
    for i in range(3):
        convert_image('phase' + str(i), auto_range=True)
        convert_image('f_amplitude' + str(i))
    convert_image('phase_full', auto_range=True)
    convert_image('f_amplitude_full', auto_range=True, max_rate=90)
    convert_image('depth_out', auto_range=True)
    convert_image_rgb('RGB_camera')