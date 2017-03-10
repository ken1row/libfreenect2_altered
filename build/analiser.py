#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import dat2png as reader
from pylab import *
import math
import cv2
import os
import sklearn.svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import colorsys
# マテリアルのラベル色と、マテリアル名。 mask.png に使用する色。 use_mask オプションが有効なときに利用。
mask_legends = [ # BGR format.
    [0, 0, 255], # plaster
    [0, 106, 255], # metal
    [0, 216, 255], # acryl, plastic
    [0, 255, 182], # POM, Polyoxymethylene, plastic
    [255, 255, 0], # wax
    [255, 148, 0], # wood
    [255, 0, 178] # BR, Polybutadiene rubber, plastic
    ,[0, 255, 76] # PP, Polypropylene, plastic
    ,[33, 255, 0] # PE, Polyethylene, plastic
    ,[110, 0, 255] # Cotton
    ,[127, 127, 255] # PVC, Polyvinyl chloride, plastic
    ,[127,178, 255] # E-PVC, Expanded-PVC, plastic
    ,[255, 201, 127] # MDF, Medium-density fibreboard, wood
    ,[255, 146, 127] # Cork, wood
    ,[127, 233, 255] # Paper
    ,[255,255, 127] # Wax2, Waseda-Okamoto's sphere wax.
    ,[182, 127, 255] # Leather, cow
    ,[255, 127, 214] # Rubber
    ,[0, 127, 91] # Ceramic
    ,[127, 255, 218] # fake fruits
    ,[144, 255, 0] # PS, polysthyrene
    ,[255, 38, 0] # Bamboo
    ,[237, 127, 255] # Fabric, 9-polyester-1-Hemp
    ,[0, 0, 127] # Apple
    ,[55, 0, 127] # Carpet
    ,[220, 0, 255] # background
]
legends = ['Plaster', 'Metal', 'Acryl', 'POM', 'Wax', 'Wood', 'BR', 'PP', 'PE', 
           'Cotton', 'PVC', 'E-PVC', 'MDF', 'Cork', 'Paper', 'Wax2',
           'Leather', 'Rubber', 'Ceramic', 'Fake fruits', 'PS', 'Bamboo', '9-PEs-1-Hemp',
           'Apple', 'Carpet',
           'Background']
           
# クラスの縮退に利用。 nn.py を参照。
material_mapper = {
    'Plastic': ['Acryl', 'POM', 'PP', 'PE', 'BR', 'PVC', 'E-PVC']
    ,'Woody': ['Wood', 'Paper', 'Cork', 'Paper']
    ,'Metaric': ['Metal']
    }

# 有効なデータが置かれているディレクトリ
#dirs = ['long00', 'long02', 'wood00', 'fabric00', 'long03', 'PVC00', 'dishes00',
#        'EPVC00', 'MDF00', 'metal00' , 'plastic00', 'wax00', 'wax01',
#        'plastic01', 'leatherandrubber00', 'leatherandrubber01']    
dirs = ['dishes01', 'dishes02', 'dishes03', 'dishes04', 'dishes05']

def gen_colormap(replace_colormap_dict={}):
    '''
    Generate a list containing #FFFFFF style color strings. Colors are generated from mask_legends.
    
    Parameters
    ----------
    replace_colormap_dict : dict(idx, new_color)
        Change the color if set. new_color is BGR format. For example, if {0:[255, 0, 0]} is set, the 0th color become blue.
    '''
    col = []
    for idx, bgr in enumerate(mask_legends):
        if idx in replace_colormap_dict:
            bgr = replace_colormap_dict[idx]
        colorcode = '#' + ('%x'%bgr[2]).zfill(2) + ('%x'%bgr[1]).zfill(2) + ('%x'%bgr[0]).zfill(2)
        col.append(colorcode)
    return col

def pseudo_color(i, num):
    h = i / float(num)
    s = 1.
    v = 1.
    rgb = map(int, np.array(colorsys.hsv_to_rgb(h,s,v))*255.)
    return '#' + ('%x'%rgb[0]).zfill(2) + ('%x'%rgb[1]).zfill(2) + ('%x'%rgb[2]).zfill(2)

img_index_x, img_index_y = np.meshgrid(range(424), range(512))

def swap_xy(pos):
    '''
    Swap x and y axis of the point.
    '''
    return [pos[1], pos[0]]
def same_color(c1, c2):
    '''
    Check if two color is same or not. Any n-dimensional data can be applied.
    '''
    return all([v1==v2 for v1, v2 in zip(c1, c2)])

def phase2depth(phase, omega_MHz=16., c_mm_ns=300.):
    '''
    Convert phase to depth. The unit of returned depth is milli-meters.
    
    Parameters
    ----------
    phase: float
        Phase ranges from 0 to 2PI.
    omega_MHz: float
        Frequency in Mega-Hertz.
    c_mm_ns: float
        Speed of light. milli-meter per nano-second.
    '''
    return c_mm_ns * phase / (2. * math.pi) * 1000. / omega_MHz / 2.
    
def odd_check(odd_check_flag, val):
    '''
    Check if the value is odd. If odd_check_flag is False, return True even if the val is even.
    '''
    if not odd_check_flag:
        return True
    return val % 2 == 1
    
def round_zero_2pi(x):
    '''
    Unwrap phase into 0 to 2PI.
    '''
    while x>2.*math.pi:
        x-=2.*math.pi
    while x < 0:
        x+=2.*math.pi
    return x

def round_mpi_pi(x):
    '''
    Unwrap phase into -PI to PI.
    '''
    while x > math.pi:
        x-=2.*math.pi
    while x < -1*math.pi:
        x+=math.pi
    return x
  
def plot_amplitudes_theta_phi(use_mask=True, use_valid_mask=False, ignore_background=True, includes=legends, y_axis_wise=False, gravity=False):
    '''
    Parameters
    ----------
    use_mask: bool, optional
        If true, plot pixels are chosen from mask.png, otherwise hard coded points.
    use_valid_mask: bool, optional
        If true, plot pixels are masked by valid_pixel mask.
    '''
    figure()
    ampl80  = reader.read_float_file('f_amplitude0.dat', kinect=True)
    ampl15  = reader.read_float_file('f_amplitude1.dat', kinect=True)
    ampl120 = reader.read_float_file('f_amplitude2.dat', kinect=True)
    xs = []
    ys = []
    labels = []
    cmap_all = gen_colormap()
    cmap = []
    valid_mask = cv2.imread('valid_pixels.png').reshape((512*424, 3))
    valid_color = [0, 0, 255]
    confirm_valid_pixel = same_color if use_valid_mask else lambda a, b: True
    
    # data of each material.
    if use_mask:
        mask = cv2.imread('mask.png').reshape((512*424, 3))
        for idx, col in enumerate(mask_legends):
            if ignore_background and legends[idx] == 'Background':
                continue
            if not legends[idx] in includes:
                continue
            a15 =  np.array([v for v, c, vp in zip(ampl15.flatten() , mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            if len(a15) == 0:
                continue
            a80 =  np.array([v for v, c, vp in zip(ampl80.flatten() , mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            a120 = np.array([v for v, c, vp in zip(ampl120.flatten(), mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            r = np.sqrt(a15*a15+a80*a80)
            theta = np.arccos(a15 / r)
            norm = np.sqrt(a15*a15+a80*a80+a120*a120)
            phi = np.arccos(r / norm)
            if not y_axis_wise:
                xs.append(theta)
                ys.append(phi)
                labels.append(legends[idx])
                cmap.append(cmap_all[idx])
            else:
                y_values = np.array([v for v, c, vp in zip(img_index_y.flatten(), mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
                for y in set(y_values):
                    theta_subset = [v for v, yv in zip(theta, y_values) if yv == y]
                    phi_subset = [v for v, yv in zip(phi, y_values) if yv == y]
                    xs.append(theta_subset)
                    ys.append(phi_subset)
                    labels.append('y = '+str(y))
                    
    else:
        assert False, 'Code removed because it is not useful anymore.'
    
#    plot(x, y, 'o')
    xg = []
    yg = []
    if not y_axis_wise:
        for xi, yi, li, cm in zip(xs, ys, labels, cmap):
            plot(xi, yi, 'o', label=li, color=cm)
    else:
        len_ys = len(xs)
        for xi, yi, li, idx in zip(xs, ys, labels, range(len_ys)):
            if len(xi) == 0:
                continue
            if not gravity:
                plot(xi, yi, 'o', color=pseudo_color(idx, len_ys))
            else:
                plot(np.mean(xi), np.mean(yi), 'o', color=pseudo_color(idx, len_ys))
                xg.append(np.mean(xi))
                yg.append(np.mean(yi))
#        if gravity:
#            plot(xg, yg)
        
    legend(loc='upper left')
    xlabel('Theta (Azimuth)')
    ylabel('Phi (Zenith)')
#    show()
    filename = 'amplitude_plot_theta_phi_mask' if use_mask else 'amplitude_plot_theta_phi'
    filename += '_yvalues' if y_axis_wise else ''
    filename += '_valid.png' if use_valid_mask else '.png'
    savefig(filename)

 
def plot_phases_in_depth(use_valid_mask=False, ignore_background=True):    
    phase15  = reader.read_float_file('phase1.dat', kinect=True).flatten()
    phase80  = reader.read_float_file('phase0.dat', kinect=True).flatten()
    phase120 = reader.read_float_file('phase2.dat', kinect=True).flatten()    
    mask = cv2.imread('mask.png').reshape((512*424, 3))
    valid_mask = cv2.imread('valid_pixels.png').reshape((512*424, 3))
    valid_color = [0, 0, 255]
    confirm_valid_pixel = same_color if use_valid_mask else lambda a, b: True
    xs = []
    ys = []
    labels = []
    cmap_all = gen_colormap()
    cmap = []
    for idx, col in enumerate(mask_legends):
        if ignore_background and legends[idx] == 'Background':
            continue
        p16 =  np.array([v for v, c, vp in zip(phase15, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        if len(p16) == 0:
            continue
        p80 =  np.array([v for v, c, vp in zip(phase80, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        p120 =  np.array([v for v, c, vp in zip(phase120, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])        
        d16 = phase2depth(p16, 16.)
        d80 = phase2depth(p80, 80.)
        d120= phase2depth(p120, 120.)
        xs.append(d120 - d80)
        ys.append(d120 - d16)
        labels.append(legends[idx])
        cmap.append(cmap_all[idx])
    figure()
    for xi, yi, li, cm in zip(xs, ys, labels, cmap):
        plot(xi, yi, 'o', label=li, color=cm)
    legend(loc='lower right')
    xlabel('120 - 80 MHz')
    ylabel('120 - 16 MHz')
#    show()
    savefig('depth_distortion_plot.png')
           
        
def generate_feature_vector_from_mask(use_valid_mask=False):
#    phase_all = reader.read_float_file('phase_full.dat', kinect=True).flatten()
    phase15  = reader.read_float_file('phase1.dat', kinect=True).flatten()
    phase80  = reader.read_float_file('phase0.dat', kinect=True).flatten()
    phase120 = reader.read_float_file('phase2.dat', kinect=True).flatten()    
    ampl80  = reader.read_float_file('f_amplitude0.dat', kinect=True).flatten()
    ampl15  = reader.read_float_file('f_amplitude1.dat', kinect=True).flatten()
    ampl120 = reader.read_float_file('f_amplitude2.dat', kinect=True).flatten()
    mask = cv2.imread('mask.png').reshape((512*424, 3))
    
    valid_mask = cv2.imread('valid_pixels.png').reshape((512*424, 3))
    valid_color = [0, 0, 255]
    confirm_valid_pixel = same_color if use_valid_mask else lambda a, b: True
    
    for idx, col in enumerate(mask_legends):
        a15 =  np.array([v for v, c, vp in zip(ampl15, mask, valid_mask)    if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        if len(a15) == 0 :
            continue
        a80 =  np.array([v for v, c, vp in zip(ampl80, mask, valid_mask)    if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        a120 = np.array([v for v, c, vp in zip(ampl120, mask, valid_mask)   if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
#        pf  =  np.array([v for v, c, vp in zip(phase_all, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        p16 =  np.array([v for v, c, vp in zip(phase15, mask, valid_mask)   if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        p80 =  np.array([v for v, c, vp in zip(phase80, mask, valid_mask)   if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        p120 = np.array([v for v, c, vp in zip(phase120, mask, valid_mask)  if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
        r = np.sqrt(a15*a15+a80*a80)
        theta = np.arccos(a15 / r)
        norm = np.sqrt(a15*a15+a80*a80+a120*a120)
        phi = np.arccos(r / norm)
        feature_vec = [(a1, a2, p1, p2, p3) for a1, a2, p1, p2, p3 in zip(theta, phi, p16, p80, p120)]
        filename = 'feature'+legends[idx]+'.npy'
        np.save(filename, feature_vec)

def plot_features_from_database(d='data', include=legends, exclude=['Background', 'E-PVC', 'PVC']):
    cmap_all = gen_colormap()
    xs = []
    ys = []
    ds1 = []
    ds2 = []
    for idx in range(len(legends)):
        xs.append([])
        ys.append([])
        ds1.append([])
        ds2.append([])
    
    for dd in dirs:
        path = os.path.join(d, dd)
        for idx, label in enumerate(legends):
            if label in exclude:
                continue
            if not label in include:
                continue
            filename = os.path.join(path, 'feature'+label+'.npy')
            if not os.path.exists(filename):
                continue
            feature_vec = np.load(filename)
            for f_vec in feature_vec:
                xs[idx].append(f_vec[0])
                ys[idx].append(f_vec[1])
                d1 = phase2depth(f_vec[2], 16.)
                d2 = phase2depth(f_vec[3], 80.)
                d3 = phase2depth(f_vec[4], 120.)
                ds1[idx].append(d3 - d2)
                ds2[idx].append(d3 - d1)
                
    figure()
    for x, y, c, l in zip(xs, ys, cmap_all, legends):
        if len(x) == 0:
            continue
        plot(x, y, 'o', color=c, label=l)
    xlabel('Theta (Azimuth)')
    ylabel('Phi (Zenith)')
    legend()
    savefig('plot_amplitude_db.png')
    
    figure()    
    for x, y, c, l in zip(ds1, ds2, cmap_all, legends):
        if len(x) == 0:
            continue
        plot(x, y, 'o', color=c, label=l)
    xlabel('Depth Distortion (120 - 80 MHz)')
    ylabel('Depth Distortion (120 - 16 MHz)')
    legend()
    savefig('plot_depths_db.png')
    
def depth_dist_mask(thresh=-25.):
    img = cv2.imread('f_amplitude_full.png')
    p16  = reader.read_float_file('phase1.dat', kinect=True)
    p80  = reader.read_float_file('phase0.dat', kinect=True)
    p120 = reader.read_float_file('phase2.dat', kinect=True)
    d16 = phase2depth(p16, 16.)
    d80 = phase2depth(p80, 80.)
    mask = np.zeros((424, 512, 3))
#    d120= phase2depth(p120, 120.)
    v = d80 - d16
    for i in range(512):
        for j in range(424):
            mask[j, i] = [0, 0, 255] if v[j, i] > thresh else np.ones(3)*img[j, i]
    cv2.imwrite('valid_pixels.png', mask)
    
    
    
if __name__ == '__main__':
    #amplitude_by_color(mode='vivid')
    
    #plot_amplitudes()
    #plot_amplitudes_theta_phi()
    #plot_phase_gaps()
    #draw_plot_area()
    
    #plot_amplitudes_theta_phi_at_different_location_and_frame()
    #plot_amplitude_multi_frames() 
    
#    plot_amplitudes_theta_phi(use_mask=True, use_valid_mask=True)
#    generate_feature_vector_from_mask(use_valid_mask=True)
#    plot_phases_in_depth(use_valid_mask=True)
    
    plot_amplitudes_theta_phi(use_mask=True, use_valid_mask=True, includes=['Carpet'], y_axis_wise=True, gravity=True)
    
#    plot_features_from_database(include=['Cotton', 'PE', 'Fake fruits', '9-PEs-1-Hemp', 'PS', 'Wood'])
#    plot_features_from_database(exclude=['Background', 'E-PVC', 'PVC'])
    
    #for l in ['POM', 'Plaster', 'Wood', 'Metal']:
    #    plot_3_depths(l, True, True)
    #plot_3_depths('Paper', filter_mask=True)
    
    #depth_dist_mask(-25.)