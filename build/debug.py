#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:05:05 2016

@author: tanaka
"""

from analiser import *

dirs = [os.path.join('data', 'debugcarpet' + ('%d'%v).zfill(2)) for v in range(9) ]
parent_mask = np.zeros((424, 512))
parent_mask[100:280, 110:380] = np.ones((180, 270))

def plot_amplitudes_debug(use_mask=True, y_axis_wise=False, gravity=False, sort_by_depth=False, clustering=False, cluster_num=100, single=False):
    '''
    Parameters
    ----------
    use_mask: bool, optional
        If true, plot pixels are chosen from mask.png, otherwise hard coded points.
    use_valid_mask: bool, optional
        If true, plot pixels are masked by valid_pixel mask.
    '''
    global dirs
    if single:
        dirs = ['data/debugcarpet00']        
    use_valid_mask=True
    figure()
    xs = []
    ys = []
    ds = []
    labels = []
    cmap_all = gen_colormap()
    cmap = []
    valid_mask = cv2.imread('valid_pixels.png').reshape((512*424, 3))
    valid_color = [0, 0, 255]
    def confirm_valid_pixel(a, b):
        if use_valid_mask:
            return all([v1 == v2 for v1, v2 in zip(a, b)])
        else:
            return True
    
    # data of each material.
    if use_mask:
        mask = parent_mask.flatten() #cv2.imread('mask.png').reshape((512*424, 3))
        def same_color(a, b):
            return a == b
        col = 1.
        for idx, d in enumerate(dirs):       
            ampl80  = reader.read_float_file(os.path.join(d, 'f_amplitude0.dat'), kinect=True)
            ampl15  = reader.read_float_file(os.path.join(d, 'f_amplitude1.dat'), kinect=True)
            ampl120 = reader.read_float_file(os.path.join(d, 'f_amplitude2.dat'), kinect=True)     
            depths = reader.read_float_file(os.path.join(d, 'depth_out.dat'), kinect=True)
            a15 =  np.array([v for v, c, vp in zip(ampl15.flatten() , mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            if len(a15) == 0:
                continue
            a80 =  np.array([v for v, c, vp in zip(ampl80.flatten() , mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            a120 = np.array([v for v, c, vp in zip(ampl120.flatten(), mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            dpth = np.array([v for v, c, vp in zip(depths.flatten(), mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            r = np.sqrt(a15*a15+a80*a80)
            theta = np.arccos(a15 / r)
            norm = np.sqrt(a15*a15+a80*a80+a120*a120)
            phi = np.arccos(r / norm)
            # remove nan
            dpth = [v for t, p, v in zip(theta, phi, dpth) if t == t and p == p]
            theta_ = [t for t, p in zip(theta, phi) if t == t and p == p]
            phi = [p for t, p in zip(theta, phi) if t == t and p == p]
            theta = theta_
            if not y_axis_wise:
                xs.append(theta)
                ys.append(phi)
                ds.append(dpth)
                labels.append('depth '+str(idx))
                cmap.append(pseudo_color(idx, len(dirs)))
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
        if sort_by_depth:
            all_data_tuples = []
            for xi, yi, di in zip(xs, ys, ds):
                all_data_tuples.extend([(x, y, d) for x, y, d in zip(xi, yi, di)])
            datas = sorted(all_data_tuples, key=lambda tup: tup[2])
            lend = len(datas)
            if clustering:
                newdata = []
                for i in range(0, lend, cluster_num):
                    data = datas[i:min(lend, i+cluster_num)]
                    mean = np.mean(data, axis=0)
                    newdata.append(mean)
                datas = newdata
                lend = len(datas)
            for idx, tup in enumerate(datas):
                plot(tup[0], tup[1], 'o', color=pseudo_color(idx, lend))
        else:
            for xi, yi, li, cm in zip(xs, ys, labels, cmap):
                if len(xi) == 0:
                    continue
                if not gravity:
                    plot(xi, yi, 'o', label=li, color=cm)
                else:
                    plot(np.mean(xi), np.mean(yi), 'o', label=li, color=cm)
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
    filename += '_gravity' if gravity else ''
    filename += '_yvalues' if y_axis_wise else ''
    filename += '_sorted' if sort_by_depth else ''
    filename += '_valid.png' if use_valid_mask else '.png'
    savefig(filename)
    
def plot_phases_debug(use_mask=True, y_axis_wise=False, gravity=False, sort_by_depth=False, clustering=False, cluster_num=100, single=False):
    '''
    Parameters
    ----------
    use_mask: bool, optional
        If true, plot pixels are chosen from mask.png, otherwise hard coded points.
    use_valid_mask: bool, optional
        If true, plot pixels are masked by valid_pixel mask.
    '''
    global dirs
    if single:
        dirs = ['data/debugcarpet00']        
    use_valid_mask=True
    figure()
    xs = []
    ys = []
    ds = []
    labels = []
    cmap_all = gen_colormap()
    cmap = []
    valid_mask = cv2.imread('valid_pixels.png').reshape((512*424, 3))
    valid_color = [0, 0, 255]
    def confirm_valid_pixel(a, b):
        if use_valid_mask:
            return all([v1 == v2 for v1, v2 in zip(a, b)])
        else:
            return True
    
    # data of each material.
    if use_mask:
        mask = parent_mask.flatten() #cv2.imread('mask.png').reshape((512*424, 3))
        def same_color(a, b):
            return a == b
        col = 1.
        for idx, d in enumerate(dirs):       
            phase80  = reader.read_float_file(os.path.join(d, 'phase0.dat'), kinect=True).flatten()
            phase15  = reader.read_float_file(os.path.join(d, 'phase1.dat'), kinect=True).flatten()
            phase120 = reader.read_float_file(os.path.join(d, 'phase2.dat'), kinect=True).flatten()
            depths = reader.read_float_file(os.path.join(d, 'depth_out.dat'), kinect=True).flatten()
            p16 =  np.array([v for v, c, vp in zip(phase15, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            if len(p16) == 0:
                continue
            p80 =  np.array([v for v, c, vp in zip(phase80, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
            p120 =  np.array([v for v, c, vp in zip(phase120, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])        
            dpth =  np.array([v for v, c, vp in zip(depths, mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])        
            d16 = phase2depth(p16, 16.)
            d80 = phase2depth(p80, 80.)
            d120= phase2depth(p120, 120.)
            if not y_axis_wise:
                xs.append(d120 - d80)
                ys.append(d120 - d16)
                ds.append(dpth)
                labels.append('depth '+str(idx))
                cmap.append(pseudo_color(idx, len(dirs)))
            else:
                y_values = np.array([v for v, c, vp in zip(img_index_y.flatten(), mask, valid_mask) if same_color(c, col) and confirm_valid_pixel(vp, valid_color)])
                for y in set(y_values):
                    d120_80_subset = [v for v, yv in zip(d120 - d80, y_values) if yv == y]
                    d120_16_subset = [v for v, yv in zip(d120 - d16, y_values) if yv == y]
                    xs.append(d120_80_subset)
                    ys.append(d120_16_subset)
                    labels.append('y = '+str(y))
                    
    else:
        assert False, 'Code removed because it is not useful anymore.'
    
#    plot(x, y, 'o')
    xg = []
    yg = []
    if not y_axis_wise:
        if sort_by_depth:
            all_data_tuples = []
            for xi, yi, di in zip(xs, ys, ds):
                all_data_tuples.extend([(x, y, d) for x, y, d in zip(xi, yi, di)])
            datas = sorted(all_data_tuples, key=lambda tup: tup[2])
            lend = len(datas)
            if clustering:
                newdata = []
                for i in range(0, lend, cluster_num):
                    data = datas[i:min(lend, i+cluster_num)]
                    mean = np.mean(data, axis=0)
                    newdata.append(mean)
                datas = newdata
                lend = len(datas)
            for idx, tup in enumerate(datas):
                plot(tup[0], tup[1], 'o', color=pseudo_color(idx, lend))
                print tup[2]
        else:
            for xi, yi, li, cm in zip(xs, ys, labels, cmap):
                if len(xi) == 0:
                    continue
                if not gravity:
                    plot(xi, yi, 'o', label=li, color=cm)
                else:
                    plot(np.mean(xi), np.mean(yi), 'o', label=li, color=cm)
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
    xlabel('120 - 80 MHz')
    ylabel('120 - 16 MHz')
#    show()
    filename = 'phase_plot_theta_phi_mask' if use_mask else 'phase_plot_theta_phi'
    filename += '_gravity' if gravity else ''
    filename += '_yvalues' if y_axis_wise else ''
    filename += '_sorted' if sort_by_depth else ''
    filename += '_clustered'+str(cluster_num) if clustering else ''
    filename += '_single' if single else ''
    filename += '_valid.png' if use_valid_mask else '.png'
    savefig(filename)
    
def plot_phases_depth(targ=''):
    m_depth = reader.read_float_file('depth_out.dat', kinect=True)
    depth = reader.read_float_file('depth_bins.dat')
    acc = reader.read_float_file('accumurate_depth.dat')
    phase80 = reader.read_float_file('phase_depth_0.dat')
    phase16 = reader.read_float_file('phase_depth_1.dat')
    phase120 = reader.read_float_file('phase_depth_2.dat')
    d16 = phase2depth(phase16, 16.)
    d80 = phase2depth(phase80, 80.)
    d120= phase2depth(phase120, 120.)
    len_d = len(depth)
    figure()
    for x, y, d, a, idx in zip(d120-d80, d120-d16, depth, acc, range(len_d)):
        if a < 10:
            continue
        plot(x, y, 'o', color=pseudo_color(idx, len_d))
    xlabel('120 - 80 MHz')
    ylabel('120 - 16 MHz')
    filename = 'depth_dependent_relative_depth_distortion'
    filename += '_'+targ+'.png' if len(targ) > 0 else '.png'
    savefig(filename)
    
#def plot_phases_depth_all(mat='wood', targ=['wood00', 'wood01', 'wood02', 'wood03', 'wood04']):
#def plot_phases_depth_all(mat='mac', targ=['mac02', 'mac03', 'mac04']):
#def plot_phases_depth_all(mat='acryl', targ=['acryl01', 'acryl02', 'acryl03', 'acryl04']):
#def plot_phases_depth_all(mat='fabric', targ=['fabric01', 'fabric02', 'fabric03']):
def plot_phases_depth_all(mat = '', targ=['wood04', 'acryl04', 'fabric03', 'mac04', 'waxball00', 'rubber00', 'leather00']):#, 'fake_banana00']):
#def plot_phases_depth_all(mat='fake', targ=['fake_banana00', 'fake_apple00']):
    figure()
    for t in targ:
        d = os.path.join('data', t)
        phase80  = reader.read_float_file(os.path.join(d, 'phase_depth_0.dat'))
        phase16  = reader.read_float_file(os.path.join(d, 'phase_depth_1.dat'))
        phase120 = reader.read_float_file(os.path.join(d, 'phase_depth_2.dat'))
        d16 = phase2depth(phase16, 16.)
        d80 = phase2depth(phase80, 80.)
        d120= phase2depth(phase120, 120.)
        len_d = len(d120)
        plot(d120-d80, d120-d16, 'o', label=t)
    legend()
    xlabel('120 - 80 MHz')
    ylabel('120 - 16 MHz')
    filename = 'depth_dependent_relative_depth_distortion_all'
    filename += '_'+mat+'.png' if len(mat) > 0 else '.png'
    savefig(filename)

if __name__ == '__main__':
#    plot_amplitudes_debug(sort_by_depth=True, clustering=True)#sort_by_depth=True)#, gravity=True)
#    plot_amplitudes_debug(sort_by_depth=True, clustering=True, single=True)#sort_by_depth=True)#, gravity=True)
#    plot_phases_debug(sort_by_depth=True, clustering=True)
#    plot_phases_debug(sort_by_depth=True, clustering=True, single=True)
    pass
#    plot_phases_depth('data/wood00')
    plot_phases_depth_all()