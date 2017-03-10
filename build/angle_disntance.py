#!/usr/bin/env python
# -*- coding: utf-8 -*-    
    
import dat2png as reader
import math
import cv2
import os
import numpy as np
import random
import pickle

def have_zero(array):
    return any([True if v==0 else False for v in array])
    
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
    
def convert_axis_S2D(a, b, c, d, e, f, acc, depths, start=600, end=1200, points=200):
    ''' リニアステージのパルス基準のデータを，キネクトで計測したなんとなくの距離データを基準としたデータに変換する．
    
    Parameters
    ----------
    a, b, c, d, e, f : arraylike
        16, 80, 120MHzのデプスデータとアンプりちゅーどデータ．順番はなんでもOK,入力順に出力．
    acc : arraylike
        あきゅみゅレーションデータ．どれだけ積算したデータか．積算量に対して加重平均．
    depths : arraylike
        各パルスにおける平均計測距離．(mm)
    start : numeric
        スタートデプス （mm）
    end : numeric
        エンドデプス
    points : int
        何点でサンプリングするか．細かすぎると，パルスからうまく変換できず歯抜けとなるかも．
        
    '''
    bins = np.linspace(start, end, points)
    acc_new = np.zeros(bins.shape)
    data = np.zeros([6, len(bins)])
    for idx, dp in enumerate(depths):
        ac = acc[idx]
        idx_new = min(len(bins), max(-1, int((dp-start)/(bins[1] - bins[0]))))
        if idx_new == len(bins):
            continue
        if idx_new == -1:
            continue
        acc_new[idx_new] += ac
        data[0, idx_new] += a[idx] * ac
        data[1, idx_new] += b[idx] * ac
        data[2, idx_new] += c[idx] * ac
        data[3, idx_new] += d[idx] * ac
        data[4, idx_new] += e[idx] * ac
        data[5, idx_new] += f[idx] * ac
        
    a = np.array([0 if v == 0 else t / v for t, v in zip(data[0], acc_new)])
    b = np.array([0 if v == 0 else t / v for t, v in zip(data[1], acc_new)])
    c = np.array([0 if v == 0 else t / v for t, v in zip(data[2], acc_new)])
    d = np.array([0 if v == 0 else t / v for t, v in zip(data[3], acc_new)])
    e = np.array([0 if v == 0 else t / v for t, v in zip(data[4], acc_new)])
    f = np.array([0 if v == 0 else t / v for t, v in zip(data[5], acc_new)])
    return a, b, c, d, e, f, acc_new
    
def condition_number(absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, ignore_80=False):
    num = 1 if absolute_depth else 0
    num += 2 if linear_stage else 0
    num += 4 if relative_120 else 0
    num += 8 if amplitude else 0
    num += 16 if ignore_80 else 0
    return num
    
def load_data(targ, base='base00', absolute_depth=True, linear_stage=True, relative_120=True, normalize_metric=True, guarantee=None, amplitude=False, ignore_80=False, points=200):
    ''' ステージパルス基準で計測したデータを読み込む．
    
    Parameters
    ----------
    absolute_depth : bool
        基準(base) からの計測誤差を計算するかどうか．
    linear_stage
        False の場合は，キネクトが計測する大体のデプスを基準としたデータに変換する
    relative_120
        120MHｚを基準とした相対値に変換するかどうか．（デプス，強度とも）
    normalize_metric
        平均分散を正規化するかどうか．未実装．
        
    '''
    file1_base = os.path.join('data', base, 'phase_depth_0.dat')        
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')        
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')      
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')        
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')        
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')      
    file1a_targ = os.path.join('data', targ, 'amp_depth_0.dat')        
    file2a_targ = os.path.join('data', targ, 'amp_depth_1.dat')        
    file3a_targ = os.path.join('data', targ, 'amp_depth_2.dat')        
    acc = reader.read_float_file(os.path.join('data', targ, 'accumurate_depth.dat'))
    depths = reader.read_float_file(os.path.join('data', targ, 'depth_data.dat'))
    
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    a16  = reader.read_float_file(file2a_targ)
    a80  = reader.read_float_file(file1a_targ)
    a120 = reader.read_float_file(file3a_targ)

    
    if absolute_depth:
        d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
        d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
        d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
        d16 -= d16_base
        d80 -= d80_base
        d120-= d120_base
    
    if not linear_stage:
        d16, d80, d120, a16, a80, a120, acc = convert_axis_S2D(d16, d80, d120, a16, a80, a120, acc, depths, points=points)
        
    if relative_120:
        d16 -= d120
        d80 -= d120
        d120 -= d120
        a16 = np.array([0 if d == 0 else v / d for v, d in zip(a16, a120)])
        a80 = np.array([0 if d == 0 else v / d for v, d in zip(a80, a120)])
        a120 = np.array([0 if d == 0 else 1. for v in a120])
    
    mean_normalizer = np.zeros(6)
    std_normalizer = np.ones(6)
    if normalize_metric:
        cond_num = condition_number(absolute_depth, linear_stage, relative_120, amplitude, ignore_80)
        f = open('results/normalization_coefficients.pickle', 'rb')
        norm_coef = pickle.load(f)
        mean_normalizer, std_normalizer = norm_coef[cond_num]
    
    if isinstance(guarantee, int):
#        d16 = np.array([v for v, a in zip(d16, acc) if a > guarantee])
#        d80 = np.array([v for v, a in zip(d80, acc) if a > guarantee])
#        d120 = np.array([v for v, a in zip(d120, acc) if a > guarantee])
#        a16 = np.array([v for v, a in zip(a16, acc) if a > guarantee])
#        a80 = np.array([v for v, a in zip(a80, acc) if a > guarantee])
#        a120 = np.array([v for v, a in zip(a120, acc) if a > guarantee])
        d16  = np.array([0 if a < guarantee else v for v, a in zip(d16 , acc)])
        d80  = np.array([0 if a < guarantee else v for v, a in zip(d80 , acc)])
        d120 = np.array([0 if a < guarantee else v for v, a in zip(d120, acc)])
        a16  = np.array([0 if a < guarantee else v for v, a in zip(a16 , acc)])
        a80  = np.array([0 if a < guarantee else v for v, a in zip(a80 , acc)])
        a120 = np.array([0 if a < guarantee else v for v, a in zip(a120, acc)])
        
    if relative_120:
        if amplitude:
            if not ignore_80:
                return ((seq - m) / s for seq, m, s in zip((d16, d80, a16, a80), mean_normalizer, std_normalizer))
            else:
                return ((seq - m) / s for seq, m, s in zip((d16, a16), mean_normalizer, std_normalizer))
        else:
            if not ignore_80:
                return ((seq - m) / s for seq, m, s in zip((d16, d80), mean_normalizer, std_normalizer))
            else:
                return (d16 - mean_normalizer[0]) / std_normalizer[0]
        
    if amplitude:
        return ((seq - m) / s for seq, m, s in zip((d16, d80, d120, a16, a80, a120), mean_normalizer, std_normalizer))
    else:
        return ((seq - m) / s for seq, m, s in zip((d16, d80, d120), mean_normalizer, std_normalizer))
        
def valid_l2_norm(vec1, vec2, ave=False):
    l2 = np.linalg.norm(vec1 - vec2, axis=1)
    valid = np.array([0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]
    
    
def plot_angle_distance(base='wood00', targ='wood_angle_test', angles=range(8, 91, 2)):
    import pylab
    import matplotlib.font_manager
    #times = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf') #, fontproperties=times
    times = matplotlib.font_manager.FontProperties(fname='C://Windows/Fonts/times.ttf') #, fontproperties=times
    pylab.rcParams.update({'font.size': 18, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

    probe = np.vstack(load_data(base, absolute_depth=False, relative_120=False, normalize_metric=False)).T
    dv = []
    for idx, angle in enumerate(angles):
        dname = targ + str(idx).zfill(2)
        test = np.vstack(load_data(dname, absolute_depth=False, relative_120=False, normalize_metric=False, guarantee=200)).T
        dv.append(valid_l2_norm(probe, test, ave=True))
    pylab.figure(figsize=(6, 2))
    dv = dv[::-1]
    limit = np.ones(len(dv)) * 4
    pylab.plot([90 - a for a in angles[::-1]], dv, lw=2)
    pylab.plot([90 - a for a in angles[::-1]], limit, lw=2, c='r')
    pylab.ylim((0, 50))
    pylab.xlim((0, 82))
    pylab.xlabel('Angle [degree]', fontproperties=times)
    pylab.ylabel('Distance [mm]', fontproperties=times)
    pylab.xticks(range(0, 81, 10), range(0, 81, 10), fontproperties=times)
    pylab.yticks(range(0, 51, 10), range(0, 51, 10), fontproperties=times)
    pylab.savefig('fig/angle_distance.pdf', bbox_inches='tight', pad_inches=0.1)
    
plot_angle_distance('wood_angle_test40')