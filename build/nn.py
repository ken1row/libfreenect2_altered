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
    
def load_data(targ, base='base00', absolute_depth=True, linear_stage=True, relative_120=True, normalize_metric=True, guarantee=None, amplitude=False, ignore_80=False, points=200, relative_center_depth_only=False, relative_frequency_only=False, both_axis=False):
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
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    a16  = reader.read_float_file(file2a_targ)
    a80  = reader.read_float_file(file1a_targ)
    a120 = reader.read_float_file(file3a_targ)

    if relative_center_depth_only:
        center_idx = int(len(d16) // 2)
        new_depths = depths - depths[center_idx]
        new_d80 = d80 - d80[center_idx]
        return new_d80 - new_depths
        
    if relative_frequency_only:
        center_idx = int(len(d16) // 2)
        return np.array((d120[center_idx] - d80[center_idx], d120[center_idx] - d16[center_idx]))
    
    if both_axis:
        center_idx = int(len(d16) // 2)
        new_depths = depths - depths[center_idx]
        new_d80 = d80 - d80[center_idx] - new_depths
        new_d120 = d120 - d120[center_idx] - new_depths
        new_d16 = d16 - d16[center_idx] - new_depths
        return np.hstack((new_d80, new_d120, new_d16))
        
    if absolute_depth:
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
    
def valid_l2_norm2(vec1, vec2, ave=False):
    dif = vec1 - vec2
    l2 = np.sqrt(dif*dif)
    valid = np.array([0 if t==0 or p==0 else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]
    
def nearest_neighbor_classify(test_set, training_set, confusion, verbose=False, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=False, ignore_80=False):
    training_data = []
    for mats in training_set:
        t_data_mat = []
        for m in mats:
            t_data_mat.append(np.vstack(load_data(m, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T)
        training_data.append(t_data_mat)

    for idx_test, tests in enumerate(test_set):
        for test in tests:
            test_vec = np.vstack(load_data(test, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T
            class_costs = []
            for idx_ref, materials in enumerate(training_set):
                costs = []
                for idx_tmp, ref in enumerate(materials):
                    costs.append(valid_l2_norm(test_vec, training_data[idx_ref][idx_tmp]))
                class_costs.append(min(costs))
            nn = np.argmin(class_costs)
            confusion[idx_test, nn] += 1
            if verbose:
                if idx_test != nn:
                    print 'Class', idx_test, test, '\tis confused at\t Class', nn, training_set[nn][0][:-2]
    return confusion
    
def cross_validation_nearest_neighbor_classifier(materials, rep=10, max_index=1, num_test=3, num_training=3, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=True, ignore_80=False):
    confusion = np.zeros((len(materials), len(materials)))
    filename = 'results/confusion_'
    filename += 'alumi-base' if absolute_depth else 'material-only'
    filename += '_linear-stage' if linear_stage else '_depth-base'
    filename += '_120-base' if relative_120 else ''
    filename += '_80-ign' if ignore_80 else ''
    filename += '_with-amp' if amplitude else ''
    filename += '_normalize' if amplitude else ''
    for idx in range(rep):
        targ = []
        prob = []
        for m in materials:
            tm = []
            while len(tm) < num_test:
                p = m + str(random.randint(0, max_index)).zfill(2)
                if not p in tm:
                    tm.append(p)
            targ.append(tm)
            pm = []
            while len(pm) < num_training:
                p = m + str(random.randint(0, max_index)).zfill(2)
                if not (p in pm or p in tm):
                    pm.append(p)
            prob.append(pm)
        confusion = nearest_neighbor_classify(targ, prob, confusion, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize=normalize, amplitude=amplitude, ignore_80=ignore_80)
    np.save(filename+'.npy', confusion)
    diag = np.diag(confusion)
    accuracy = 100. * sum(diag) / np.sum(confusion)
    print sum(diag), np.sum(confusion), 'Accuracy =', accuracy
    confusion_img = confusion * 255 / np.amax(confusion)
    cv2.imwrite(filename+str(accuracy)[:4]+'.png', confusion_img)
    
def cross_validation_many_condition(mats):
    rep=16
    max_idx=10
    num_test = 6
    num_training = 2
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training)
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, linear_stage=False)
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, amplitude=True)
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, amplitude=True, linear_stage=False)
#    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, absolute_depth=False)
#    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, absolute_depth=False, amplitude=True)
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, absolute_depth=False, amplitude=True, linear_stage=False)
    cross_validation_nearest_neighbor_classifier(mats, rep=rep, max_index=max_idx, num_test=num_test, num_training=num_training, absolute_depth=False, linear_stage=False)
    
def get_mean_and_std(dir_list, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, ignore_80=False):
    cond_num = condition_number(absolute_depth, linear_stage, relative_120, amplitude, ignore_80)    
    data = []
    for d in dir_list:
        t = np.vstack(load_data(d, guarantee=100, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, amplitude=amplitude, ignore_80=ignore_80)).T
        data.append(t)
    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print cond_num, mean, std
    return cond_num, (mean, std)
    
def calculate_mean_and_std(materials, rep=10):
    filename='results/normalization_coefficients.pickle'
    f = open(filename, 'wb')
    dir_lists = []
    data = {}
    for m in materials:
        for i in range(rep):
            dir_lists.append(m+str(i).zfill(2))
    num, res = get_mean_and_std(dir_lists, True, True, True, True)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, True, True, True, False)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, True, False, True, True)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, False, True, True, True)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, True, False, True, False)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, False, False, True, True)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, False, True, True, False)
    data[num] = res
    num, res = get_mean_and_std(dir_lists, False, False, True, False)
    data[num] = res
    pickle.dump(data, f)
    f.close()
        
def convert_dataset_to_kinect_depth_axis(materials, target='../../../demo/libfreenect2/build/data', max_idx=0, randomize=False, average=False):
    for m in materials:
        indir = os.path.join('data', m+'00')
        outdir = os.path.join(target, m)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        t = np.vstack(load_data(m+'00', guarantee=100, absolute_depth=False, linear_stage=False, relative_120=True, amplitude=False, ignore_80=False, normalize_metric=False))
        np.save(os.path.join(outdir, '3mm.npy'), t)
     
def convert_dataset_to_kinect_depth_axis2(materials, target='../../../depthcorrection/libfreenect2/build/data', max_idx=0, randomize=False, average=False):
    for m in materials:
        indir = os.path.join('data', m)
        outdir = os.path.join(target, m)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        t = np.vstack(load_data(m, points=600, guarantee=100, absolute_depth=False, linear_stage=False, relative_120=True, amplitude=False, ignore_80=False, normalize_metric=False))
        np.save(os.path.join(outdir, '1mm.npy'), t)
    
def cross_validation_nearest_neighbor_classifier_probeset(materials, probe, rep=10, concave=False, angle=False, max_index=1, max_index_prob=24, num_test=3, num_training=3, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=True, ignore_80=False):
    confusion = np.zeros((max_index_prob, len(materials)))
    filename = 'results/'
    filename += 'concave_confusion_' if concave else ''
    filename += 'angle_confusion_' if angle else 'confusion_'
    filename += 'alumi-base' if absolute_depth else 'material-only'
    filename += '_linear-stage' if linear_stage else '_depth-base'
    filename += '_120-base' if relative_120 else ''
    filename += '_80-ign' if ignore_80 else ''
    filename += '_with-amp' if amplitude else ''
    filename += '_normalize' if amplitude else ''
    for idx in range(rep):
        targ = []
        prob = []
        for m in materials:
            if concave:
                if 'cardboard' in m:
                    tm = [probe+'12'] * num_test
                    targ.append(tm)
                    continue
            if angle:
                if 'wood' in m:
                    tm = [probe+'36'] * num_test
                    targ.append(tm)
                    continue
            tm = []
            while len(tm) < num_test:
                p = m + str(random.randint(0, max_index)).zfill(2)
                if not p in tm:
                    tm.append(p)
            targ.append(tm)
        for i in range(max_index_prob):
            prob.append([probe+str(i).zfill(2)])
        confusion = nearest_neighbor_classify(prob, targ, confusion, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize=normalize, amplitude=amplitude, ignore_80=ignore_80)
    np.save(filename+'.npy', confusion)
    diag = np.diag(confusion)
    accuracy = 100. * sum(diag) / np.sum(confusion)
    print sum(diag), np.sum(confusion), 'Accuracy =', accuracy
    confusion_img = confusion * 255 / np.amax(confusion)
    cv2.imwrite(filename+str(accuracy)[:4]+'.png', confusion_img)
    
def distance_stats_matrix(mats, reps=10, absolute_depth=False, linear_stage=False, relative_120=False, amplitude=False, normalize=False, ignore_80=False):
    filename = 'results/distance'
    distance_mat = []
    for mid, m in enumerate(mats):
        distances = []
        for i in range(reps):
            vec1 = np.vstack(load_data(m+str(i).zfill(2), absolute_depth=False, linear_stage=False, relative_120=False, normalize_metric=False)).T
            for idx, m2 in enumerate(mats):
                for j in range(reps):
                    vec2 = np.vstack(load_data(m+str(i).zfill(2), absolute_depth=False, linear_stage=False, relative_120=False, normalize_metric=False)).T
                    distances.append(valid_l2_norm(vec1, vec2, True))
#                    distances.append(mid*1000+i*100+idx*10+j)
        distance_mat.append(distances)
    distance_mat = np.array(distance_mat).reshape((len(mats), reps, len(mats), reps))
    mean = np.mean(np.mean(distance_mat, axis=3), axis=1)
    min_ = np.min(np.min(distance_mat, axis=3), axis=1)
    max_ = np.max(np.max(distance_mat, axis=3), axis=1)
    np.save(filename+'mean.npy', mean)
    np.save(filename+'max.npy', max_)
    np.save(filename+'min.npy', min_)
    print ''
    
    
def nearest_neighbor_classify_single_distortion(test_set, training_set, confusion, axis='depth', verbose=False):
    training_data = []
    if axis == 'depth':
        depth_only = True
        freq_only = False
        both = False
    elif axis == 'frequency':
        freq_only = True
        depth_only = False
        both = False
    elif axis == 'both':
        freq_only=False
        depth_only=False
        both = True
    else:
        return confusion
    for mats in training_set:
        t_data_mat = []
        for m in mats:
            t_data_mat.append(load_data(m, relative_center_depth_only=depth_only, relative_frequency_only=freq_only, both_axis=both))
        training_data.append(t_data_mat)

    for idx_test, tests in enumerate(test_set):
        for test in tests:
            test_vec = load_data(test, relative_center_depth_only=depth_only, relative_frequency_only=freq_only)
            class_costs = []
            for idx_ref, materials in enumerate(training_set):
                costs = []
                for idx_tmp, ref in enumerate(materials):
                    costs.append(valid_l2_norm2(test_vec, training_data[idx_ref][idx_tmp]))
                class_costs.append(min(costs))
            nn = np.argmin(class_costs)
            confusion[idx_test, nn] += 1
            if verbose:
                if idx_test != nn:
                    print 'Class', idx_test, test, '\tis confused at\t Class', nn, training_set[nn][0][:-2]
    return confusion
      
    
def cross_validation_nearest_neighbor_classifier_single_distortion(materials, rep=10, max_index=1, num_test=3, num_training=3, axis='depth'):
    confusion = np.zeros((len(materials), len(materials)))
    filename = 'results/confusion_' + axis + '_axis'
    for idx in range(rep):
        targ = []
        prob = []
        for m in materials:
            tm = []
            while len(tm) < num_test:
                p = m + str(random.randint(0, max_index)).zfill(2)
                if not p in tm:
                    tm.append(p)
            targ.append(tm)
            pm = []
            while len(pm) < num_training:
                p = m + str(random.randint(0, max_index)).zfill(2)
                if not (p in pm or p in tm):
                    pm.append(p)
            prob.append(pm)
        confusion = nearest_neighbor_classify_single_distortion(targ, prob, confusion, axis = axis)
    np.save(filename+'.npy', confusion)
    diag = np.diag(confusion)
    accuracy = 100. * sum(diag) / np.sum(confusion)
    print sum(diag), np.sum(confusion), 'Accuracy =', accuracy
    confusion_img = confusion * 255 / np.amax(confusion)
    cv2.imwrite(filename+str(accuracy)[:4]+'.png', confusion_img)
    
#mats = ['almi',   'cooper', 'ceramic', #'stainless', 
mats = ['alumi',   'copper', 'ceramic', #'stainless', 
        'plaster','paper', 'blackpaper',  'wood',     'cork', 'mdf', 'bamboo', 'cardboard',
         'fabric', 'fakeleather', 'leather', 'carpet',
        #'banana', 'fakebanana', 'fakeapple',
         'polystyrene', 'epvc',   'pvc', 'silicone', 'pp',
        'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm',  'whiteglass', 'sponge']
        
test_mats = ['paper', 'plaster', 'acryl']
#cross_validation_nearest_neighbor_classifier(mats, rep=20, max_index=12, num_training=2, absolute_depth=False)
#cross_validation_many_condition(mats)

cross_validation_nearest_neighbor_classifier_single_distortion(mats, rep=20, max_index=12, axis='depth')
cross_validation_nearest_neighbor_classifier_single_distortion(mats, rep=20, max_index=12, axis='frequency')
cross_validation_nearest_neighbor_classifier_single_distortion(mats, rep=20, max_index=12, axis='both')

#convert_dataset_to_kinect_depth_axis(mats)
#calculate_mean_and_std(mats)

#distance_stats_matrix(mats)

#cross_validation_nearest_neighbor_classifier_probeset(mats, 'paper_concave_angle_test', rep=60, num_test=1, concave=True, max_index_prob=24, max_index=12, linear_stage=False, absolute_depth=False, relative_120=False, normalize=False)
#cross_validation_nearest_neighbor_classifier_probeset(mats, 'wood_angle_test', rep=3, angle=True, max_index_prob=41, max_index=12, linear_stage=False, absolute_depth=False, relative_120=False, normalize=False)