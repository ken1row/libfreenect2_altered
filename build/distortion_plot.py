#!/usr/bin/env python

from analiser import *
import scipy
from mpl_toolkits.mplot3d import Axes3D
import random

import matplotlib.font_manager
#times = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf') #, fontproperties=times
times = matplotlib.font_manager.FontProperties(fname='C://Windows/Fonts/times.ttf') #, fontproperties=times


#params = {
#   'axes.labelsize': 20,
#   'text.fontsize': 20,
#   'legend.fontsize': 20,
#   'xtick.labelsize': 20,
#   'ytick.labelsize': 20,
#    }
#pylab.rcParams.update(params)



def plot_angles(l=range(60, 83), file1='phase_depth_0.dat', 
                                    file2='phase_depth_1.dat', 
                                    file3='phase_depth_2.dat'
                                    ):
    for i in l:
        path = os.path.join('data', 'almi'+str(i)+'00')
        pfile1 = os.path.join(path, file1)
        pfile2 = os.path.join(path, file2)
        pfile3 = os.path.join(path, file3)
        d16  = phase2depth(reader.read_float_file(pfile2), 16.)
        d80  = phase2depth(reader.read_float_file(pfile1), 80.)
        d120 = phase2depth(reader.read_float_file(pfile3), 120.)
        x = d120 - d80
        y = d120 - d16
        figure()
        xlim((-5, 15))
        ylim((-100, 0))
        col = range(len(d16))
        scatter(x, y, c=col)
#        plot(x, y, 'o', color=pseudo_color(i - l[0], l[-1] - l[0]))
        xlabel('120 - 80 MHz')
        ylabel('120 - 16 MHz')
        savefig('fig/angle'+str(i)+'.png')
        
def plot_angles2(l=range(60, 83), file1='phase_depth_0.dat', 
                                    file2='phase_depth_1.dat', 
                                    file3='phase_depth_2.dat'
                                    ):
    for i in l:
        path = os.path.join('data', i)
        pfile1 = os.path.join(path, file1)
        pfile2 = os.path.join(path, file2)
        pfile3 = os.path.join(path, file3)
        d16  = phase2depth(reader.read_float_file(pfile2), 16.)
        d80  = phase2depth(reader.read_float_file(pfile1), 80.)
        d120 = phase2depth(reader.read_float_file(pfile3), 120.)
        x = d120 - d80
        y = d120 - d16
        figure()
        xlim((-10, 10))
        ylim((-120, 0))
        col = range(len(d16))
        scatter(x, y, c=col)
#        plot(x, y, 'o', color=pseudo_color(i - l[0], l[-1] - l[0]))
        xlabel('120 - 80 MHz')
        ylabel('120 - 16 MHz')
        savefig('fig/angle'+str(i)+'.png')
        
    
        
def plot_amps(l=range(60, 83), file1='amp_depth_0.dat', 
                                    file2='amp_depth_1.dat', 
                                    file3='amp_depth_2.dat',
                                    file4 = 'accumurate_depth.dat'
                                    ):
    for i in l:
        path = os.path.join('data', 'almi'+str(i)+'00')
        pfile1 = os.path.join(path, file1)
        pfile2 = os.path.join(path, file2)
        pfile3 = os.path.join(path, file3)
        pfile4 = os.path.join(path, file4)
        acc = reader.read_float_file(pfile4)
        a16  = phase2depth(reader.read_float_file(pfile2), 16.) / acc
        a80  = phase2depth(reader.read_float_file(pfile1), 80.) / acc
        a120 = phase2depth(reader.read_float_file(pfile3), 120.) / acc
        r = np.sqrt(a16*a16+a80*a80)
        theta = np.arccos(a16 / r)
        norm = np.sqrt(a16*a16+a80*a80+a120*a120)
        phi = np.arccos(r / norm)
        figure()
        xlim((.183, .195))
        ylim((.123, .133))
        col = range(len(a16))
        scatter(theta, phi, c=col)
#        plot(x, y, 'o', color=pseudo_color(i - l[0], l[-1] - l[0]))
        xlabel('Theta (Azimuth)')
        ylabel('Phi (Zenith)')
        savefig('fig/amp'+str(i)+'.png')
        
def reduce_resolution(seq, acc, ratio):
    new = np.zeros(seq.shape)
    for i in range(len(seq)/ratio):
        s = seq[i*ratio:i*ratio+ratio]
        a = seq[i*ratio:i*ratio+ratio]
        new[int(i*ratio+ratio/2)] = (sum(s*a) / sum(a))
    return new
        
def _base_depth(base='base00', targ='paper00', raw=False, rel=False, acc_thresh=200,
                acc_minus_mode=False, sub_sample=False, sub_sample_ratio=5,
                partial=False, partial_range=(100, 200),
                low_res=False, res_ratio=20):
    file1_base = os.path.join('data', base, 'phase_depth_0.dat')        
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')        
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')      
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')        
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')        
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')        
    acc = reader.read_float_file(os.path.join('data', targ, 'accumurate_depth.dat'))
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    
    if low_res:
        d16_base  = reduce_resolution(d16_base , acc, res_ratio)
        d80_base  = reduce_resolution(d80_base , acc, res_ratio)
        d120_base = reduce_resolution(d120_base, acc, res_ratio)
        d16  = reduce_resolution(d16 , acc, res_ratio)
        d80  = reduce_resolution(d80 , acc, res_ratio)
        d120 = reduce_resolution(d120, acc, res_ratio)
    
    if partial:
        d16_base  = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d16_base)])
        d80_base  = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d80_base)])
        d120_base = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d120_base)])
        d16  = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d16)])
        d80  = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d80)])
        d120 = np.array([0 if idx < partial_range[0] or idx > partial_range[1] else v for idx, v in enumerate(d120)])
        
    if sub_sample:
        d16_base  = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d16_base)])
        d80_base  = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d80_base)])
        d120_base = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d120_base)])
        d16  = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d16)])
        d80  = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d80)])
        d120 = np.array([0 if random.randint(0, sub_sample_ratio) > 0 else v for idx, v in enumerate(d120)])
    
    if raw:
        return d16, d80, d120, d16_base, d80_base, d120_base
        
    d16  -= d16_base
    d80  -= d80_base
    d120 -= d120_base
    if acc_minus_mode:
        d16  = np.array([0 if a < acc_thresh else v for a, v in zip(acc, d16)])
        d80  = np.array([0 if a < acc_thresh else v for a, v in zip(acc, d80)])
        d120 = np.array([0 if a < acc_thresh else v for a, v in zip(acc, d120)])
    if rel:
        return d16, d80, d120
    x = [0 if a == 0 or b == 0 else a - b for a, b in zip(d120, d80)]
    y = [0 if a == 0 or b == 0 else a - b for a, b in zip(d120, d80)]
    return x, y
    
def _base_phase(base='base00', targ='paper00'):
    file1_base = os.path.join('data', base, 'phase_depth_0.dat')        
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')        
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')      
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')        
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')        
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')        
    def phase2depth(x, y):
        return x
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    
    return d16, d80, d120, d16_base, d80_base, d120_base
    
def plot_depth_dist(l=['acryl02']):
    for i in l:
        d16, d80, d120, b16, b80, b120 = _base_depth('almi02', i, True)
        x = np.arange(600., 1200., .25)
        figure()
#        plot(x, d16)
#        plot(x, d80)
#        plot(x, d120)
#        plot(x, d16 - b16)
#        plot(x, d80 - b80)
#        plot(x, d120 - b120)
#        plot(x, b16 - b120)
#        plot(x, b80 - b120)
#        plot(x, b120 - b120)
        plot(x, d16 - d120)
        plot(x, d80 - d120)
        plot(x, d120 - d120)
#        plot(x, b16)
#        plot(x, b80)
#        plot(x, b120)
#        ylim((600,1300))
    
def plot_base_depth2(l=['acryl02'], mode=0):
    figure()
    for i in l:
        x, y = _base_depth('almi02', i)
        col = range(len(x))
        if mode==0:
            scatter(x, y, c=col)
        else:
            plot(x, y, label=i)
    if mode>0:
        legend(loc='upper left')
    savefig('fig/base_depth'+str(mode)+'.png')

def plot_base_depth(base='base00', targ='paper00'):
    file1_base = os.path.join('data', base, 'phase_depth_0.dat')        
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')        
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')      
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')        
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')        
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')        
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.) - d16_base
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) - d80_base
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) - d120_base
    
    x = d120 - d80
    y = d120 - d16
    figure()
    col = range(len(d16))
    scatter(x, y, c=col)
    xlabel('120 - 80 MHz')
    ylabel('120 - 16 MHz')
    savefig('fig/depth_base.png')
    
def plot_phases_along_stage(d='acryl00', base='almi00', start=0, end=1650, relative=False):
    d16, d80, d120, b16, b80, b120 = _base_phase(base, d)
    pulses = reader.read_float_file('pulse_bins.dat')
    offset = b120[0]
    pulses_2pi_120 = phase2depth(2. * np.pi, 120.) * 40 / 2.
    gt = np.linspace(offset, pulses[-1] / pulses_2pi_120 + offset, len(pulses))
#    plot(pulses, gt)
#    plot(pulses, d120)
    figure()
    if relative:
        plot(d120-b120, label='120')
        plot(d80 - b80, label='80')
        plot(d16 - b16, label='16')
        ylim((0, 0.2))
        legend()
    else:
        plot(b120, color='blue')
        plot(d120, color='red')
        plot(b80, color='blue')
        plot(d80, color='red')
        plot(b16, color='blue')
        plot(d16, color='red')
#    ylim((2.5, 7.))
    xlim((start, end))
    savefig('fig/phase_difference'+d+'.png')

def plot_depths_along_stage(d='acryl00', base='base00', start=0, end=1650, relative=False):
    d16, d80, d120, b16, b80, b120 = _base_depth(base, d, True)
    pulses = reader.read_float_file('pulse_bins.dat')
#    X, Y = np.meshgrid(pulses, pulses)
#    Z = [[a, b, c] for a, b, c in zip(d120 - b120, d80 - b80, d16 - b16)]
    rcParams.update({'font.size': 18, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
#    figure()
    col = range(len(pulses))
    if relative:
        figure(figsize=(6, 3))
        plot(d120-b120, label='freq. 1')
        plot(d80 - b80, label='freq. 2')
        plot(d16 - b16, label='freq. 3')
        ylim((-5, 70))
        xlim((120, 1700))
        xlabel('The ground-truth depth [mm]', fontproperties=times)
        ylabel('Depth distortion [mm]', fontproperties=times)
        xticks(range(120, 1700, 300), range(600, 1280, 120), fontproperties=times)
        yticks(range(0, 71, 10), range(0, 71, 10), fontproperties=times)
        legend(prop=times)
        
    else:
        fig = figure()
        ax = Axes3D(fig)
    #    ax.scatter3D(X.ravel(), Y.ravel(), Z)
        a, b, c = (d120 - b120, d80 - b80, d16 - b16)
        ax.scatter3D(a[start:end], b[start:end], c[start:end], c=col[start:end])
        xlabel('120 MHz')
        ylabel('80 MHz')
        ax.set_zlabel('16 MHz')
    #    ax.set_xlim((-5, 20))
    #    ax.set_ylim((-5, 35))
    #    ax.set_zlim((-5, 70))
    savefig('fig/spiral_3d'+d+'.pdf', bbox_inches='tight', pad_inches=0.1)
    
def DPmatching_core(target='almi01', probe='almi01', base='base00', save_img=False):
    pass
    t16, t80, t120 = _base_depth(base, target, rel=True)
    d16, d80, d120 = _base_depth(base, probe, rel=True)
    targ_vecs = np.vstack((t16, t80, t120)).T
    probe_vecs = np.vstack((d16, d80, d120)).T
    dim = len(t16)
#    targ_vecs = np.roll(targ_vecs, 100, axis=0) # test code
    return __DPmatching_core(targ_vecs, probe_vecs, dim, target, probe, True, save_img)
    
def __DPmatching_core(targ_vecs, probe_vecs, dim, target='almi01', probe='almi01', cache=False, save_img=False, refresh=False):
    fn1 = os.path.join('data', target, 'dpm_'+probe+'cost.npy')
    fn2 = os.path.join('data', target, 'dpm_'+probe+'back_vec.npy')
    if os.path.exists(fn1) and os.path.exists(fn2) and cache and not refresh:
        cost_mat = np.load(fn1)
        back_vec_mat = np.load(fn2)
        route_mat = np.zeros(cost_mat.shape)
    else:
        cost_mat = np.zeros((dim, dim))
        back_vec_mat = np.zeros((dim, dim)) # 0 for diagonal, 1 for upward, -1 for left-side
        route_mat = np.zeros((dim, dim)) # 1 for minimum route
        cost_left = 1.
        cost_up = 1.
        cost_diag = 0.
        cost_vec = (cost_left, cost_diag, cost_up)
        for idx in range(dim):
            targ_vec = targ_vecs[idx]
    #        cost_mat[idx] = [np.linalg.norm(p - targ_vec) for p in probe_vecs]
            for idx2 in range(dim):
                cost_here = np.linalg.norm(probe_vecs[idx2] - targ_vec)
                if have_zero(probe_vecs[idx2]) or have_zero(targ_vec):
                    cost_here = 0
                if idx == 0:
                    if idx2 == 0:
                        cost_mat[idx, idx2] = cost_here
                    else:
                        cost_mat[idx, idx2] = cost_here + cost_mat[idx, idx2-1] + cost_left
                        back_vec_mat[idx, idx2] = -1
                if idx2 == 0:
                    cost_mat[idx, idx2] = cost_here + cost_mat[idx-1, idx2] + cost_up
                    back_vec_mat[idx, idx2] = 1
                else:
                    comp = (cost_mat[idx, idx2-1], cost_mat[idx-1, idx2-1], cost_mat[idx-1, idx2])
                    argmin = np.argmin(comp)
                    cost_mat[idx, idx2] = cost_here + comp[argmin] + cost_vec[argmin]
                    back_vec_mat[idx, idx2] = argmin - 1
        if cache:
            np.save(fn1, cost_mat)
            np.save(fn2, back_vec_mat)
    # get route
    p_x = dim -1
    p_y = dim -1
    translation_list = []
    while(p_x>0 and p_y>0):
        route_mat[p_y, p_x] = 1
        translation_list.append(p_x - p_y)
        if back_vec_mat[p_y, p_x] == -1:
            p_x -= 1
        elif back_vec_mat[p_y, p_x] == 1:
            p_y -= 1
        else:
            p_x -= 1
            p_y -= 1
    max_cost = np.amax(cost_mat)
#    print max_cost, np.mean(cost_mat)
    mode, count = scipy.stats.mstats.mode(translation_list)
    print 'Levenshtein Distance:', cost_mat[-1, -1], 'Mode:', mode
    if save_img:
        cost_mat2 = cost_mat * 255 / max_cost
        cost_w_route = [[v, v, v] if r < 1 else [0, 0, 255] for v, r in zip(cost_mat2.ravel(), route_mat.ravel())]
        cv2.imwrite('fig/cost.png', np.array(cost_w_route).reshape((dim, dim, 3)))
    return cost_mat, back_vec_mat, route_mat, cost_mat[-1,-1], np.median(translation_list), mode[0]
    
def have_zero(array):
    return any([True if v==0 else False for v in array])
    
def eval_l2_after_dp_core(target='almi01', probe='almi01', base='base00', refresh=False, two_d=False, skip_dp=False):
    pass
    targets = _base_depth(base, target, rel=(not two_d), acc_minus_mode=True)
    probes = _base_depth(base, probe,  rel=(not two_d), acc_minus_mode=True)
    targ_vecs = np.vstack(targets).T
    probe_vecs = np.vstack(probes).T
    dim = len(targets[0])
#    targ_vecs = np.roll(targ_vecs, 100, axis=0) # test code
    if not skip_dp:
        a, b, c, cost, median, mode = __DPmatching_core(targ_vecs, probe_vecs, dim, target, probe, cache=True, refresh=refresh)
        targ_vecs = np.roll(targ_vecs, int(mode), axis=0)
    l2norms = np.linalg.norm(probe_vecs - targ_vecs, axis=1)
    valid_mask = [0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(targ_vecs, probe_vecs)]
    l2norms *= np.array(valid_mask)
    return sum(l2norms)
    
def eval_l2_after_DPm_all(targ=['acryl00'], probe=['almi01'], base='base00', refresh=False,
                          two_d=False, skip_dp=False, save_img=False, confusion=None):
    eval_mat = np.zeros((len(targ), len(probe)))
    if confusion is None:
        confusion = np.zeros((len(targ), len(probe)))
    for i, t in enumerate(targ):
        for j, p in enumerate(probe):
#            print t, 'vs.', p, 
            cost = eval_l2_after_dp_core(t, p, base, refresh, two_d, skip_dp)
#            print 'Cost:', cost
            eval_mat[i, j] = cost
        argmin = np.argmin(eval_mat[i])
        confusion[i, argmin] += 1.
#        print t, 'is classfied as', probe[argmin]
        if not probe[argmin].startswith(t[:-3]):
            print t, '\tis miss-classfied as', probe[argmin], ' \tnot', probe[i]
    if save_img:
        np.save('eval_mat.npy', eval_mat)
        eval_mat *= 255 / np.amax(eval_mat)
        confusion_img = confusion * 255 / np.amax(confusion)
        cv2.imwrite('fig/eval_mat.png', eval_mat)
        cv2.imwrite('fig/confusion_mat.png', confusion_img)
    return confusion
    
def convert_axis_S2D(a, b, c, d, e, f, acc, depths, start=600, end=1200, points=200):
    bins = np.linspace(start, end, points)
    acc_new = np.zeros(bins.shape)
    data = np.zeros([6, len(bins)])
    for idx, d in enumerate(depths):
        ac = acc[idx]
        idx_new = min(len(bins), max(0, int((d-start)/(bins[1] - bins[0]))))
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
    
def load_data(target, base='base00', absolute_depth=True, linear_stage=True, relative_phase=True, normalize_metric=True, guarantee=200):
    
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
    depths = reader.read_float_file(os.path.join('data', targ, 'depth_out.dat'))
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    a16  = reader.read_float_file(file2a_targ)
    a80  = reader.read_float_file(file1a_targ)
    a120 = reader.read_float_file(file3a_targ)

    
    if absolute_depth:
        d16 -= d16_base
        d80 -= d80_base
        d120-= d120_base
    
    if not linear_stage:
        d16, d80, d120, a16, a80, a120, acc = convert_axis_S2D(d16, d80, d120, a16, a80, a120, acc, depths)
        
    if relative_phase:
        d16 -= d120
        d80 -= d120
    
    if normalize_metric:
        pass
    
    if isinstance(guarantee, int):
        d16 = np.array([v for v, a in zip(d16, acc) if acc > guarantee])
        
    return d16, d80, d120, a16, a80, a120
    
def cross_validation_nearest_neighbor_classifier(materials, rep=10, max_index=1, absolute_depth=True, linear_stage=True, relative_phase=True, amplitude=False, depth=True, nomalize=False):
    confusion = np.zeros((len(materails), len(materials)))
    
def make_confusion_mat(materials=[], rep=10, max_index=1, fn=''):
    confusion = np.zeros((len(materials), len(materials)))
    for idx in range(rep):
        targ = [m + str(random.randint(0, max_index)).zfill(2) for m in materials]
        prob = []
        for m in materials:
            p = m + str(random.randint(0, max_index)).zfill(2)
            while p in targ:
                p = m + str(random.randint(0, max_index)).zfill(2)
            prob.append(p)
        confusion = eval_l2_after_DPm_all(targ, prob, two_d=True, skip_dp=True, confusion=confusion)
    np.save('fig/confusion'+fn+'.npy', confusion)
    diag = np.diag(confusion)
    print sum(diag)
    confusion_img = confusion * 255 / np.amax(confusion)
    cv2.imwrite('fig/confusion_mat'+fn+'.png', confusion_img)
    
def accuracy_from_confusion_mat(fname):
    confusion = np.load(fname)
    diag = np.diag(confusion)
    print sum(diag), np.sum(confusion), 100. * sum(diag) / np.sum(confusion)
    
def eval_l2_low_res(targ='fabric01', prob=[]):
    for p in prob:
        print p, '\t', eval_l2_after_dp_core(targ, p, two_d=True, skip_dp=True)
        
    for pr in prob:
        targets = _base_depth('base00', targ, rel=False, acc_minus_mode=True, partial_range=(400, 800))#low_res=True)
        probes  = _base_depth('base00', pr   , rel=False, acc_minus_mode=True)
        t_vec = np.vstack(targets).T
        p_vec = np.vstack(probes).T
        dim = len(targets[0])
        l2norms = np.linalg.norm(p_vec - t_vec, axis=1)
        valid_mask = [0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(t_vec, p_vec)]
        l2norms *= np.array(valid_mask)
        print pr, sum(l2norms)
#plot_base_depth()
        
#plot_angles(l=range(60,82))
#plot_angles2(l=['almi02', 'paper00', 'acryl02', 'fabric_m00'])
#plot_depth_dist()
#plot_base_depth2(l=['almi02', 'paper00', 'acryl02', 'fabric00'])
#plot_base_depth2(l=['almi02', 'paper00', 'acryl02', 'fabric00'], mode=1)
#plot_amps(l=range(60,82))
#plot_phases_along_stage('fakeapple00', start=90, relative=True)
#plot_depths_along_stage('fakebanana00', start=90)
#DPmatching_core()
#DPmatching_core('almi02', 'almi01', save_img=True)
targ2 = ['almi08', 'acryl07', 'wood07', 'paper07', 'fabric07', 'whiteglass07', 'polystyrene07']
targ = ['almi02', 'acryl01', 'wood01', 'paper01', 'fabric01', 'whiteglass01', 'polystyrene01']
prob = ['almi01', 'acryl00', 'wood00', 'paper00', 'fabric00', 'whiteglass00', 'polystyrene00']
#mats = ['almi',   'cooper', 'ceramic', #'stainless', 
mats = ['alumi',   'copper', 'ceramic', #'stainless', 
        'paper', 'blackpaper',  'wood',     'cork', 'mdf', 'bamboo', 'cardboard',
         'fabric', 'fakeleather', 'leather', 'carpet',
        'banana', 'fakebanana', 'fakeapple',
        'plaster', 'polystyrene', 'epvc',   'pvc', 'silicone', 'pp',
        'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm',  'whiteglass', 'sponge']
targ = ['acryl01', 'wood01', 'whiteglass01', 'polystyrene01']
targ2= ['acryl07', 'wood07', 'whiteglass07', 'polystyrene07']
for t in targ:
    plot_depths_along_stage(t, start=100, relative=True)
for t in targ2:
    plot_depths_along_stage(t, start=100, relative=True)
#for p in prob:
#    plot_depths_along_stage(p, start=100, relative=True)
#eval_l2_after_DPm_all(prob, targ, skip_dp=True, two_d=True)
#make_confusion_mat(mats, rep=12*len(mats), max_index=10, fn='l_rd2')
#print len(mats)
#accuracy_from_confusion_mat('fig/confusionl_rd2.npy')        
        
#eval_l2_after_dp_core()
#eval_l2_low_res(prob=prob)
#bnn = ['banana', 'fakebanana',
#        'paper',   'wood',   'fabric',   'cork', #'MDF'
#        'fakeleather',
#        'polystyrene',   'pvc', 'silicone',
#        'acryl',   'whiteglass']
#make_confusion_mat(bnn, rep=12*len(mats), max_index=10, fn='banana')