import numpy as np
import dat2png as reader
from pylab import *
from real_time_nn import *

def plot_data_and_test(test='alumi'):
    p16  = phase2depth(reader.read_float_file('phase_depth_0.dat'), 16.)
    p80  = phase2depth(reader.read_float_file('phase_depth_1.dat'), 80.)
    p120 = phase2depth(reader.read_float_file('phase_depth_2.dat'), 120.)
    acc = reader.read_float_file('accumurate_depth.dat')
    d80 = np.array([b - c for a, b, c in zip(acc, p80, p120) if a > 100])
    d16 = np.array([b - c for a, b, c in zip(acc, p16, p120) if a > 100])
#    plot(d16, d80)
    
    v = np.load('data/'+test+'/3mm.npy')
    plot(-1*v[1], -1*v[0])
    
    show()
#    savefig('test.pdf')
    
plot_data_and_test()