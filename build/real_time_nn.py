#!/usr/bin/env python

    
import sys
import os
import time
import dat2png as reader
import math

import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *

mats = ['alumi',   'copper', 'ceramic', #'stainless', 
        'paper', 'blackpaper',  'wood',     'cork', 'mdf', 'bamboo', 'cardboard',
         'fabric', 'fakeleather', 'leather', 'carpet',
        #'banana', 'fakebanana', 'fakeapple',
        'plaster', 'polystyrene', 'epvc', #  'pvc', 'silicone', 'pp',
        'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm',  'whiteglass']
        
mat_label = ['Metal - Aluminum',   'Metal - Copper', 'Ceramic', #'stainless', 
        'Paper', 'Flock paper',  'Wood - Natural', 'Wood - Cork', 'Wood - MDF', 'Wood - Bamboo', 'Paper - Cardboard',
         'Fabric - Cotton', 'Fabric - Fake leather', 'Fabric - Leather', 'Fabric - Carpet',
        #'Plant - Banana', 'Plastic - Unknown', 'Plastic - Unknown',
        'Plaster', 'Plastic - PS', 'Plastic - E-PVC', #  'Plastic - PVC', 'Plastic - Silicone', 'Plastic - PP',
        'Plastic - Acryl', 'Plastic - Acryl, 3mm', 'Plastic - Acryl, 2mm', 'Plastic - Acryl, 1mm',  'Diffusion glass']
        
test_mats = ['paper', 'plaster', 'acryl']
ignored = ['fakebanana', 'fakeapple', 'banana', 'cardboard', 'polystyrene']

#pi2 = math.pi #/ 2.
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
#    if omega_MHz > 100:
#        phase = np.array([p + 2.* math.pi if p < pi2 else p for p in phase])
    return c_mm_ns * phase / (2. * math.pi) * 1000. / omega_MHz / 2.
    
 
def have_zero(array):
    return any([True if v==0 else False for v in array])
 
def valid_l2_norm(vec1, vec2):
    l2 = np.linalg.norm(vec1 - vec2, axis=0)
    valid = np.array([0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(vec1.T, vec2.T)])
    return sum(l2 * valid)          

class AppFormNect(QMainWindow):
    ''' Main application GUI form for scatter plot. Watches Protonect output files and calculate phase values.
    
    Attributes
    ----------
    x_file : str
        Filename of the x values of plot data.
    y_file : str
        Filename of the y values of plot data.
    wait_for_file_close : float
        Wait time between file modified detection and file open for load data.
    scatterplot : ScatterPlot
        Plot widget wrapping matplotlib.
        
    Examples
    --------
    >>> app = QApplication(sys.argv)
    >>> form = AppForm()
    >>> form.show()
    >>> sys.exit(app.exec_())
    '''
#    def __init__(self, parent=None, file1='phase_depth_0_rt.dat', 
#                                    file2='phase_depth_1_rt.dat', 
#                                    file3='phase_depth_2_rt.dat', 
    def __init__(self, parent=None, file1='phase_depth_0.dat', 
                                    file2='phase_depth_1.dat', 
                                    file3='phase_depth_2.dat', 
                                    wait_for_file_close=.01,
                                    accuracy=100,
                                    debug=False):
        QMainWindow.__init__(self, parent)
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.wait_for_file_close = wait_for_file_close
        self.accuracy = accuracy
        
#        self.creat_main_window()
        self.create_label_window()
        
        # Add watchdog for each file
        if not debug:
            self.watcher = QFileSystemWatcher(self)
            self.watcher.fileChanged.connect(self._on_file_changed)
#            self.watcher.addPath(self.file1)
#            self.watcher.addPath(self.file2)
            self.watcher.addPath(self.file3)
            self.load_database()
            self.estimate_material()

        
        
    def create_label_window(self):
        # window
        self.main_frame = QWidget()
        self.setGeometry(750, 0, 800, 500)
        self.setWindowTitle('Material Classifier')
        
        # layout
        vbox = QVBoxLayout()
        
        # widgets
        self.label = QLabel('Put material.')
        self.label.setFont(QFont('SansSerif', 40))
        self.mat2 = QLabel('rank2')
        self.mat2.setFont(QFont('SansSerif', 32))
        self.mat3 = QLabel('rank3')
        self.mat3.setFont(QFont('SansSerif', 28))
        self.mat4 = QLabel('rank4')
        self.mat4.setFont(QFont('SansSerif', 24))
        self.mat5 = QLabel('rank5')
        self.mat5.setFont(QFont('SansSerif', 20))
        
        # set all
        vbox.addWidget(self.label)
        vbox.addWidget(self.mat2)
        vbox.addWidget(self.mat3)
        vbox.addWidget(self.mat4)
        vbox.addWidget(self.mat5)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
    
    def _on_file_changed(self):
        time.sleep(self.wait_for_file_close)
        self.estimate_material()
        
    def load_database(self):
        self.materials = []
        self.training = []
        for idx, mat in enumerate(mats):
            if mat in ignored:
                continue
            self.materials.append(mat_label[idx])
            self.training.append(np.load('data/'+mat+'/3mm.npy'))
#        self.materials = mat_label
#        self.training = [np.load('data/'+m+'/3mm.npy') for m in mats]
        
    def load_file(self):
        flag = True
        flag &= os.path.exists(self.file1)
        flag &= os.path.exists(self.file2)
        flag &= os.path.exists(self.file3)
        self.all_file_exists = flag
        if flag:
            self.p16  = phase2depth(reader.read_float_file(self.file2), 16.)
            self.p80  = phase2depth(reader.read_float_file(self.file1), 80.)
            self.p120 = phase2depth(reader.read_float_file(self.file3), 120.)
            self.acc = reader.read_float_file('accumurate_depth.dat')
            self.d80 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p80, self.p120)])
            self.d16 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p16, self.p120)])
        
    def clear_labels(self):
        self.mat2.setText('')
        self.mat3.setText('')
        self.mat4.setText('')
        self.mat5.setText('')
        
    def estimate_material(self):
        self.load_file()
        if not self.all_file_exists:
            self.clear_labels()
            self.label.setText('Empty. Put material.')
            return
        
        valid_pixels = len([True for v in self.acc if v > self.accuracy])
        if valid_pixels < 20:
            self.clear_labels()
            if valid_pixels == 0:
                self.label.setText('Put material.')
            else:
                self.label.setText('Measureing. ')
            return
        
        test_vec = np.vstack((self.d16, self.d80))
        training = self.training
        costs = [valid_l2_norm(test_vec, v) for v in self.training]
#        argmin = np.argmin(costs)
        ranking = np.argsort(costs)
        self.label.setText(self.materials[ranking[0]])
        self.mat2.setText(self.materials[ranking[1]])
        self.mat3.setText(self.materials[ranking[2]])
        self.mat4.setText(self.materials[ranking[3]])
        self.mat5.setText(self.materials[ranking[4]])
        
def main(args):
    app = QApplication(args)
    form = AppFormNect()
    form.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main(sys.argv)
     
