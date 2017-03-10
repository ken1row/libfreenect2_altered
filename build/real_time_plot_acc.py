#!/usr/bin/env python

    
import sys
import os
import time
import dat2png as reader
import math

import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
    
class LinePlot():
    '''Plot widget for Qt, wrapping plot of pyplot.
    
    Parameters
    ----------
    parent : QtWidget?, optional
        Parent of this widget. I'm not sure the type of the parent object.
    '''
    def __init__(self, parent=None):
        
        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        self.dpi = 100
        self.fig = Figure((5,4), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)    #pass a figure to the canvas
        self.canvas.setParent(parent)        
        self.axes = self.fig.add_subplot(111)        
        
    def set_labels(self, x_label, y_label):
        ''' Set labels for each axis.
        
        This function seems not to work.
        
        Parameters
        ----------
        x_label : str
            Label for x axis.
        y_label : str
            Label for y axis.
        '''
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
#        self.fig.xlabel(x_label)
#        self.fig.ylabel(y_label)

    def set_lims(self, xlim, ylim):
        '''Set plot range of the graph.
        
        Parameters
        ----------
        xlim : tuple(2)
            range of x values.
        ylim : tuple(2)
            range of y values.
        '''
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)

    def on_draw(self, x, y):
        """Redraw the figure. Range of the figure is automatically adjusted.
        
        Parameters
        ----------
        x : arraylike(N)
            X values of the data.
        y : arraylike(N)
            Y values of the data.
        """
        self.axes.clear()
        self.axes.grid()

#        col = np.linspace(0, 1, len(x))
#        self.set_lims((min(x), max(x)),(min(y), max(y)))
        self.axes.plot(x, y)

        self.canvas.draw()
    

class AppFormNectHist(QMainWindow):
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
    def __init__(self, parent=None, file1='accumurate_depth.dat', 
                                    file2='depth_bins.dat', 
                                    wait_for_file_close=.01):
        QMainWindow.__init__(self, parent)
        self.file1 = file1
        self.file2 = file2
        self.wait_for_file_close = wait_for_file_close
        
        self.creat_main_window()
        
        # Initial plot
        self.acc  = reader.read_float_file(self.file1)
#        self.x    = reader.read_float_file(self.file2)
        self.x = np.arange(len(self.acc))
        self.plot.set_labels('120 - 80 MHz', '120 - 16 MHz')
        self.plot.on_draw(self.x, self.acc)
        
        # Add watchdog for each file
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_file_changed)
        self.mtime1 = os.path.getmtime(self.file1)
        self.watcher.addPath(self.file1)
        
    def creat_main_window(self):
        self.main_frame = QWidget()
        self.plot = LinePlot(self.main_frame)

        #set layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.plot.canvas)    #add canvs to the layout

        self.main_frame.setLayout(vbox)

        #set widget
        self.setCentralWidget(self.main_frame)
    
    def _on_file_changed(self):
        # This signal includes modified, renamed, and removed events. Should be checked.
        if not os.path.exists(self.file1):
            qApp.quit()
            return
        newtime1 = os.path.getmtime(self.file1)
        if self.mtime1 < newtime1:
            self.mtime1 = newtime1
            time.sleep(self.wait_for_file_close)
            self.acc  = reader.read_float_file(self.file1)
        self.plot.on_draw(self.x, self.acc)
        
def main(args):
    app = QApplication(args)
    form = AppFormNectHist()
    form.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
#    acc = reader.read_float_file('accumurate_depth.dat')
#    x = reader.read_float_file('depth_bins.dat')
#    import matplotlib.pyplot as plt
#    plt.plot(x, acc)
    main(sys.argv)
     