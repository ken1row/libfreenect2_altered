#!/usr/bin/env python

    
import sys
import os
import time
import dat2png as reader
import math
import subprocess
from subprocess import Popen, PIPE

import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import sigma_stage as stage
import signal

def esc_press():
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input='key Escape')
        
def __control_c_stage_stop_callback(port, p1, p2):
    def callback(no, fr):
        stage.emergency_stop(port)
        p1.terminate()
        p2.terminate()
        time.sleep(.01)
        stage.resume(port, stages=2)
        raise KeyboardInterrupt
    return callback
    
def get_mean_accuracy(file1):
    try:
        acc = reader.read_float_file(file1)
        return np.mean(acc)
    except:
        return -1

def capture_and_move_stage(accuracy=150, speed=1, file1='accumurate_depth.dat', com='bin/Protonect', angle=70):
    # Connect to the stage controller.
    port = stage.open()
    # Initialize
    stage.verbose_level = 2
    if angle > 0:
        stage.mechanical_resume(port)
    stage.set_drive_speed(port, 1, 1000,10000,1000)
    if angle > 0:
        stage.set_drive_speed(port, 2, 1000,20000,1000)
        deg = angle * 1000
        pls = deg * 2 / 5
        stage.move(port, pls, stage=2)
    p = subprocess.Popen([com, 'cpu']) # running background.
    direction = 1
    stage.jog(port, 1, direction)
    time.sleep(5)
    p2 = subprocess.Popen(['python', 'real_time_plot.py'])
    p3 = subprocess.Popen(['python', 'real_time_plot_acc.py'])
    signal.signal(signal.SIGINT, __control_c_stage_stop_callback(port, p2, p3))
    while(get_mean_accuracy(file1) < accuracy):
        # Max steps - 63000
        if stage.is_stopped_at_limit_sensor_or_ready(port, 1):
            direction *= -1
            stage.stop(port)
            stage.jog(port, 1, direction)
        time.sleep(1)
        if stage.get_status(port)[0][0] > 67000 and direction > 0:
            direction *= -1
            stage.stop(port)
            time.sleep(0.01)
            stage.jog(port, 1, direction)
    stage.stop(port)
#    p.send_signal(signal.CTRL_C_EVENT)
    p.send_signal(signal.SIGINT)
#    p.terminate()
    time.sleep(.01)
    if angle > 0:
        stage.resume(port, stages=2)
    else:
        stage.resume(port, stages=1)
    p2.terminate()
    p3.terminate()
#    try:
#        for i in range(5):
#            time.sleep(1)
#            esc_press()
#    except:
#        raise
#    
#def main():
##    watcher = QFileSystemWatcher()
##    watcher.fileChanged.connect(on_file_changed)
##    watcher.addPath(file1)    
#    port = stage.open()
#    
#def on_file_changed():
#    pass

if __name__ == "__main__":
#    acc = reader.read_float_file('accumurate_depth.dat')
#    x = reader.read_float_file('depth_bins.dat')
#    import matplotlib.pyplot as plt
#    plt.plot(x, acc)
    import argparse
    parser = argparse.ArgumentParser(description='Move stages and capture images for DSLR.')
    parser.add_argument('-a', '--accuracy', type=int, default=480, help="Target accumulate number. Continue caputure while mean of accumulate number is less than this value.")
    parser.add_argument('-r', '--angle', type=int, default=65, help="Angle of the target material")
    args = parser.parse_args()
    capture_and_move_stage(args.accuracy, angle=args.angle)
     
