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

import sysv_ipc

def esc_press():
#    p = Popen(['xte'], stdin=PIPE)
#    p.communicate(input='key Escape')
    subprocess.call(['xte', 'key Escape'])
        
def __control_c_stage_stop_callback(port, p1, p2):
    def callback(no, fr):
        stage.emergency_stop(port)
        p1.terminate()
        p2.terminate()
        time.sleep(.1)
        stage.resume(port, stages=2)
        raise KeyboardInterrupt
    return callback
    
last_check_time = time.time()
capture_start_time = time.time()
def get_mean_accuracy(file1, goal=460):
    global last_check_time
    now = time.time()
    if now - last_check_time < 3:
        return -1
    try:
        acc = reader.read_float_file(file1)
        val = np.mean(acc)
        if not val == 0:
            estimated_time = int((now - capture_start_time) * (goal - val) / val)
            m, s = divmod(estimated_time, 60)
        else:
            m, s = (-1, 0)
        print ' Accuracy Mean:', int(val), 'Estimated time', m, ':', s
        last_check_time = now
        return val
    except:
        print ' Failed accuracy evaluation.'
#        raise
        return -1

def capture_and_move_stage(accuracy=150, speed=1000, file1='accumurate_depth.dat', com='bin/Protonect', angle=70, shm_key=23456):
    global capture_start_time
    p = subprocess.Popen([com, 'cpu']) # running kinect in background.
    # Connect to the stage controller.
    port = stage.open()
    # Initialize
    stage.verbose_level = 1
    if angle > 0:
        stage.mechanical_resume(port)
    else:
        stage.mechanical_resume_single(port, 1)
    stage.set_drive_speed(port, 1, 1000,10000,1000)
    if angle > 0:
        stage.set_drive_speed(port, 2, 1000,20000,1000)
        deg = angle * 1000
        pls = deg * 2 / 5
        stage.move(port, pls, stage=2)
	# shared memory
    shm = sysv_ipc.SharedMemory(shm_key)
    direction = 1
    stage.jog(port, 1, direction)
#    time.sleep(5)
    p2 = subprocess.Popen(['python', 'real_time_plot.py'])
    p3 = subprocess.Popen(['python', 'real_time_plot_acc_sync.py'])
    signal.signal(signal.SIGINT, __control_c_stage_stop_callback(port, p2, p3))
    capture_start_time = time.time()
    while(get_mean_accuracy(file1, accuracy) < accuracy):
        # Max steps - 63000
        pulses = stage.get_status(port)[0][0]
        shm.write(str(pulses)+'\0')
        if (pulses < 0 and direction < 0) or (pulses > 69000 and direction > 0):
            direction *= -1
            stage.stop(port)
            time.sleep(0.1)
            stage.jog(port, 1, direction)
        time.sleep(0.01)
    
    shm.write('0\0')    
    stage.stop(port)
#    p.send_signal(signal.CTRL_C_EVENT)
    p.send_signal(signal.SIGINT)
#    p.terminate()
    time.sleep(.1)
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
    print
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
    parser.add_argument('-a', '--accuracy', type=int, default=460, help="Target accumulate number. Continue caputure while mean of accumulate number is less than this value.")
    parser.add_argument('-r', '--angle', type=int, default=65, help="Angle of the target material")
    parser.add_argument('-s', '--speed', type=int, default=1000, help="Jog speed")
    args = parser.parse_args()
    capture_and_move_stage(args.accuracy, speed=args.speed, angle=args.angle)
#    time.sleep(2)
#    get_mean_accuracy('accumurate_depth.dat')
     
