#!/usr/bin/env python

    
import sys
import os
import time
import subprocess



def invoke_demo(com='bin/Protonect'):
    p = subprocess.Popen([com, 'cpu']) # running background.
    time.sleep(3)
    p3 = subprocess.Popen(['python', 'real_time_plot_acc.py'])
    time.sleep(1)
    p2 = subprocess.Popen(['python', 'real_time_nn.py'])
    p_stdout = p.communicate()[0]
    p2.terminate()
    p3.terminate()

if __name__ == "__main__":
    invoke_demo()
     
