#!/bin/bash

obj="wood_angle_test"
acc=350
speed=1000

#acc=1200
#speed=100

#for ag in {8..80..2}
for ag in {82..90..2}
do
    python auto_capture_sigma_stage_sync.py -a $acc -r $ag -s $speed
    python backup_data.py -p $obj
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    sleep 3
done     
