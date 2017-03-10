#!/bin/bash

obj="copper"
acc=460
speed=1000

#acc=1200
#speed=100

for i in 65 65
do
    python auto_capture_sigma_stage_sync.py -a $acc -r $i -s $speed
    python backup_data.py -p $obj
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    sleep 3
done
#{60..70}
#for ag in 60 90 120 150 180 210 240 270 300 330
#for ag in 60 65 70 75 80 85 90 95 100 105 110
for ag in {60..70}
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
