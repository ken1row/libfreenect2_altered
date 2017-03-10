#!/bin/bash

obj="paper_concave_angle_test"
acc=350
speed=1000

echo "CAUTION: Check clearance on stages. They will rotate 180 degree."
echo "         Press Enter to continue..."
read INPUT

python sigma_stage.py -r

python sigma_stage.py --multimove 0 36000 36000

echo "INFO: Set the target object."
echo "      Press Enter to continue..."
read INPUT


#acc=1200
#speed=100

#for ag in {8..80..2}
for ag in {-60..60..5}
do
    pulse=$((ag*400))
    pulsem=$((-400*ag))
    python sigma_stage.py --multimove 0 $pulsem $pulse
    python auto_capture_sigma_stage_sync.py -a $acc -r -1 -s $speed
    python backup_data.py -p $obj
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    sleep 1
    xte "key Escape"
    python sigma_stage.py --multimove 0 $pulse $pulsem
    sleep 3
done     
