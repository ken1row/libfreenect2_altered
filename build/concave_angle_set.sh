#!/bin/bash

echo "      Press Enter to continue..."
read INPUT

ag=-60
pulse=$((ag*400))
pulsem=$((-400*ag))
python sigma_stage.py --multimove 0 $pulsem $pulse

echo "      Press Enter to continue..."
read INPUT

python sigma_stage.py --multimove 0 $pulse $pulsem

echo "      Press Enter to continue..."
read INPUT

ag=60
pulse=$((ag*400))
pulsem=$((-400*ag))
python sigma_stage.py --multimove 0 $pulsem $pulse

echo "      Press Enter to continue..."
read INPUT

python sigma_stage.py --multimove 0 $pulse $pulsem

