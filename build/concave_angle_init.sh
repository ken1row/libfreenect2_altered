#!/bin/bash


echo "CAUTION: Check clearance on stages. They will rotate 180 degree."
echo "         Press Enter to continue..."
read INPUT

python sigma_stage.py -r

python sigma_stage.py --multimove 0 36000 36000

echo "INFO: Set the target object."
echo "      Press Enter to continue..."
read INPUT

