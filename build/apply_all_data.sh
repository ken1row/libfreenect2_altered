#!/bin/bash

# 'long02', 'wood00', 'fabric00', 'long03', 'PVC00', 'EPVC00', 'MDF00', 'metal00', 'wood00'
for targ in long02  wood00 fabric00 long03 PVC00 EPVC00 MDF00 matal00
do
    d="data/${targ}"
    echo $d
    python backup_data.py -r $targ
    python analiser.py
    python backup_data.py
done
