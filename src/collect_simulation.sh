#!/bin/bash

#SEED=12345

#seeds for large data
#SEEDS=(22517 25620 45568 75218 41403 25515 83476 4511 47887 16688)

#seeds for control data
SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)


DT=0.05 # 0.1 0.15 0.2 0.25)
GYM="multi-point" #for both origin and pose
#GYM="multi-point-min-pos-and-origin" #only 2 of unit vectors and origin
#GYM="multi-point-pose" #only the unit vectors
#GYM="end-effector"

#end effector
RUNS=1000

#multi-point large dataset
#RUNS=1000
for seed in ${SEEDS[@]}
do
    #line for large data
    #python3 collect_data.py --seed $seed --runs 30000  --dt $DT --environment $GYM --rand_init

    #modest data for end-effector
    echo "$seed $RUNS $DT $GYM"
    python3 collect_data.py --seed $seed --runs $RUNS  --dt $DT --environment $GYM  --save_dir "data-control"


done 
