#!/bin/bash



#seeds for control data
SEEDS=(12345) # 45 212 458 30 84 11893 27948 8459 984)


DT=0.05 # 0.1 0.15 0.2 0.25)
#GYM="kinova-2d-planar" #for both origin and pose
#GYM="kinova-camera-2d-planar"
GYM="7-dof-kinova"

#kinova 2DOF runs
#RUNS=100
#end effector
RUNS=1000

for seed in ${SEEDS[@]}
do
    #modest data for end-effector
    echo "$seed $RUNS $DT $GYM"
    python3 collect_data.py --seed $seed --runs $RUNS  --dt $DT --environment $GYM  --save_dir "data-control" --gain 0.15


done 
