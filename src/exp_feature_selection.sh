#!/bin/bash

SEEDS=(12345 45 212 458 30 84)

ALGORITHMS=("global-neuralnetwork"  "global-neuralnetwork-multitask")
FEATURES=("angles" "position,angles" "position,velocity" "angles,velocity" "position,angles,velocity")


DT=(0.05) # 0.1 0.15 0.2 0.25)

GYM="end-effector"
#GYM="multi-point" #for both origin and pose
#GYM="multi-point-min-pos-and-origin" #only 2 of unit vectors and origin
#GYM="multi-point-pose" #only the unit vectors
for alg in ${ALGORITHMS[@]}
do
    for feats in ${FEATURES[@]}
    do
        for seed in ${SEEDS[@]}
        do
            for dt in ${DT[@]}
            do
                 python3 main.py --seed $seed --runs 110  --policy_name $alg --dt $dt --environment $GYM --partial_state $feats
            done
        done
    done
done 
