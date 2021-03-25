#!/bin/bash

#SEED=12345

SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)
#ALGORITHM="blackbox-kinematics-custom" 
ALGORITHM="blackbox-kinematics-custom-fixed" 



#ALGORITHM=("multipoint-inversejacobian")
DT=(0.05) # 0.1 0.15 0.2 0.25)

#GYM="end-effector"
GYM="multi-point" #for both origin and pose
#GYM="multi-point-min-pos-and-origin" #only 2 of unit vectors and origin
#GYM="multi-point-pose" #only the unit vectors
NUMHIDDENS=(1 2 4 8 16)
ACTIVATIONS=("sigmoid" "tanh" "relu")
EPOCHS=45
for act in ${ACTIVATIONS[@]}
do
    for numhid in ${NUMHIDDENS[@]}
    do
        for seed in ${SEEDS[@]}
        do
            for dt in ${DT[@]}
            do
            EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
            echo $EPISODES
            TARGS=".targets/$GYM/$seed/targets.pth"
            echo $TARGS
            python3 main.py --seed $seed --runs 110  --policy_name $ALGORITHM --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act

            done
        done
    done
done 
