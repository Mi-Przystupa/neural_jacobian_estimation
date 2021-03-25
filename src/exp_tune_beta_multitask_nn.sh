#!/bin/bash

#SEED=12345

SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)
#ALGORITHM="global-neuralnetwork-multitask-custom" #"global-neuralnetwork-multitask"
ALGORITHM="global-neuralnetwork-nullspace" # add "-custom" for multi-point

DT=(0.05) # 0.1 0.15 0.2 0.25)

#GYM="end-effector"
#REMINDER: CHANGE NETWORK PARAMETERS FOR MULTI-POINT!!!!!!
#GYM="multi-point" #for both origin and pose
GYM="end-effector"
BETAS=(0.05 0.1 .2 .4 .8 1.6 0.0 1.0)
EPOCHS=45

#numhid=4
#act="relu"

#for end-effector...more of just in case thing than actually being necessary
numhid=2
act="relu"


for beta in ${BETAS[@]}
do
    for seed in ${SEEDS[@]}
    do
        for dt in ${DT[@]}
        do
        EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
        echo $EPISODES
        TARGS=".targets/$GYM/$seed/targets.pth"
        echo $TARGS
        #python3 main.py --seed $seed --runs 110  --policy_name $ALGORITHM --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --beta $beta
        python3 main.py --seed $seed --runs 110  --policy_name $ALGORITHM --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act --beta $beta


        done
    done
done
