#!/bin/bash

#SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)
SEEDS=(984)


#ALGORITHM=("blackbox-kinematics")  #("2-dof-inversejacobian" "local-uvs" "broyden" "global-locallinear" "global-neuralnetwork" "global-neuralnetwork-multitask" "blackbox-kinematics")
#ALGORITHM=("blackbox-kinematics-custom" "global-neuralnetwork-custom" "global-neuralnetwork-multitask-custom" "2-dof-inversejacobian")
#ALGORITHM=("global-neuralnetwork-multitask-custom" "2-dof-inversejacobian")

#ALGORITHM=("knn-neuraljacobian-custom" "multitask-knn-neuraljacobian-custom")

#ALGORITHMS FOR UVS
#ALGORITHM=("blackbox-kinematics-custom" "knn-neuraljacobian-custom" "multitask-knn-neuraljacobian-custom" "global-locallinear-kd" "broyden" "local-uvs")
#ALGORITHM=("blackbox-kinematics-custom")
#ALGORITHM=("global-locallinear-kd")

#ALGORITHMS for 7 DOF
#ALGORITHM=("blackbox-kinematics-custom" "knn-neuraljacobian-custom" "multitask-knn-neuraljacobian-custom" "global-locallinear-kd")
ALGORITHM=("global-locallinear-kd")




#ALGORITHM=("2-dof-inversejacobian")

DT=0.05
#GYM="kinova-2d-planar" #only the unit vectors
#GYM="kinova-camera-2d-planar"
GYM="7-dof-kinova"
#EPOCHS=35 #number of epochs might be related to using unnormalized pixel coords
EPOCHS=40 # was set-up we use in original end-effector experiments
#numhid=1 # for 2DOF Exp
numhid=2 # for 7 dof exp
#act="relu"

EXCEPTION="blackbox-kinematics-custom"
NEIGHBOR="global-locallinear-kd"
k=10
for value in ${ALGORITHM[@]}
do
    for seed in ${SEEDS[@]}
    do
        #EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
        EPISODES=".data-control/$GYM/12345/12345-data.pth"

        echo $EPISODES
        TARGS=".targets/$GYM/$seed/targets.pth"
        echo $TARGS
        echo "$GYM $seed"
        #LINE BELOW WAS FOR IN ROBOT coordinates
        #rosrun UncalibratedVisualServoingLearning robo_main.py --gain 0.9 --seed $seed --runs 25  --policy_name $value --dt $DT --environment $GYM --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --load_episodes $EPISODES --num_hiddens $numhid --activation $act --k 10 --partial_state "raw_angles"

        #BELOW IS FOR VISUAL SERVOING
        if [ "$EXCEPTION" = "$value" ];
        then
            act="tanh"
        else
            act="relu"
        fi
        echo $act

        #BELOW IS FOR VISUAL SERVOING
        if [ "$NEIGHBOR" = "$value" ];
        then
            k="50"
        else
            k="10"
        fi
        echo $k
        echo $value

        
        rosrun UncalibratedVisualServoingLearning robo_main.py --gain 0.7 --seed $seed --runs 10  --policy_name $value --dt $DT --environment $GYM --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --load_episodes $EPISODES --num_hiddens $numhid --activation $act --partial_state "raw_angles" --k $k



    done
done 
