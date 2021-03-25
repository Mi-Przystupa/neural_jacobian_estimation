#!/bin/bash
#SBATCH --time=0-16:00
#SBATCH --account=def-jag
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=16
#SBATCH --job-name=multiptknnjac
#SBATCH --output=%x-%j.out


#SEED=12345

#SEEDS=(12345 45 212 458 30 84) # original seeds I used
#SEEDS=(12345)
#ALGORITHM=("inversejacobian" "local-uvs" "broyden" "global-locallinear" "global-neuralnetwork" "global-neuralnetwork-multitask" "rl_uvs")
#ALGORITHM=("broyden" "bfgs")
#ALGORITHM=("blackbox-kinematics" "inversejacobian" "global-linear" "global-linear-multitask"  "global-locallinear" "global-neuralnetwork" "global-neuralnetwork-multitask")
#ALGORITHM=("blackbox-kinematics" "blackbox-rbf"  "inversejacobian" "global-neuralnetwork" "global-neuralnetwork-multitask" "blackbox-el" "global-locallinear")
#ALGORITHM=("blackbox-kinematics" "blackbox-rbf"  "multipoint-inversejacobian" "global-neuralnetwork" "global-neuralnetwork-multitask" "blackbox-el" "global-locallinear" "broyden" "local-uvs")
#ALGORITHM=("multipoint-inversejacobian")
#ALGORITHM=("broyden" "local-uvs")
#ALGORITHM=("global-locallinear-kd" "global-locallinear")


SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)

#ALGORITHM=("multitask-knn-neuraljacobian")
#ALGORITHM=("knn-neuraljacobian")
#ALGORITHM=("local-uvs" "broyden" "inversejacobian" "global-neuralnetwork" "global-neuralnetwork-multitask" "blackbox-kinematics" "global-locallinear-kd")
#ALGORITHM=("global-reverse-neuralnetwork")
#ALGORITHM=("global-reverse-neuralnetwork" "global-reverse-neuralnetwork-custom" "global-reverse-neuralnetwork-multitask-custom" "global-reverse-neuralnetwork-multitask")


#multi-point
#ALGORITHM=("global-neuralnetwork-custom" "global-neuralnetwork-multitask-custom" "blackbox-kinematics-custom" "global-locallinear-kd" "local-uvs" "broyden" "multipoint-inversejacobian")
#ALGORITHM=("knn-neuraljacobian-custom")
#ALGORITHM=("multitask-knn-neuraljacobian-custom")
ALGORITHM=("knn-neuraljacobian")
#ALGORITHM=("multitask-knn-neuraljacobian")




#ALGORITHM=("blackbox-rbf"  "blackbox-el")

#ALGORITHM=("multipoint-inversejacobian")
DT=(0.05) # 0.1 0.15 0.2 0.25)
#GYM="multi-point" #for both origin and pose
#numhid=4

GYM="end-effector"
numhid=2
act="relu"

EPOCHS=30 # single point, 100,000 data points
#EPOCHS=40 # multi point, 200,000 data points

source ~/projects/def-jag/przy/visual_servo/bin/activate

for value in ${ALGORITHM[@]}
do
    for seed in ${SEEDS[@]}
    do
        for dt in ${DT[@]}
        do
            #python3 main.py --seed $seed --runs 110  --policy_name $value --dt $dt --environment $GYM
            
            EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
            echo $EPISODES
            TARGS=".targets/$GYM/$seed/targets.pth"
            echo $TARGS
            #python3 main.py --seed $seed --runs 110 --policy_name $value --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act 
            python3 main.py --seed $seed --runs 110 --policy_name $value --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act --partial_state "raw_angles" --k 10



        done
    done
done 
