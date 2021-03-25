#!/bin/bash
#SBATCH --time=0-16:00
#SBATCH --account=def-jag
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=16
#SBATCH --array=4
#SBATCH --job-name=tune_l2_neural_kinematics
#SBATCH --output=%x-%j.out


SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)

#ALGORITHM=("blackbox-kinematics")
ALGORITHM=("blackbox-kinematics-custom")


DT=(0.05) # 0.1 0.15 0.2 0.25)
GYM="multi-point" #for both origin and pose
#GYM="end-effector"
numhid=4
act="tanh"

#EPOCHS=30 #same as original end-effector 100,000 data point experiment
EPOCHS=40 #same as original multi-point 200,000 data point experiment

WEIGHTDECAY=(1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0)

L2=${WEIGHTDECAY[$SLURM_ARRAY_TASK_ID]}
echo $SLURM_ARRAY_TASK_ID
echo $L2

source ~/projects/def-jag/przy/visual_servo/bin/activate

for value in ${ALGORITHM[@]}
do
    for seed in ${SEEDS[@]}
    do
        for dt in ${DT[@]}
        do
            
            EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
            echo $EPISODES
            TARGS=".targets/$GYM/$seed/targets.pth"
            echo $TARGS
           python3 main.py --seed $seed --runs 110 --policy_name $value --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act --l2 $L2


        done
    done
done 
