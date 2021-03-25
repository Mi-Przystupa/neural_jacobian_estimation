#!/bin/bash
#SBATCH --time=0-16:00
#SBATCH --account=def-jag
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=16
#SBATCH --array=0-8
#SBATCH --job-name=knn_local_linear
#SBATCH --output=%x-%j.out


SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)

#ALGORITHM=("knn-neuraljacobian-custom")
#ALGORITHM=("multitask-knn-neuraljacobian-custom")
#ALGORITHM=("knn-neuraljacobian")
#ALGORITHM=("multitask-knn-neuraljacobian")

ALGORITHM=("global-locallinear-kd")


DT=(0.05) # 0.1 0.15 0.2 0.25)
GYM="multi-point" #for both origin and pose
#GYM="end-effector"
numhid=4
#numhid=2 # just in case...is for end-effector
act="relu"

EPOCHS=50 #want to give sufficient epochs to converge despite initialization

NEIGHBORS=(1 2 4 8 16 32 50 64 128)
#NEIGHBORS=(1 2 4 8 16 10)
#NEIGHBORS=(16)



NEIGHBOR=${NEIGHBORS[$SLURM_ARRAY_TASK_ID]}
echo $SLURM_ARRAY_TASK_ID
echo $NEIGHBOR

source ~/projects/def-jag/przy/visual_servo/bin/activate

for value in ${ALGORITHM[@]}
do
    echo $value
    for seed in ${SEEDS[@]}
    do
        for dt in ${DT[@]}
        do
            
            EPISODES=".data-control/$GYM/$seed/$seed""-data.pth"
            echo $EPISODES
            TARGS=".targets/$GYM/$seed/targets.pth"
            echo $TARGS
           python3 main.py --seed $seed --runs 110 --policy_name $value --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act --partial_state "raw_angles" --k $NEIGHBOR

        done
    done
done 
