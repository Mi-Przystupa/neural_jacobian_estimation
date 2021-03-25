#!/bin/bash
#SBATCH --time=0-30:00
#SBATCH --account=def-jag
#SBATCH --mem=96000M
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --job-name=big_knn_tuning
#SBATCH --output=knn-tuning-%x-%j.out


#originally used array=0-9 
SEEDS=(12345 45 212 458 30 84 11893 27948 8459 984)
#SEEDS=(458 30)


#ALGORITHM=("knn-neuraljacobian-custom")
#ALGORITHM=("multitask-knn-neuraljacobian-custom")

ALGORITHM=("knn-neuraljacobian")
#ALGORITHM=("multitask-knn-neuraljacobian-custom")


DT=(0.05) # 0.1 0.15 0.2 0.25)
#GYM="multi-point" #for both origin and pose
GYM="end-effector"
#numhid=4
numhid=2 #just incase for endeffector
act="relu"

EPOCHS=50 #want to give sufficient epochs to converge despite initialization


NEIGHBOR=32

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Job start"
echo $SLURM_ARRAY_TASK_ID
echo $NEIGHBOR
echo $SEED

source ~/projects/def-jag/przy/visual_servo/bin/activate

echo "run experiment"
for value in ${ALGORITHM[@]}
do
    for dt in ${DT[@]}
    do
        
        EPISODES=".data-control/$GYM/$SEED/$SEED""-data.pth"
        echo $EPISODES
        TARGS=".targets/$GYM/$SEED/targets.pth"
        echo $TARGS
        python3 main.py --seed $SEED --runs 110 --policy_name $value --dt $dt --environment $GYM --load_episodes $EPISODES --targets_pth $TARGS --eval_horizon 200 --epochs $EPOCHS --num_hiddens $numhid --activation $act --partial_state "raw_angles" --k $NEIGHBOR


    done
done
