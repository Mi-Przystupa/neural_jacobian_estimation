#!/bin/bash

#VALUES=(1 2 5 10 50 100)

#For temp
#VALUES=(0.1 0.2 0.3 )
VALUES=(1234 54 293 8568 3061)

for value in ${VALUES[@]}
do
    python play_pen_baseline.py --seed $value --runs 100 --target_generation kinematic  --algorithm ppo1 --reward_type "l2,precision,action-norm" 
done
