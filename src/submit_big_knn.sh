#!/bin/bash

name="big-knn"
echo "hello world"
#sbatch exp_big_knn_search.sh 32
#sbatch exp_big_knn_search.sh 64
sbatch exp_big_knn_search.sh 128
