#!/bin/bash 
#SBATCH --job-name=totahv                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH -t 5-0                  ### time limit 

#rsync -arz ./waymo_dataset /tmp/

## run job    
./train.sh 
