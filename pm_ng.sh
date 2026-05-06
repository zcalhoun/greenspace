#!/bin/bash

#SBATCH --job-name=pm_ng
#SBATCH --array=1,10,11,15,20
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=./pm_ng/%a.txt
#SBATCH --error=./pm_ng/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/pm/ng_2/ \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 128 \
    --window-size 500 \
    --bayes-opt-iters 20 \
    --num-inducing-points 100