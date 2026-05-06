#!/bin/bash

#SBATCH --job-name=pm_gs
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=./pm_gs/%a.txt
#SBATCH --error=./pm_gs/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/pm/gs/ \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 128 \
    --window-size 500 \
    --bayes-opt-iters 40 \
    --greenspace \
    --gs-downsample 10 \
    --num-inducing-points 100