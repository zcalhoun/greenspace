#!/bin/bash

#SBATCH --job-name=pm_fit
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=8G
#SBATCH --output=./pm_fit/%a.txt
#SBATCH --error=./pm_fit/%a.err

source ~/.bashrc
conda activate svgp

python create_maps.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/ridge/pm_ip100/ \
    --patience 5 \
    --batch-size 128 \
    --pretrain-lr 0.1 \
    --lr 0.1 \
    --window-size 101 \
    --epochs 20 \
    --pretrain-epochs 2 \
    --num-inducing-points 100