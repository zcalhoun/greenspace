#!/bin/bash

#SBATCH --job-name=tg3dur
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./dur3_g_pm.txt
#SBATCH --error=./dur3_g_pm.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/durham/ \
    --output-dir /work/zdc6/greenspace/results/w51/pm/g/durham3/ \
    --patience 3 \
    --greenspace \
    --batch-size 1024 \
    --pretrain-lr 0.1 \
    --lr 0.1