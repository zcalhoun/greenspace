#!/bin/bash

#SBATCH --job-name=pm_gs_w101
#SBATCH --array=0-5
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=./e20_pm_gs_w101/%a.txt
#SBATCH --error=./e20_pm_gs_w101/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/w101_e20/pm/gs/ \
    --patience 100 \
    --greenspace \
    --batch-size 128 \
    --pretrain-lr 0.1 \
    --lr 0.1 \
    --window-size 101 \
    --epochs 20 \
    --pretrain-epochs 2