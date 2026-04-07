#!/bin/bash

#SBATCH --job-name=w101gs
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./w101gs/%a.txt
#SBATCH --error=./w101gs/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/w101/pm/gs/ \
    --patience 3 \
    --greenspace \
    --batch-size 1024 \
    --pretrain-lr 0.1 \
    --window-size 101