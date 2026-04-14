#!/bin/bash

#SBATCH --job-name=af_gs_w101
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=./af_gs_w101/%a.txt
#SBATCH --error=./af_gs_w101/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/w101/af/gs/ \
    --time af \
    --patience 3 \
    --greenspace \
    --batch-size 128 \
    --pretrain-lr 0.1 \
    --lr 0.05 \
    --window-size 101 \
    --epochs 10