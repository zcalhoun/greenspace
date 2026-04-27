#!/bin/bash

#SBATCH --job-name=nn_am_fit
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=8G
#SBATCH --output=./nn_am_fit/%a.txt
#SBATCH --error=./nn_am_fit/%a.err

source ~/.bashrc
conda activate svgp

python create_maps.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/ridge/am_ip100_nn/ \
    --time am \
    --patience 5 \
    --batch-size 64 \
    --pretrain-lr 0.1 \
    --lr 0.1 \
    --window-size 201 \
    --epochs 20 \
    --pretrain-epochs 10 \
    --num-inducing-points 100 \
    --non-negative