#!/bin/bash

#SBATCH --job-name=af_ng
#SBATCH --array=1,10,11,15,20
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=./af_ng/%a.txt
#SBATCH --error=./af_ng/%a.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/af/ng/ \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 128 \
    --window-size 500 \
    --bayes-opt-iters 50 \
    --num-inducing-points 100 \
    --time af