#!/bin/bash

#SBATCH --job-name=pm_causal
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=./pm_causal/%a.txt
#SBATCH --error=./pm_causal/%a.err

source ~/.bashrc
conda activate svgp

python causal_estimates.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/pm_causal/ \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 128 \
    --window-size 500 \
    --gs-downsample 10 \
    --num-inducing-points 100 \
    --lengthscale 120 \
    --l2-penalty 0.001 \
    --spatial-mask-km 10.0 \
    --feature-mask-quantile 0.999
