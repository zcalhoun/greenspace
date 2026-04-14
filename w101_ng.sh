#!/bin/bash

#SBATCH --job-name=pm_ng_w101
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=./pm_ng_w101/%a.txt
#SBATCH --error=./pm_ng_w101/%a.err

source ~/.bashrc
conda activate svgp

which python
python --version
ldd $(which python) | grep libstdc++

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/w101/pm/ng/ \
    --patience 3 \
    --batch-size 128 \
    --pretrain-lr 0.1 \
    --lr 0.05 \
    --window-size 101 \
    --epochs 10