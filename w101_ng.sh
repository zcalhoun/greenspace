#!/bin/bash

#SBATCH --job-name=w101ng
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./w101ng/%a.txt
#SBATCH --error=./w101ng/%a.err

source ~/.bashrc
conda activate svgp

which python
python --version
ldd $(which python) | grep libstdc++

python main.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/w101/pm/ng/ \
    --patience 3 \
    --batch-size 1024 \
    --pretrain-lr 0.1 \
    --window-size 101