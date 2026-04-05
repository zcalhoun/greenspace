#!/bin/bash

#SBATCH --job-name=gs_pp
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./preprocess.txt
#SBATCH --error=./preprocess.err

source ~/.bashrc
conda activate geo

python process_dataset.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/data/ \
    --window-size 201 \