#!/bin/bash

#SBATCH --job-name=tgash
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./ash_g_pm.txt
#SBATCH --error=./ash_g_pm.err

source ~/.bashrc
conda activate svgp

python main.py \
    --data-dir /work/zdc6/greenspace/data/w51/asheville/pm/ \
    --output-dir /work/zdc6/greenspace/results/w51/pm/g/asheville/ \
    --patience 3 \
    --greenspace
    