#!/bin/bash

#SBATCH --job-name=nn_save_pm
#SBATCH --array=1-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./nn_save_pm/%a.txt
#SBATCH --error=./nn_save_pm/%a.err

source ~/.bashrc
conda activate svgp

python save_maps.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/ridge/pm_ip100_nn/