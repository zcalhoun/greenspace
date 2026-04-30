#!/bin/bash

#SBATCH --job-name=nn_save_af
#SBATCH --array=0-21
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=./nn_save_af/%a.txt
#SBATCH --error=./nn_save_af/%a.err

source ~/.bashrc
conda activate svgp

python save_maps.py \
    --data-dir /hpc/group/carlsonlab/zdc6/greenspace/data/traversals/ \
    --output-dir /work/zdc6/greenspace/results/ridge/af_ip100_nn/ \
    --time af