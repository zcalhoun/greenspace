#!/bin/bash

python causal_estimates.py \
    --data-dir ../../data/traversals/ \
    --epochs 50 \
    --test \
    --lr 0.01 \
    --batch-size 128 \
    --gs-downsample 50 \
    --l2-penalty 0.01 \
    --lengthscale 100 \
    --spatial-mask-km 10.0 \