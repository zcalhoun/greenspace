#!/bin/bash

python main.py \
    --data-dir ../../data/traversals/ \
    --output-dir ./test_results/ \
    --epochs 10 \
    --test \
    --lr 0.01 \
    --batch-size 128 \
    --window-size 100 \
    --bayes-opt-iters 10 \
    --greenspace \
    --gs-downsample 10