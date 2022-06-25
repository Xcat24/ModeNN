#!/bin/bash
export dataset='MNIST'
export data_dir='/disk/Dataset/MNIST'
export model_name='2MODENN'
export log_dir='/disk/Log/'
# export CUDA_VISIBLE_DEVICES=0

python main.py \
    --dataset ${dataset} \
    --data-dir ${data_dir} \
    --seed 24211 \
    --num-class 10 \
    --input-size 1 28 28 \
    --net ModeNN \
    --order 2 \
    --lr 0.1 \
    --max-epochs 300 \
    -b 50 \
    --bar 1 \
    --run-name ${model_name} \
    --model-name ${model_name} \
    --is-checkpoint \
    --saved-path ${log_dir} \
    --gpus 1 

