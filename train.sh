#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenmingan/mambaforge/envs/paddle/include/nccl.h
export LD_LIBRARY_PATH=/home/chenmingan/mambaforge/envs/paddle/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/chenmingan/projects/paddle/prop_regr/lib:$LD_LIBRARY_PATH

python test_main_fg.py

