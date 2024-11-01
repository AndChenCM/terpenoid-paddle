#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

### start pretrain
# compound_encoder_config="model_configs/geognn_l8.json"
model_config="configs/pretrain_gem.json"
dataset="visnet-pretrain-on-train-bar-80-wH-allmask"
data_path="data/train_smiles"
python pretrain_geometry.py \
		--batch_size=32 \
		--num_workers=40 \
		--max_epoch=50 \
		--lr=1e-3 \
		--dropout_rate=0.2 \
		--data_path=$data_path \
		--model_config=$model_config \
		--model_dir=./pretrain_models/$dataset
