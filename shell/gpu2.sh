#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
python main_pretrain.py --batch_size 128 --wandb True --wandb_id 4005 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 5e-5 --loss_type ecg_cmr --ecg_drop_out 0.2 --cmr_drop_out 0.2 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt

export CUDA_VISIBLE_DEVICES=2
python main_pretrain.py --batch_size 128 --wandb True --wandb_id 4006 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 5e-6 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt

export CUDA_VISIBLE_DEVICES=2
python main_pretrain.py --batch_size 128 --wandb True --wandb_id 4007 --loss clip_loss --device cuda:0 --latent_dim 512 --blr 5e-5 --loss_type ecg_cmr --ecg_drop_out 0.2 --cmr_drop_out 0.2 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt

export CUDA_VISIBLE_DEVICES=2
python main_pretrain.py --batch_size 128 --wandb True --wandb_id 4008 --loss clip_loss --device cuda:0 --latent_dim 512 --blr 5e-6 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt