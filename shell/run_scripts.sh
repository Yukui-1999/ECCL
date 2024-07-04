#!/bin/bash
python main_pretrain.py --batch_size 64 --num_workers 8 --wandb True --wandb_id 1013 --loss clip_loss --device cuda:0 --latent_dim 1024 --blr 1e-4 --loss_type ecg_cmr --use_snp False
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --batch_size 64 --wandb False --wandb_id 1031 --loss clip_loss --device cuda:0 --latent_dim 1024 --blr 1e-3 --loss_type ecg_cmr --use_snp False

python main_pretrain.py --batch_size 256 --num_workers 8 --wandb True --wandb_id 1001 --loss clip_loss --device cuda:0 --latent_dim 1024 --blr 1e-4 > /mnt/data/dingzhengyao/work/checkpoint/preject_version1/log_dir/1001/output.txt &
python main_pretrain.py --batch_size 256 --num_workers 8 --wandb True --wandb_id 1002 --loss triplet --device cuda:1 --latent_dim 1024 --blr 1e-4 > /mnt/data/dingzhengyao/work/checkpoint/preject_version1/log_dir/1002/output.txt &
python main_pretrain.py --batch_size 256 --num_workers 8 --wandb True --wandb_id 1003 --loss clip_loss --device cuda:2 --latent_dim 512 --blr 1e-5 > /mnt/data/dingzhengyao/work/checkpoint/preject_version1/log_dir/1003/output.txt &
python main_pretrain.py --batch_size 256 --num_workers 8 --wandb True --wandb_id 1004 --loss triplet --device cuda:3 --latent_dim 512 --blr 1e-3 > /mnt/data/dingzhengyao/work/checkpoint/preject_version1/log_dir/1004/output.txt &

