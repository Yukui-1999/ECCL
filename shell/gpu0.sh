#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python main_pretrain.py --batch_size 64 --wandb True --wandb_id 2005 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.1 --cmr_drop_out 0.1

export CUDA_VISIBLE_DEVICES=0
python main_pretrain.py --batch_size 64 --wandb True --wandb_id 2006 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.05 --cmr_drop_out 0.05

export CUDA_VISIBLE_DEVICES=0
python main_pretrain.py --batch_size 64 --wandb True --wandb_id 2007 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.75 --cmr_drop_out 0.75 --weight_decay 0.2

export CUDA_VISIBLE_DEVICES=0
python main_pretrain.py --batch_size 64 --wandb True --wandb_id 2008 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.1 --cmr_drop_out 0.1 --weight_decay 0.1