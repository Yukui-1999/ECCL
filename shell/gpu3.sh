export CUDA_VISIBLE_DEVICES=3
python main_finetune.py --batch_size 32 --wandb True --wandb_id finetune_f7_1 --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/output_dir/f7_lamda5_ori_cor/checkpoint-43-correlation-0.39.pth

##!/bin/bash
#
#export CUDA_VISIBLE_DEVICES=3
#python main_pretrain.py --batch_size 128 --wandb True --wandb_id 5005 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --temperature 0.2 --ecg_drop_out 0.2 --cmr_drop_out 0.2 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt
#
#export CUDA_VISIBLE_DEVICES=3
#python main_pretrain.py --batch_size 128 --wandb True --wandb_id 5006 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --temperature 0.05 --ecg_drop_out 0.1 --cmr_drop_out 0.1 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt
#
#export CUDA_VISIBLE_DEVICES=3
#python main_pretrain.py --batch_size 128 --wandb True --wandb_id 5007 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --cmr_use_seg False --cmr_inchannels 2 --ecg_drop_out 0.2 --cmr_drop_out 0.2 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt
#
#export CUDA_VISIBLE_DEVICES=3
#python main_pretrain.py --batch_size 128 --wandb True --wandb_id 5008 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.05 --cmr_drop_out 0.05 --data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt --val_data_path /mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt