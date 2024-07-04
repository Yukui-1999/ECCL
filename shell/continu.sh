#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0


# export CUDA_VISIBLE_DEVICES=0
# python main_pretrain.py --batch_size 128 --wandb True --wandb_id 7005 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.1 --cmr_drop_out 0.1

# export CUDA_VISIBLE_DEVICES=0


export CUDA_VISIBLE_DEVICES=1
python main_pretrain.py --batch_size 32 --wandb True --wandb_id fab_large --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --ecg_model vit_large_patchX --Vit_embbeding 1024 --ecg_pretrained_model /mnt/data2/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1022/checkpoint-225-ncc-0.98.pth
#
#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f7_lamda1_cor --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 0.2 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42
#
#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f7_lamda2_cor --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 5 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42
#
#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f7_lamda3_cor --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.2 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42
#
#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f7_lamda4_cor --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.8 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42

#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1050_ --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained False --downstream regression

#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f8 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010.2/checkpoint-114-correlation-0.56.pth --seed 44
#
#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f8_0 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010.2/checkpoint-114-correlation-0.56.pth --seed 44

#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f10_test2 --loss triplet --margin 2.5 --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010.2/checkpoint-114-correlation-0.56.pth --seed 44

##export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c20 --loss triplet --margin 2.5 --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I21 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2006/checkpoint-17-auc-0.77.pth --seed 43
#
##export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c21 --loss triplet --margin 2.5 --device cuda:1 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2007/checkpoint-23-auc-0.69.pth --seed 43
#
##export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c22 --loss triplet --margin 2.5 --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2008/checkpoint-24-auc-0.77.pth --seed 43
#
##export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c23 --loss triplet --margin 2.5 --device cuda:3 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I50 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2009/checkpoint-11-auc-0.77.pth --seed 43

#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f_ssl250 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1011/checkpoint-397-ncc-0.97.pth --ecg_patch_width 250
#
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f_ssl500 --loss clip_loss --device cuda:3 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1010/checkpoint-389-ncc-0.98.pth --ecg_patch_width 500

#python main_pretrain.py --batch_size 32 --wandb True --wandb_id f_ab1 --loss clip_loss --device cuda:1 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --ecg_pretrained False
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I21ab1 --loss clip_loss --device cuda:1 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I21 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2002/checkpoint-17-auc-0.77.pth --seed 42 --ecg_pretrained False

#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I42ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2003/checkpoint-17-auc-0.71.pth --seed 42 --ecg_pretrained False
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I48ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2004/checkpoint-17-auc-0.76.pth --seed 42 --ecg_pretrained False
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I50ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2005/checkpoint-16-auc-0.79.pth --seed 42 --ecg_pretrained False

# python main_pretrain.py --batch_size 32 --wandb True --wandb_id f_ab2 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --temperature 0.2

