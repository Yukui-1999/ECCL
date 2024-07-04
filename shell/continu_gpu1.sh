#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0


# export CUDA_VISIBLE_DEVICES=0
# python main_pretrain.py --batch_size 128 --wandb True --wandb_id 7005 --loss clip_loss --device cuda:0 --latent_dim 128 --blr 1e-5 --loss_type ecg_cmr --cmr_use_continue True --cmr_use_seg False --cmr_inchannels 50 --ecg_drop_out 0.1 --cmr_drop_out 0.1

# export CUDA_VISIBLE_DEVICES=0

#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c5 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I21 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2006/checkpoint-17-auc-0.77.pth --seed 43
#
#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c6 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2007/checkpoint-23-auc-0.69.pth --seed 43
#
#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c7 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2008/checkpoint-24-auc-0.77.pth --seed 43
#
#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c8 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I50 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2009/checkpoint-11-auc-0.77.pth --seed 43

#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c9 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I21 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2010/checkpoint-19-auc-0.78.pth --seed 44

#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c10 --loss clip_loss --device cuda:0 --latent_dim 256  --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2011/checkpoint-32-auc-0.73.pth --seed 44
#
#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c11 --loss clip_loss --device cuda:0 --latent_dim 256  --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2012/checkpoint-20-auc-0.76.pth --seed 44
#
#export CUDA_VISIBLE_DEVICES=2
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c12 --loss clip_loss --device cuda:0 --latent_dim 256  --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I50 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2009/checkpoint-11-auc-0.77.pth --seed 43

#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I42ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2003/checkpoint-17-auc-0.71.pth --seed 42 --ecg_pretrained False
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I48ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2004/checkpoint-17-auc-0.76.pth --seed 42 --ecg_pretrained False
#
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c_I50ab1 --loss clip_loss --device cuda:2 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream classification --classification_dis I48 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2005/checkpoint-16-auc-0.79.pth --seed 42 --ecg_pretrained False

# python main_pretrain.py --batch_size 32 --wandb True --wandb_id f_ab3 --loss clip_loss --device cuda:1 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --temperature 0.05
export CUDA_VISIBLE_DEVICES=2
python main_pretrain.py --batch_size 32 --wandb True --wandb_id fab_huge --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-5 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --alpha_weight 0.5 --downstream regression --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth --seed 42 --ecg_model vit_huge_patchX --Vit_embbeding 1280 --ecg_pretrained_model /mnt/data2/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1021/checkpoint-239-ncc-0.98.pth
