#export CUDA_VISIBLE_DEVICES=2
#python cmr_pretrain.py --batch_size 128 --wandb True --wandb_id 1015 --device cuda:0 --blr 1e-6 --cmr_model swin --downstream regression

#export CUDA_VISIBLE_DEVICES=0
#python cmr_pretrain.py --batch_size 128 --wandb True --wandb_id 2006 --device cuda:0 --blr 1e-6 --cmr_model swin --downstream classification --classification_dis I21 --seed 43
#
#export CUDA_VISIBLE_DEVICES=0
#python cmr_pretrain.py --batch_size 128 --wandb True --wandb_id 2007 --device cuda:0 --blr 1e-6 --cmr_model swin --downstream classification --classification_dis I42 --seed 43

#export CUDA_VISIBLE_DEVICES=0
#python main_pretrain.py --batch_size 32 --wandb True --wandb_id c2_lr1e6 --loss clip_loss --device cuda:0 --latent_dim 256 --blr 1e-6 --loss_type ecg_cmr --ecg_drop_out 0.1 --cmr_drop_out 0.1 --warmup_epochs 40 --lamda 1 --downstream classification --classification_dis I42 --cmr_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2003/checkpoint-17-auc-0.71.pth --seed 42
