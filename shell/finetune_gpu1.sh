#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1050 --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained False --downstream regression

#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1054 --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1050.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained False --downstream regression
#
#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1054.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth
#
#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1050.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained False --downstream regression
#
#export CUDA_VISIBLE_DEVICES=0
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1054.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

# export CUDA_VISIBLE_DEVICES=1
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1030.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained False --downstream classification --classification_dis I21

# export CUDA_VISIBLE_DEVICES=1
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1031.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained False --downstream classification --classification_dis I42

# export CUDA_VISIBLE_DEVICES=1
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1032.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained False --downstream classification --classification_dis I48

# export CUDA_VISIBLE_DEVICES=1
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1033.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained False --downstream classification --classification_dis I50

export CUDA_VISIBLE_DEVICES=1
python main_finetune.py --batch_size 32 --wandb True --wandb_id fab_large --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained True --downstream regression --ecg_model vit_large_patchX --ecg_pretrained_model /mnt/data2/dingzhengyao/work/checkpoint/preject_version1/output_dir/fab_large/checkpoint-35-correlation-0.39.pth