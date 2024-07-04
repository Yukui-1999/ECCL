#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1030.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained False --downstream classification --classification_dis I21
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1031.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained False --downstream classification --classification_dis I42
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1032.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained False --downstream classification --classification_dis I48
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1033.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained False --downstream classification --classification_dis I50



#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1034.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I21 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1035.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I42 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1036.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I48 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth
#
#export CUDA_VISIBLE_DEVICES=1
#python main_finetune.py --batch_size 32 --wandb True --wandb_id 1037.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I50 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth


# export CUDA_VISIBLE_DEVICES=2
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1034.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream classification --classification_dis I21 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

# export CUDA_VISIBLE_DEVICES=2
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1035.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream classification --classification_dis I42 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

# export CUDA_VISIBLE_DEVICES=2
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1036.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream classification --classification_dis I48 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

# export CUDA_VISIBLE_DEVICES=2
# python main_finetune.py --batch_size 32 --wandb True --wandb_id 1037.2 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream classification --classification_dis I50 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth

export CUDA_VISIBLE_DEVICES=2
python main_finetune.py --batch_size 32 --wandb True --wandb_id fab_huge --device cuda:0 --blr 1e-6 --seed 42 --ecg_pretrained True --downstream regression --ecg_model vit_huge_patchX --ecg_pretrained_model /mnt/data2/dingzhengyao/work/checkpoint/preject_version1/output_dir/fab_huge/checkpoint-28-correlation-0.39.pth