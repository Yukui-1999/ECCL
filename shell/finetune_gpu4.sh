
export CUDA_VISIBLE_DEVICES=2
python main_finetune.py --batch_size 32 --wandb True --wandb_id finetune_f8 --device cuda:0 --blr 1e-6 --seed 44 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/output_dir/f8/checkpoint-40-correlation-0.39.pth


#export CUDA_VISIBLE_DEVICES=2
#python main_finetune.py --batch_size 32 --wandb True --wandb_id Baseline_NC_fine_f0.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream regression --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth
#
#export CUDA_VISIBLE_DEVICES=2
#python main_finetune.py --batch_size 32 --wandb True --wandb_id Baseline_NC_fine_c1.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I21 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth
#
#export CUDA_VISIBLE_DEVICES=2
#python main_finetune.py --batch_size 32 --wandb True --wandb_id Baseline_NC_fine_c2.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I42 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth
#
#export CUDA_VISIBLE_DEVICES=2
#python main_finetune.py --batch_size 32 --wandb True --wandb_id Baseline_NC_fine_c3.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I48 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth
#
#export CUDA_VISIBLE_DEVICES=2
#python main_finetune.py --batch_size 32 --wandb True --wandb_id Baseline_NC_fine_c4.1 --device cuda:0 --blr 1e-6 --seed 43 --ecg_pretrained True --downstream classification --classification_dis I50 --ecg_pretrained_model /mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth
