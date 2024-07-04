import torch
import torch.nn as nn
import argparse
import model.ECGEncoder as ECGEncoder
from typing import Tuple

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    # downstream task
    parser.add_argument('--downstream', default='regression', type=str, help='downstream task')
    parser.add_argument('--regression_number', default=0, type=int, help='regression_number')
    parser.add_argument('--regression_dim', default=82, type=int, help='regression_dim')
    parser.add_argument('--classification_dis', default='I21', type=str, help='classification_dis')

    # Model parameters
    parser.add_argument('--latent_dim', default=2048, type=int, metavar='N',
                        help='latent_dim')
    # SNP parameters
    parser.add_argument('--snp_size', default=(49, 120), type=Tuple, help='ecg input size')
    parser.add_argument('--use_snp', default=True, type=str2bool, help='use_snp')
    parser.add_argument('--snp_drop_out', default=0.0, type=float)
    parser.add_argument('--snp_att_depth', default=12, type=int)
    parser.add_argument('--snp_global_pool', default=False, type=str2bool, help='snp_global_pool')
    # ECG Model parameters
    parser.add_argument('--ecg_pretrained', default=False, type=str2bool, help='ecg_pretrained or not')
    parser.add_argument('--ecg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',
                        default="/mnt/data/dingzhengyao/work/checkpoint/preject_version1/output_dir/f6/checkpoint-49-correlation-0.39.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--ecg_input_channels', type=int, default=1, metavar='N',
                        help='ecginput_channels')
    parser.add_argument('--ecg_input_electrodes', type=int, default=12, metavar='N',
                        help='ecg input electrodes')
    parser.add_argument('--ecg_time_steps', type=int, default=5000, metavar='N',
                        help='ecg input length')
    parser.add_argument('--ecg_input_size', default=(12, 5000), type=Tuple,
                        help='ecg input size')
    parser.add_argument('--ecg_patch_height', type=int, default=1, metavar='N',
                        help='ecg patch height')
    parser.add_argument('--ecg_patch_width', type=int, default=100, metavar='N',
                        help='ecg patch width')
    parser.add_argument('--ecg_patch_size', default=(1, 100), type=Tuple,
                        help='ecg patch size')
    parser.add_argument('--ecg_globle_pool', default=False, type=str2bool, help='ecg_globle_pool')
    parser.add_argument('--ecg_drop_out', default=0.0, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # CMR Model parameters
    parser.add_argument('--cmr_model', default='swin', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--cmr_inchannels', default=50, type=int, metavar='N',
                        help='cmr_inchannels')
    parser.add_argument('--cmr_pretrained', default=True, type=str2bool,
                        help='cmr_pretrained or not')
    parser.add_argument('--cmr_pretrained_model',
                        default="/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1007/checkpoint-78-loss-0.83.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--cmr_frooze', default=True, type=str2bool, help='cmr_frooze')
    parser.add_argument('--img_size', default=80, type=int, metavar='N', help='img_size of cmr')
    parser.add_argument('--cmr_patch_height', type=int, default=8, metavar='N',
                        help='cmr patch height')
    parser.add_argument('--cmr_patch_width', type=int, default=8, metavar='N',
                        help='cmr patch width')
    parser.add_argument('--cmr_drop_out', default=0.0, type=float)
    parser.add_argument('--cmr_use_seg', default=False, type=str2bool, help='whether use seg mask')
    parser.add_argument('--cmr_use_continue', default=True, type=str2bool, help='whether use continue data')

    # TAR Model parameters
    parser.add_argument('--tar_pretrained', default=True, type=str2bool, help='tar_pretrained or not')
    parser.add_argument('--tar_number', default=195, type=int, metavar='N',
                        help='Name of model to train')
    parser.add_argument('--tar_pretrained_path',
                        default='/home/dingzhengyao/Work/ECG_CMR/tabnet/pretrain_tabnet_model_by_train_data_1.zip',
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--tar_model', default='tabnet', type=str, metavar='MODEL')
    parser.add_argument('--tar_hidden_features', default=256, type=int, metavar='N')
    parser.add_argument('--tar_drop_out', default=0.0, type=float)

    # LOSS parameters
    parser.add_argument('--loss', default='clip_loss', type=str, metavar='LOSS', help='loss function')

    # Augmentation parameters
    parser.add_argument('--input_size', type=tuple, default=(12, 5000))

    parser.add_argument('--timeFlip', type=float, default=0.33)

    parser.add_argument('--signFlip', type=float, default=0.33)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=10, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0.015, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters

    parser.add_argument('--data_path',
                        default='/home/dingzhengyao/data/ECG_CMR/train_data_dict_v7.pt',
                        type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path',
                        default='/home/dingzhengyao/data/ECG_CMR/val_data_dict_v7.pt',
                        type=str,
                        help='validation dataset path')
    parser.add_argument('--test_data_path',
                        default='/home/dingzhengyao/data/ECG_CMR/test_data_dict_v7.pt',
                        type=str,
                        help='test dataset path')

    parser.add_argument('--output_dir',
                        default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/finetune_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/finetune_log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', type=str2bool, default=True)
    parser.add_argument('--wandb_project', default='CMR_ECG_TAR_finetune',
                        help='project where to wandb log')
    # parser.add_argument('--wandb_dir', default='/mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/wandb/1002',
    #                     help='project where to wandb save')
    parser.add_argument('--wandb_id', default='1002', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # checkpoint = torch.load("/mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth", map_location='cpu')
    checkpoint = torch.load("/mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir/baseline_nc_1/checkpoint-53-loss-1.38.pth", map_location='cpu')
    ecg_checkpoint_model = checkpoint['model']
    print(ecg_checkpoint_model.keys())
    model = ECGEncoder.__dict__[args.ecg_model](
                    img_size=args.ecg_input_size,
                    patch_size=args.ecg_patch_size,
                    in_chans=args.ecg_input_channels,
                    num_classes=args.regression_dim,
                    drop_rate=args.ecg_drop_out,
                    args=args,
                )
    ECG_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ecg_encoder')}
    new_keys = {k.replace('ecg_encoder.', ''): v for k, v in ECG_encoder_keys.items()}
    msg = model.load_state_dict(new_keys, strict=False)
    print('----------------------------------')
    print(msg)
import os
from pathlib import Path
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.cmr_model.startswith('swin') or args.cmr_model.startswith('vit_base_patch16'):
        args.resizeshape = 224
        args.img_size = 224
    else:
        args.resizeshape = 80
    args.ecg_patch_num = (args.ecg_time_steps // args.ecg_patch_width) * (
                args.ecg_input_electrodes // args.ecg_patch_height) + 1
    args.log_dir = os.path.join(args.log_dir, args.wandb_id)
    args.output_dir = os.path.join(args.output_dir, args.wandb_id)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)