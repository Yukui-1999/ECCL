import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path
import model.ECGEncoder as ECGEncoder
import sys
import torch
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn
import wandb

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model.Trimodal_clip import Trimodal_clip
# sys.path.append("..")
import timm
from data.mutimodal_dataset import mutimodal_dataset
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.callbacks import EarlyStop

from engine_finetune import train_one_epoch, evaluate


# from engine_pretrain import train_one_epoch, evaluate

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
    #downstream task
    parser.add_argument('--downstream', default='regression', type=str, help='downstream task')
    parser.add_argument('--regression_number',default=0,type=int,help='regression_number')
    parser.add_argument('--regression_dim',default=82,type=int,help='regression_dim')
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
    parser.add_argument('--ecg_pretrained', default=False, type=str2bool,help='ecg_pretrained or not')
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
    parser.add_argument('--ecg_globle_pool', default=False,type=str2bool, help='ecg_globle_pool')
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
    parser.add_argument('--tar_pretrained', default=True, type=str2bool,help='tar_pretrained or not')
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

    parser.add_argument('--output_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/finetune_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/finetune_log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', type=str2bool,  default=True)
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
    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # load data
    dataset_train = mutimodal_dataset(data_path=args.data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    data_scaler = dataset_train.get_scaler()
    dataset_val = mutimodal_dataset(data_path=args.val_data_path, transform=True, augment=False, args=args,scaler=data_scaler,downstream=args.downstream)
    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    if args.downstream == 'classification':
        args.regression_dim = 1
        if args.classification_dis == 'I21':
            class_num = 5
        elif args.classification_dis == 'I42':
            class_num = 6
        elif args.classification_dis == 'I48':
            class_num = 7
        elif args.classification_dis == 'I50':
            class_num = 8
        positive_indices = []
        negative_indices = []
        for i in range(len(dataset_train)):

            label = dataset_train[i][class_num]
            if label == 1:
                positive_indices.append(i)
            else:
                negative_indices.append(i)
        # positive_indices = [i for i in range(len(dataset_train)) if dataset_train[i][class_num] == 1]
        # negative_indices = [i for i in range(len(dataset_train)) if dataset_train[i][class_num] == 0]
        print(f'positive_indices:{len(positive_indices)},negative_indices:{len(negative_indices)}')
        # I21 positive_indices:1573,negative_indices:23335
        # 根据1:2的比例计算每个组合需要的阴性样本数量
        neg_samples_per_group = len(positive_indices) * 2
        # 计算可以形成的组合数量
        num_groups = len(negative_indices) // neg_samples_per_group
        negative_splits = [negative_indices[i * neg_samples_per_group: (i + 1) * neg_samples_per_group] for i in
                           range(num_groups)]
        if len(negative_indices) % neg_samples_per_group != 0:
            remaining_negatives = negative_indices[num_groups * neg_samples_per_group:]
            negative_splits[-1].extend(remaining_negatives)

    if args.wandb:
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config)
        else:
            wandb.init(project=args.wandb_project, config=config)
    if args.downstream == 'regression':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    model = ECGEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.regression_dim,
                drop_rate=args.ecg_drop_out,
                args=args,
            )
    if args.ecg_pretrained:
        print("load pretrained ecg_model")
        ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
        ecg_checkpoint_model = ecg_checkpoint['model']
        if 'output_dir' in args.ecg_pretrained_model:
            ECG_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if
                                k.startswith('ECG_encoder') or k.startswith('regression_linear')}
            ECG_encoder_keys = {k: v for k, v in ECG_encoder_keys.items() if not k.startswith('ECG_encoder.head')}
            ECG_encoder_keys = {k.replace('regression_linear', 'ECG_encoder.head'): v for k, v in
                                ECG_encoder_keys.items()}

            ecg_checkpoint_model = {k.replace('ECG_encoder.', ''): v for k, v in ECG_encoder_keys.items()}
            print('load f8 finetune model')
        if 'baseline' in args.ecg_pretrained_model:
            ECG_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ecg_encoder')}
            ecg_checkpoint_model = {k.replace('ecg_encoder.', ''): v for k, v in ECG_encoder_keys.items()}
            print('load baseline NC model')

        msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
        print('load ecg model')
        # ECG_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ECG_encoder')}
        # new_keys = {k.replace('ECG_encoder.', ''): v for k, v in ECG_encoder_keys.items()}
        # filtered_keys = {k: v for k, v in new_keys.items() if not k.startswith('head')}
        # msg = model.load_state_dict(filtered_keys, strict=False)
        print(msg)
    model.to(device)
    print(f'model device:{next(model.parameters()).device}')
    # state_dict = model.state_dict()
    # for name, param in state_dict.items():
    #     print(f'Parameter name: {name}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    eval_criterion = "correlation"
    best_stats = {'correlation': -np.inf}
    if args.downstream == 'classification':
        eval_criterion = "auc"
        best_stats = {'auc': -np.inf}

    # ecg_data = torch.randn(2,1,12,5000).to(device)
    # tar_data = torch.randn(2,195).to(device)
    # cmr_data = torch.randn(2,10,80,80).to(device)
    # total_loss = model(ecg_data,tar_data,cmr_data,is_train=True)

    # print(f'ecg:{ecg.shape},tar:{tar.shape},cmr:{cmr.shape}')
    # print(f'total_loss:{total_loss}')
    # return 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.downstream == 'classification':
            for neg_split in negative_splits:
                # 合并阳性样本索引和当前阴性子集索引进行训练
                combined_indices = positive_indices + neg_split
                balanced_dataset = Subset(dataset_train, combined_indices)
                data_loader_train = torch.utils.data.DataLoader(balanced_dataset,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                num_workers=args.num_workers,
                                                                pin_memory=args.pin_mem,
                                                                drop_last=False,)
                train_stats, train_history = train_one_epoch(
                    model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args=args
                )
            val_stats, test_history = evaluate(data_loader_val, model, device, epoch, args=args)
            print(f"Loss of the network on the {len(dataset_val)} val dataset: {val_stats['loss']:.4f}")

            if eval_criterion == "loss":
                if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                    break
                if args.output_dir and val_stats[eval_criterion] <= best_stats[eval_criterion]:
                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)
            else:
                if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                    break
                if args.output_dir and val_stats[eval_criterion] >= best_stats[eval_criterion]:
                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)

            best_stats['auc'] = max(best_stats['auc'], val_stats['auc'])

            if args.wandb:
                wandb.log(train_history | test_history)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))


        else:

            train_stats, train_history = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args=args
            )

            val_stats, test_history = evaluate(data_loader_val, model, device, epoch, args=args)
            print(f"Loss of the network on the {len(dataset_val)} val dataset: {val_stats['loss']:.4f}")

            if eval_criterion == "loss":
                if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                    break
                if args.output_dir and val_stats[eval_criterion] <= best_stats[eval_criterion]:
                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)
            else:
                if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                    break
                if args.output_dir and val_stats[eval_criterion] >= best_stats[eval_criterion]:
                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)

            best_stats['correlation'] = max(best_stats['correlation'], val_stats['correlation'])

            if args.wandb:
                wandb.log(train_history | test_history)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

    return 0


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
