import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path
import model.CMREncoder as CMREncoder
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

from engine_pretrain import train_one_epoch, evaluate


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
    parser.add_argument('--regression_dim',default=82,type=int,help='regression_dim')
    
    # Model parameters
    parser.add_argument('--latent_dim', default=256, type=int, metavar='N',
                        help='latent_dim')
    
    

    # CMR Model parameters
    parser.add_argument('--cmr_model', default='vit_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--cmr_inchannels', default=50, type=int, metavar='N',
                        help='cmr_inchannels')
    parser.add_argument('--cmr_pretrained', default=False, type=str2bool,
                        help='cmr_pretrained or not')
    parser.add_argument('--img_size', default=80, type=int, metavar='N', help='img_size of cmr')
    parser.add_argument('--cmr_patch_height', type=int, default=8, metavar='N',
                        help='cmr patch height')
    parser.add_argument('--cmr_patch_width', type=int, default=8, metavar='N',
                        help='cmr patch width')
    parser.add_argument('--cmr_drop_out', default=0.0, type=float)
    parser.add_argument('--cmr_use_seg', default=True, type=str2bool, help='whether use seg mask')
    parser.add_argument('--cmr_use_continue', default=False, type=str2bool, help='whether use continue data')
    
    

    
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
    parser.add_argument('--patience', default=20, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0.2, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters

    parser.add_argument('--data_path',
                        default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v6.pt',
                        type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path',
                        default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v6.pt',
                        type=str,
                        help='validation dataset path')
    parser.add_argument('--test_data_path',
                        default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v6.pt',
                        type=str,
                        help='test dataset path')

    parser.add_argument('--output_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', type=str2bool,  default=True)
    parser.add_argument('--wandb_project', default='CMR_pretrain',
                        help='project where to wandb log')
    # parser.add_argument('--wandb_dir', default='/mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/wandb/1002',
    #                     help='project where to wandb save')
    parser.add_argument('--wandb_id', default='1001', type=str,
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

    if args.wandb:
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config)
        else:
            wandb.init(project=args.wandb_project, config=config)

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
    model = CMREncoder.__dict__[args.cmr_model](
                    in_chans=args.cmr_inchannels,
                    img_size=args.img_size,
                    num_classes=args.regression_dim,
                    drop_rate=args.cmr_drop_out,
                    args=args,
                )
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
    eval_criterion = "loss"
    best_stats = {'loss': np.inf}

    # ecg_data = torch.randn(2,1,12,5000).to(device)
    # tar_data = torch.randn(2,195).to(device)
    # cmr_data = torch.randn(2,10,80,80).to(device)
    # total_loss = model(ecg_data,tar_data,cmr_data,is_train=True)

    # print(f'ecg:{ecg.shape},tar:{tar.shape},cmr:{cmr.shape}')
    # print(f'total_loss:{total_loss}')
    # return 0
    for epoch in range(args.start_epoch, args.epochs):

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

        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])

        if args.wandb:
            wandb.log(train_history | test_history)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    return 0


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    args.cmr_patch_num = (args.img_size // args.cmr_patch_width) * (args.img_size // args.cmr_patch_height) + 1
    args.log_dir = os.path.join(args.log_dir, args.wandb_id)
    args.output_dir = os.path.join(args.output_dir, args.wandb_id)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
