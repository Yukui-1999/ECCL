import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path
from model.NC_method import NC_method
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

from engine_NC import train_one_epoch, evaluate


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
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    parser.add_argument('--downstream', default='regression', type=str)





    # Model parameters
    parser.add_argument('--latent_dim', default=256, type=int, metavar='N',
                        help='latent_dim')
    parser.add_argument('--resizeshape',default=80,type=int,help='resize shape')


    



    # LOSS parameters

    parser.add_argument('--temperature', default=0.1, type=float, metavar='TEMPERATURE',
                        help='temperature for nt_xent loss')
    parser.add_argument('--alpha_weight', default=0.5, type=float, metavar='ALPHA_WEIGHT',
                        help='alpha_weight for nt_xent loss')

    parser.add_argument('--lamda', default=1, type=float, help='lamda')
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

    parser.add_argument('--output_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/baseline_log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', type=str2bool,  default=True)
    parser.add_argument('--wandb_project', default='CMR_ECG_TAR',
                        help='project where to wandb log')
    # parser.add_argument('--wandb_dir', default='/mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/wandb/1002',
    #                     help='project where to wandb save')
    parser.add_argument('--wandb_id', default='f2', type=str,
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


    model = NC_method( args=args)
    model.to(device)
    # print(model)

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

    print(f'model device:{next(model.parameters()).device}')


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

    args.log_dir = os.path.join(args.log_dir, args.wandb_id)
    args.output_dir = os.path.join(args.output_dir, args.wandb_id)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)