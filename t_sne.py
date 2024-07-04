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
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import model.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model.Trimodal_clip import Trimodal_clip
# sys.path.append("..")
import timm
import torch.nn as nn
from data.mutimodal_dataset import mutimodal_dataset
import timm.optim.optim_factory as optim_factory
import pytorchvideo.models.resnet
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.callbacks import EarlyStop
from model.swin_transformer import SwinTransformer
import numpy as np
from cmr_pretrain_engine import train_one_epoch, evaluate
import pandas as pd
import model.ECGEncoder as ECGEncoder
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
    # downstream task
    parser.add_argument('--downstream', default='regression', type=str, help='downstream task')
    parser.add_argument('--regression_dim', default=82, type=int, help='regression_dim')
    parser.add_argument('--classification_dis', default='I21', type=str, help='classification_dis')

    # Model parameters
    parser.add_argument('--latent_dim', default=256, type=int, metavar='N',
                        help='latent_dim')
    # SNP parameters
    parser.add_argument('--snp_size', default=(49, 120), type=Tuple, help='ecg input size')
    parser.add_argument('--use_snp', default=False, type=str2bool, help='use_snp')
    parser.add_argument('--snp_drop_out', default=0.0, type=float)
    parser.add_argument('--snp_att_depth', default=12, type=int)
    parser.add_argument('--snp_global_pool', default=False, type=str2bool, help='snp_global_pool')
    # ECG Model parameters
    parser.add_argument('--ecg_pretrained', default=True, type=str2bool, help='ecg_pretrained or not')
    parser.add_argument('--ecg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',
                        default="/mnt/data/dingzhengyao/work/checkpoint/preject_version1/output_dir/f7_lamda5_ori_cor/checkpoint-43-correlation-0.39.pth",
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
                        default="/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/2002/checkpoint-17-auc-0.77.pth",
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
    parser.add_argument('--margin', default=0.025, type=float, metavar='MARGIN', help='margin for triplet loss')
    parser.add_argument('--temperature', default=0.1, type=float, metavar='TEMPERATURE',
                        help='temperature for nt_xent loss')
    parser.add_argument('--alpha_weight', default=0.25, type=float, metavar='ALPHA_WEIGHT',
                        help='alpha_weight for nt_xent loss')
    parser.add_argument('--loss_type', default='ecg_cmr', type=str, help='loss_type')
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

    parser.add_argument('--output_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', type=str2bool, default=True)
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

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

@torch.no_grad()
def main(args):
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    device = torch.device(args.device)
    if args.downstream == 'classification':
        args.regression_dim = 1

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True



    if args.cmr_model.startswith('swin'):
        args.resizeshape = 224
        cmr_encoder = SwinTransformer(img_size=(224, 224),
                                patch_size=(4, 4),
                                in_chans=50,
                                num_classes=args.regression_dim,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.,
                                attn_drop_rate=0.,
                                drop_path_rate=0.2,
                                norm_layer=nn.LayerNorm,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
    else:
        args.resizeshape = 80
        cmr_encoder = CMREncoder.__dict__[args.cmr_model](
            in_chans=args.cmr_inchannels,
            img_size=args.img_size,
            num_classes=args.latent_dim,
            drop_rate=args.cmr_drop_out,
            args=args,
        )

    ecg_model = ECGEncoder.__dict__[args.ecg_model](
        img_size=args.ecg_input_size,
        patch_size=args.ecg_patch_size,
        in_chans=args.ecg_input_channels,
        num_classes=args.regression_dim,
        drop_rate=args.ecg_drop_out,
        args=args,
    )
    # args.regression_dim

    cmr_msg = cmr_encoder.load_state_dict(torch.load("/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/1010/checkpoint-81-loss-0.69.pth", map_location='cpu')['model'], strict=False)
    print(cmr_msg)
    ecg_msg = ecg_model.load_state_dict(torch.load("/mnt/data/dingzhengyao/work/checkpoint/preject_version1/finetune_output_dir/1054.2/checkpoint-39-correlation-0.34.pth", map_location='cpu')['model'], strict=False)
    print(ecg_msg)

    # print("load pretrained ecg_model")
    # checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
    # checkpoint_model = checkpoint['model']
    #
    #
    # CMR_encoder_keys = {k: v for k, v in checkpoint_model.items() if k.startswith('CMR_encoder') }
    # CMR_encoder_keys = {k.replace('CMR_encoder.', ''): v for k, v in CMR_encoder_keys.items()}
    # cmr_msg = cmr_encoder.load_state_dict(CMR_encoder_keys, strict=False)
    # print(cmr_msg)
    #
    # ECG_encoder_keys = {k: v for k, v in checkpoint_model.items() if k.startswith('ECG_encoder') }
    # ECG_encoder_keys = {k.replace('ECG_encoder.', ''): v for k, v in ECG_encoder_keys.items()}
    # ecg_msg = ecg_model.load_state_dict(ECG_encoder_keys, strict=False)
    # print(ecg_msg)

    ecg_model.to(device)
    cmr_encoder.to(device)



    # load data
    dataset_train = mutimodal_dataset(data_path=args.data_path, transform=True, augment=True, args=args,
                                      downstream=args.downstream)
    data_scaler = dataset_train.get_scaler()
    dataset_val = mutimodal_dataset(data_path=args.val_data_path, transform=True, augment=False, args=args,
                                    scaler=data_scaler, downstream=args.downstream)
    dataset_test = mutimodal_dataset(data_path=args.test_data_path, transform=True, augment=False, args=args,
                                     scaler=data_scaler, downstream=args.downstream)
    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    print("Test set size: ", len(dataset_test))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    ecg_model.eval()
    cmr_encoder.eval()
    ecg_features = []
    cmr_features = []
    for i, batch in enumerate(data_loader_val):
        ecg, cmr, tar, snp, cha, I21, I42, I48, I50 = batch
        ecg = ecg.float().unsqueeze(1).to(args.device)
        cmr = cmr.float().to(args.device)
        with torch.cuda.amp.autocast():
            ecg_feat,_ = ecg_model(ecg)
            cmr_feat,_ = cmr_encoder(cmr)
        ecg_features.append(ecg_feat.cpu().numpy())
        cmr_features.append(cmr_feat.cpu().numpy())

    # 将列表转换为NumPy数组
    ecg_features = np.concatenate(ecg_features, axis=0)
    cmr_features = np.concatenate(cmr_features, axis=0)
    print(f'ecg_features.shape: {ecg_features.shape}, cmr_features.shape: {cmr_features.shape}')
    # 整体特征用于T-SNE
    all_features = np.concatenate([ecg_features, cmr_features], axis=0)

    # 使用T-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # 分割ECG和CMR的2D特征
    ecg_features_2d = all_features_2d[:len(ecg_features)]
    cmr_features_2d = all_features_2d[len(ecg_features):]

    # 可视化
    # plt.figure(figsize=(10, 6))
    # plt.scatter(ecg_features_2d[:, 0], ecg_features_2d[:, 1], c='red', label='ECG Features')
    # plt.scatter(cmr_features_2d[:, 0], cmr_features_2d[:, 1], c='blue', label='CMR Features')
    # plt.legend()
    # plt.title('ECG and CMR Features Visualization using T-SNE')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # 创建图表
    plt.figure(figsize=(10, 6))

    # 为了使点更易区分，减小点的大小并设置透明度
    plt.scatter(ecg_features_2d[:, 0], ecg_features_2d[:, 1], c='red', alpha=0.5, label='ECG Features', s=10)
    plt.scatter(cmr_features_2d[:, 0], cmr_features_2d[:, 1], c='blue', alpha=0.5, label='CMR Features', s=10)

    # 图例放在图表外侧
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Distinct ECG and CMR Feature Space Pre-Contrastive Learning
    # 设置标题和轴标签
    plt.title('Distinct ECG and CMR Feature Space Pre-Contrastive Learning', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)

    # 设置轴刻度大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 减少边界宽度
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)

    # 设置背景为白色
    plt.gca().set_facecolor('white')

    # 调整布局以防止标签被截断
    plt.tight_layout()

    # 保存图像
    plt.savefig('/home/dingzhengyao/Work/ECG_CMR/ECG_CMR_TAR/Project_version1/output_cor/3_1_sin_wo.png')






    return 0

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.cmr_model.startswith('swin') or args.cmr_model.startswith('vit_base_patch16'):
        args.resizeshape = 224
        args.img_size = 224
    else:
        args.resizeshape = 80
    args.cmr_patch_num = (args.img_size // args.cmr_patch_width) * (args.img_size // args.cmr_patch_height) + 1

    args.val_savepath = os.path.join(os.path.dirname(args.ecg_pretrained_model),"val")
    args.test_savepath = os.path.join(os.path.dirname(args.ecg_pretrained_model),"test")

    if args.val_savepath:
        Path(args.val_savepath).mkdir(parents=True, exist_ok=True)
    if args.test_savepath:
        Path(args.test_savepath).mkdir(parents=True, exist_ok=True)
    main(args)
