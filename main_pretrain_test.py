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
    parser.add_argument('--latent_dim', default=2048, type=int, metavar='N',
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
                        default="/mnt/data/dingzhengyao/work/checkpoint/ECG_CMR/outputdir/1012/checkpoint-294-ncc-0.98.pth",
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

    # load data
    dataset_train = mutimodal_dataset(data_path=args.data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    data_scaler = dataset_train.get_scaler()
    dataset_val = mutimodal_dataset(data_path=args.val_data_path, transform=True, augment=False, args=args,scaler=data_scaler,downstream=args.downstream)
    dataset_test = mutimodal_dataset(data_path=args.test_data_path, transform=True, augment=False, args=args,scaler=data_scaler,downstream=args.downstream)
    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    print("Test set size: ", len(dataset_test))

    if args.cmr_model.startswith('swin'):
        args.resizeshape = 224
    else:
        args.resizeshape = 80

    model = ECGEncoder.__dict__[args.ecg_model](
        img_size=args.ecg_input_size,
        patch_size=args.ecg_patch_size,
        in_chans=args.ecg_input_channels,
        num_classes=args.regression_dim,
        drop_rate=args.ecg_drop_out,
        args=args,
    )

    print("load pretrained ecg_model")
    ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
    ecg_checkpoint_model = ecg_checkpoint['model']

    # regression_linear_keys = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('regression_linear')}
    # new_keys = {k.replace('regression_linear', 'head'): v for k, v in regression_linear_keys.items()}

    ECG_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ECG_encoder') or k.startswith('regression_linear')}
    ECG_encoder_keys = {k: v for k, v in ECG_encoder_keys.items() if not k.startswith('ECG_encoder.head')}
    ECG_encoder_keys = {k.replace('regression_linear', 'ECG_encoder.head'): v for k, v in ECG_encoder_keys.items()}

    new_keys = {k.replace('ECG_encoder.', ''): v for k, v in ECG_encoder_keys.items()}

    msg = model.load_state_dict(new_keys, strict=False)
    print(msg)
    model.to(device)


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
    model.eval()

    loss_fn = torch.nn.MSELoss()
    loss = []
    output = []
    label = []
    for i, batch in enumerate(data_loader_val):
        ecg, cmr, tar, snp, cha,I21,I42,I48,I50 = batch
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        ecg = ecg.float().unsqueeze(1).to(args.device)
        cha = cha.float().to(args.device)
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(args.device)
            loss_fn = torch.nn.BCEWithLogitsLoss()

        with torch.cuda.amp.autocast():
        # print(ecg.shape)
            _ , out = model(ecg)
            loss_value = loss_fn(out, cha)
        # print(out.shape)

        loss.append(loss_value.item())
        label.append(cha.cpu().detach().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)

    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    print(output.shape)
    print(label.shape)
    print(f"val loss:{np.mean(loss)}")
    if args.downstream == 'classification':
        auc = roc_auc_score(label, output)
        print(f"val auc:{auc}")
    if args.downstream == 'regression':
        output = data_scaler.inverse_transform(output)
        label = data_scaler.inverse_transform(label)
        import matplotlib.pyplot as plt

        # 创建一个画布
        fig = plt.figure(figsize=(25, 17 * 5))
        corr_list = []
        for i in range(82):
            # 创建一个子图
            ax = fig.add_subplot(17, 5, i + 1)
            # 计算相关系数
            corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            corr_list.append(corr)

            label_flat = label[:, i].flatten()
            output_flat = output[:, i].flatten()
            # 绘制散点图
            ax.scatter(label_flat, output_flat)

            label_flat = label_flat.astype(np.float32)
            output_flat = output_flat.astype(np.float32)
            # Fit a line to the data
            fit = np.polyfit(label_flat, output_flat, 1)

            # Create a sequence of x values spanning the range of the data
            x = np.linspace(min(label_flat), max(label_flat), 100)

            # Use the polynomial fit to calculate the corresponding y values
            y = np.polyval(fit, x)

            # Plot the fit line
            ax.plot(x, y, color='black')

            # 在图上添加相关系数的文本
            ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes)
            # 设置子图的标题
            ax.set_title(cor_index[i])
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predict')

        # 显示图形
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(args.val_savepath,"regression.png"))
        data2 = pd.DataFrame(data=corr_list, index=cor_index, columns=['Pearson correlation coefficient'])
        # PATH为导出文件的路径和文件名
        print(f"corr_mean:{np.mean(corr_list)}")
        data2.to_csv(os.path.join(args.val_savepath,"regression.csv"))
    print('val done')
    # plt.show()

    output = []
    label = []
    loss = []
    for i, batch in enumerate(data_loader_test):
        ecg, cmr, tar, snp, cha,I21,I42,I48,I50 = batch
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        ecg = ecg.float().unsqueeze(1).to(args.device)
        cha = cha.float().to(args.device)
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(args.device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        with torch.cuda.amp.autocast():
            # print(ecg.shape)
            _, out = model(ecg)
            loss_value = loss_fn(out, cha)

        # print(out.shape)
        loss.append(loss_value.item())
        label.append(cha.cpu().detach().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)

    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    print(output.shape)
    print(label.shape)
    print(f"test loss:{np.mean(loss)}")
    if args.downstream == 'classification':
        auc = roc_auc_score(label, output)
        print(f"test auc:{auc}")
    if args.downstream == 'regression':
        output = data_scaler.inverse_transform(output)
        label = data_scaler.inverse_transform(label)
        import matplotlib.pyplot as plt

        # 创建一个画布
        fig = plt.figure(figsize=(25, 17 * 5))
        corr_list = []
        for i in range(82):
            # 创建一个子图
            ax = fig.add_subplot(17, 5, i + 1)
            # 计算相关系数
            corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            corr_list.append(corr)

            label_flat = label[:, i].flatten()
            output_flat = output[:, i].flatten()
            # 绘制散点图
            ax.scatter(label_flat, output_flat)

            label_flat = label_flat.astype(np.float32)
            output_flat = output_flat.astype(np.float32)
            # Fit a line to the data
            fit = np.polyfit(label_flat, output_flat, 1)

            # Create a sequence of x values spanning the range of the data
            x = np.linspace(min(label_flat), max(label_flat), 100)

            # Use the polynomial fit to calculate the corresponding y values
            y = np.polyval(fit, x)

            # Plot the fit line
            ax.plot(x, y, color='black')

            # 在图上添加相关系数的文本
            ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes)
            # 设置子图的标题
            ax.set_title(cor_index[i])
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predict')

        # 显示图形
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(args.test_savepath, "regression.png"))
        data2 = pd.DataFrame(data=corr_list, index=cor_index, columns=['Pearson correlation coefficient'])
        # PATH为导出文件的路径和文件名
        print(f"corr_mean:{np.mean(corr_list)}")
        data2.to_csv(os.path.join(args.test_savepath, "regression.csv"))
        # plt.show()
    print('test done')


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
