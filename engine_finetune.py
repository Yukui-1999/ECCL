import math
import sys
from typing import Iterable
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import wandb
import nibabel
import utils.misc as misc
import utils.lr_sched as lr_sched

import matplotlib.pyplot as plt




def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    training_history = {}
    optimizer.zero_grad()
    loss_fn = torch.nn.MSELoss()

    for data_iter_step, (ecg, cmr, tar,snp,cha,I21,I42,I48,I50) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        ecg = ecg.unsqueeze(1).float().to(device)
        cha = cha.float().to(device)
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        # print(cha.shape)
        #
        # for i in range(cmr.shape[0]):
        #     nibabel.save(nibabel.Nifti1Image(cmr[i].cpu().numpy(), None), f'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/test/{data_iter_step}_cmr_{i}.nii.gz')
        #

        with torch.cuda.amp.autocast():
            _,output = model(ecg)
            # print(output.shape)
            loss = loss_fn(output, cha)
        loss_value = loss.item()
        loss_name = args.downstream
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    print("stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history[f'train_{loss_name}_loss'] = train_stats["loss"]
        training_history['lr'] = train_stats["lr"]
    
    return train_stats, training_history

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    test_history = {}  
    model.eval()
    loss_fn = torch.nn.MSELoss()
    output = []
    label = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        I21 = batch[5]
        I42 = batch[6]
        I48 = batch[7]
        I50 = batch[8]
        ecg = batch[0].unsqueeze(1).float().to(device)
        cha = batch[4].float().to(device)
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()

        with torch.cuda.amp.autocast():
            _,out = model(ecg)
            loss = loss_fn(out, cha)
        loss_value = loss.item()
        loss_name = args.downstream

        label.append(cha.cpu().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)
        metric_logger.update(loss=loss_value)


    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    if args.downstream == 'classification':
        auc = roc_auc_score(label, output)
        metric_logger.update(auc=auc)

    if args.downstream == 'regression':
        corr_list = []
        for i in range(82):
            corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            corr_list.append(corr)
        metric_logger.update(correlation=np.mean(corr_list))

    print("validation stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
            test_history['epoch'] = epoch
            test_history[f'val_{loss_name}_loss'] = test_stats["loss"]
            if args.downstream == 'classification':
                test_history['auc'] = test_stats['auc']
            if args.downstream == 'regression':
                test_history['correlation'] = np.mean(corr_list)
    return test_stats, test_history