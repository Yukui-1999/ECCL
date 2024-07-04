import math
import sys
from typing import Iterable

import torch
from sklearn.metrics import roc_auc_score
import wandb
import nibabel
import utils.misc as misc
import utils.lr_sched as lr_sched
import numpy as np
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

        ecg = ecg.float().unsqueeze(1).to(device)
        cmr = cmr.float().to(device)

        with torch.cuda.amp.autocast():
            rec_loss,clip_loss = model(ecg,cmr)
            rec_loss_value = rec_loss.item()
            clip_loss_value = clip_loss.item()
            loss = rec_loss + args.lamda * clip_loss
            loss_value = loss.item()

        if not math.isfinite(rec_loss_value) or not math.isfinite(clip_loss_value):
            print("Loss is {}, stopping training".format(rec_loss_value))
            print("Loss is {}, stopping training".format(clip_loss_value))
            sys.exit(1)


        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        metric_logger.update(rec_loss=rec_loss_value)
        metric_logger.update(clip_loss=clip_loss_value)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    print("stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history['train_rec_loss'] = train_stats["rec_loss"]
        training_history['train_clip_loss'] = train_stats["clip_loss"]
        training_history['train_total_loss'] = train_stats["loss"]
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

        ecg = batch[0].float().unsqueeze(1).to(device)
        cmr = batch[1].float().to(device)


        with torch.cuda.amp.autocast():
            rec_loss, clip_loss = model(ecg,cmr)
            rec_loss_value = rec_loss.item()
            clip_loss_value = clip_loss.item()
            loss = rec_loss + args.lamda * clip_loss
            loss_value = loss.item()


        # batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss=rec_loss_value)
        metric_logger.update(clip_loss=clip_loss_value)


    print("validation stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
            test_history['epoch'] = epoch
            test_history['test_rec_loss'] = test_stats["rec_loss"]
            test_history['test_clip_loss'] = test_stats["clip_loss"]
            test_history['test_total_loss'] = test_stats["loss"]

    # plt.close('all')
    return test_stats, test_history