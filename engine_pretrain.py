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

        if 'ecg' in args.loss_type or args.loss_type == 'total':
            ecg = ecg.float().unsqueeze(1).to(device)
        else:
            ecg = None
        if 'cmr' in args.loss_type or args.loss_type == 'total':
            cmr = cmr.float().to(device)
        else:
            cmr = None
        if 'tar' in args.loss_type or args.loss_type == 'total':
            tar = tar.float().to(device)
        else:
            tar = None
        if 'snp' in args.loss_type or args.loss_type == 'total':
            snp = snp.float().to(device)
        else:
            snp = None
        cha = cha.float().to(device)
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        #
        # for i in range(cmr.shape[0]):
        #     nibabel.save(nibabel.Nifti1Image(cmr[i].cpu().numpy(), None), f'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/test/{data_iter_step}_cmr_{i}.nii.gz')
        #
            

        with torch.cuda.amp.autocast():
            loss,ecg_regression = model(ecg,cmr,tar,snp)
            loss_reg = loss_fn(ecg_regression, cha)
            loss_reg_value = loss_reg.item()
            loss_reg_name = args.downstream

            loss_value = loss.item()
            loss_name = args.loss_type
            total_loss = loss + loss_reg * args.lamda

        if not math.isfinite(loss_value) or not math.isfinite(loss_reg_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("Loss is {}, stopping training".format(loss_reg_value))
            sys.exit(1)


        total_loss /= accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_downstream=loss_reg_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    print("stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history[f'train_{loss_name}_loss'] = train_stats["loss"]
        training_history[f'train_{loss_reg_name}_loss'] = train_stats["loss_downstream"]
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
        ecg = None
        cmr = None
        tar = None
        snp = None
        I21 = batch[5]
        I42 = batch[6]
        I48 = batch[7]
        I50 = batch[8]
        if 'ecg' in args.loss_type or args.loss_type == 'total':
            ecg = batch[0].float().unsqueeze(1).to(device)
        if 'cmr' in args.loss_type or args.loss_type == 'total':
            cmr = batch[1].float().to(device)
        if 'tar' in args.loss_type or args.loss_type == 'total':
            tar = batch[2].float().to(device)
        if 'snp' in args.loss_type or args.loss_type == 'total':
            snp = batch[3].float().to(device)
        cha = batch[4].float().to(device)
        classification_dict = {'I21': I21, 'I42': I42, 'I48': I48, 'I50': I50}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()


        with torch.cuda.amp.autocast():
            loss,ecg_regression = model(ecg,cmr,tar,snp)
            loss_reg = loss_fn(ecg_regression, cha)
            loss_reg_value = loss_reg.item()
            loss_reg_name = args.downstream

            loss_value = loss.item()
            loss_name = args.loss_type

        label.append(cha.cpu().detach().numpy())
        out = ecg_regression.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)
        # batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_downstream=loss_reg_value)

    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)

    if args.downstream == 'classification':
        auc = roc_auc_score(label, output)
        metric_logger.update(auc=auc)
    import matplotlib.pyplot as plt

    # 创建一个画布
    # fig = plt.figure(figsize=(20, 82 * 5))
    if args.downstream == 'regression':
        corr_list = []
        for i in range(82):
            # 创建一个子图
            # ax = fig.add_subplot(82, 1, i + 1)
            # 计算相关系数
            corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            corr_list.append(corr)
            # 绘制散点图
            # ax.scatter(label[:, i].flatten(), output[:, i].flatten())
            # 在图上添加相关系数的文本
            # ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes)
            # 设置子图的标题
            # ax.set_title(f'Feature {i + 1}')

        metric_logger.update(correlation=np.mean(corr_list))

    print("validation stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.wandb == True:
            test_history['epoch'] = epoch
            test_history[f'val_{loss_name}_loss'] = test_stats["loss"]
            test_history[f'val_{loss_reg_name}_loss'] = test_stats["loss_downstream"]
            if args.downstream == 'classification':
                test_history['auc'] = test_stats['auc']
            if args.downstream == 'regression':
                test_history['correlation'] = np.mean(corr_list)
            # if epoch % 10 == 0:
            #     test_history['regression'] = wandb.Image(plt)
    # plt.close('all')
    return test_stats, test_history