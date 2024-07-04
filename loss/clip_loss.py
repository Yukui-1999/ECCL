"""
This NTXentLoss implementation is taken from: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/loss/nt_xent.py
"""

import torch
import torch.nn.functional as F

class ClipLoss(torch.nn.Module):

    def __init__(self, temperature, alpha_weight, args):
        super(ClipLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = temperature
        self.device = args.device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.args = args
    # def softXEnt(self, target, logits):
    #     logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
    #     loss = -(target * logprobs).sum() / logits.shape[0]
    #     return loss

    def forward(self, zis, zjs,norm=True):
        temperature = self.temperature
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        # labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = torch.arange(len(hidden1)).to(self.device)
        logits = torch.matmul(hidden1, torch.transpose(hidden2,0, 1)) / temperature
        zis_findmostgood_zjs = F.cross_entropy(logits, labels)
        zjs_findmostgood_zis = F.cross_entropy(torch.transpose(logits,0, 1), labels)
        # loss = F.cross_entropy(logits, labels) + F.cross_entropy(torch.transpose(logits,0, 1), labels)
        # print(f'zis_findmostgood_zjs:{zis_findmostgood_zjs},zjs_findmostgood_zis:{zjs_findmostgood_zis}')
        loss = self.args.alpha_weight * zis_findmostgood_zjs + (1 - self.args.alpha_weight) * zjs_findmostgood_zis
        return loss