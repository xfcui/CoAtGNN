#!/opt/miniconda3/bin/python -Bu

import sys
import numpy as np
import torch as pt
from time import time

from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from model import *
from lamb.lamb import Lamb
from torch.optim.swa_utils import AveragedModel


model_name = sys.argv[1] if len(sys.argv) > 1 else 'GIN11110'
use_edge = model_name[-5] == '1'
use_atten = model_name[-4] == '1'
use_global = model_name[-3] == '1'
use_dense = model_name[-2] == '1'
model_size = int(model_name[-1])
rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pt.cuda.set_device(rank)
print('#device:', rank, pt.backends.cuda.matmul.allow_tf32, pt.backends.cudnn.allow_tf32)


batch_size, batch_aug, num_epoch = 1024//4**model_size, 4, 200
dataset = PygPCQM4Mv2Dataset(root='ogb', smiles2graph=smiles2graph)
split_idx = dataset.get_idx_split()
train_dataset = dataset[split_idx['train']]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataset = dataset[split_idx['valid']]
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
evaluator = PCQM4Mv2Evaluator()
print('#data:', len(train_dataset), len(valid_dataset), batch_size)


def configOpt(model):
    config = [{'params': [], 'lr': learn_rate, 'weight_decay': 0},
              {'params': [], 'lr': learn_rate, 'weight_decay': weight_decay},
              {'params': [], 'lr': learn_rate/2, 'weight_decay': weight_decay*2}]
    for n, p in model.named_parameters():
        if n.find('embed') >= 0: config[0]['params'].append(p)
        elif n.endswith('zero'): config[0]['params'].append(p)
        elif n.endswith('bias'): config[0]['params'].append(p)
        elif n.startswith('head'): config[2]['params'].append(p)
        elif n.endswith('weight'): config[1]['params'].append(p)
        else: raise Exception('Unknown parameter name:', n)
    return config

learn_rate, weight_decay = 1e-3/2**model_size, 1e-1
if model_size == 0:
    student = GNN(width=64*4, scale=3, depth=4, use_edge=use_edge, use_atten=use_atten, use_global=use_global, use_dense=use_dense).cuda()  # base
elif model_size == 1:
    student = GNN(width=64*6, scale=4, depth=12, use_edge=use_edge, use_atten=use_atten, use_global=use_global, use_dense=use_dense).cuda()  # small: base * 3 * 3
elif model_size == 2:
    student = GNN(width=64*8, scale=4, depth=24, use_edge=use_edge, use_atten=use_atten, use_global=use_global, use_dense=use_dense).cuda()  # large: small * 1.77 * 2
else:
    raise Exception('Unknown model size:', model_size)
opt = Lamb(configOpt(student), lr=learn_rate, weight_decay=weight_decay)
sched_period = len(train_dataset)//batch_size
sched = pt.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, sched_period)
ema_decay = 1e-4
teacher = AveragedModel(student, avg_fn=lambda x, y, n: ema_decay * x + (1 - ema_decay) * y).cuda()
teacher.update_parameters(student)


def maskNode(edge, batch, radius_max=2):
    radius_p = np.array([10.0**i for i in range(radius_max, -1, -1)]); radius_p /= np.sum(radius_p)
    radius = np.random.choice(radius_max+1, p=radius_p)
    prob = 0.15 / [1.0, 3.0, 5.5][radius]

    mask0 = pt.rand(len(batch)).cuda() < prob
    mask1 = mask0.clone().detach()
    for i in range(radius):
        mask1[edge[0][pt.any(pt.where(mask1)[0][:, None] == edge[None, 1], dim=0)]] = True
    return mask0, mask1

print()
print('#training model ...')
for epoch in range(num_epoch):
    check_stat = 0
    epoch_time = check_time = time()

    student.train()
    teacher.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.cuda()
        lr = sched.get_last_lr()[1]

        loss = 0
        opt.zero_grad()
        true0, true1 = batch.y, teacher(batch)[-1]
        for i in range(batch_aug):
            mask0, mask1 = maskNode(batch.edge_index, batch.batch)
            pred0, pred1 = student(batch, mask1)
            loss += nn.functional.l1_loss(pred0.reshape(-1), true0.reshape(-1))
            loss += nn.functional.smooth_l1_loss(pred1[mask0], true1[mask0], beta=4.0)
        loss /= batch_aug
        loss.backward()
        opt.step()
        sched.step()
        teacher.update_parameters(student)

        with pt.no_grad():
            check_stat = check_stat * 0.9 + loss.item() * 0.1  # moving average
            if time() - check_time > 20:
                e = epoch + (batch_idx + 1) / sched_period
                print('#check[%.3f]: %.4f %.2e' % (e, check_stat, lr))
                check_time = time()

    student.eval()
    teacher.eval()
    y_pred, y_true = [], []
    for batch_idx, batch in enumerate(valid_loader):
        y_true.append(batch.y.numpy())
        y_pred.append(teacher(batch.cuda())[0].detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    result_dict = evaluator.eval(input_dict)

    e = epoch + 1
    pt.save(teacher.state_dict(), 'model/%s-%04d.pt' % (model_name, epoch))
    print('#valid[%.3f]: %.4f %.1fs *' % (e, result_dict['mae'], time() - epoch_time))
    epoch_time = time()
print('#done!!!')

