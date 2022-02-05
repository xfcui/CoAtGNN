#!/opt/miniconda3/bin/python -Bu

import sys
import torch as pt

from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from model import *


batch_size = 4096
dataset = PygPCQM4Mv2Dataset(root='ogb', smiles2graph=smiles2graph)
split_dict = dataset.get_idx_split()
valid_idx = split_dict['valid']
testdev_idx = split_dict['test-dev']
testchallenge_idx = split_dict['test-challenge']
evaluator = PCQM4Mv2Evaluator()


model = GNN(width=64*4, scale=3, depth=4, use_edge=True, use_atten=True, use_global=True, use_dense=True).cuda()  # base
model.load_state_dict(pt.load('GIN11110-0099.pt'))
model.eval()
print()


y_pred, y_true = [], []
loader = DataLoader(dataset[valid_idx], batch_size=batch_size, shuffle=False, drop_last=False)
for batch_idx, batch in enumerate(loader):
    y_true.append(batch.y.numpy())
    y_pred.append(model(batch.cuda())[0].detach().cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
y_true = np.concatenate(y_true, axis=0).reshape(-1)
input_dict = {'y_pred': y_pred, 'y_true': y_true}
result_dict = evaluator.eval(input_dict)
print('#valid: %.4f' % result_dict['mae'])

y_pred = []
loader = DataLoader(dataset[testdev_idx], batch_size=batch_size, shuffle=False, drop_last=False)
for batch_idx, batch in enumerate(loader):
    y_pred.append(model(batch.cuda())[0].detach().cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
input_dict = {'y_pred': y_pred}
evaluator.save_test_submission(input_dict=input_dict, dir_path='.', mode ='test-dev')
print('#testdev:', y_pred, y_pred.shape)

