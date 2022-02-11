#!/opt/miniconda3/bin/python -Bu

import torch as pt
from torch_geometric.loader import DataLoader

from data import *
from model import *


batch_size = 1024
model = GNN(width=64*4, scale=3, depth=4, use_edge=True, use_atten=True, use_global=True, use_dense=True).cuda()  # base
model.load_state_dict(pt.load('submit/model1111.pt'))
model.eval()
print()


y_pred, y_true = [], []
loader = DataLoader(dataset[dataidx['valid']], batch_size=batch_size, shuffle=False, drop_last=False)
for batch_idx, batch in enumerate(loader):
    y_true.append(batch.y.numpy())
    y_pred.append(model(batch.cuda())[0].detach().cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
y_true = np.concatenate(y_true, axis=0).reshape(-1)
input_dict = {'y_pred': y_pred, 'y_true': y_true}
result_dict = dataeval.eval(input_dict)
print('#valid: %.4f' % result_dict['mae'])


y_pred = []
loader = DataLoader(dataset[dataidx['test-dev']], batch_size=batch_size, shuffle=False, drop_last=False)
for batch_idx, batch in enumerate(loader):
    y_pred.append(model(batch.cuda())[0].detach().cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
input_dict = {'y_pred': y_pred}
dataeval.save_test_submission(input_dict=input_dict, dir_path='submit', mode ='test-dev')
print('#testdev:', y_pred, y_pred.shape)

