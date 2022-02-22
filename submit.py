#!/home/xfcui/miniconda3/bin/python3 -Bu

import re
import sys
import torch as pt
from time import time
from torch_geometric.loader import DataLoader

from data import *
from model import *


model_name = sys.argv[1] if len(sys.argv) > 1 else 'gnn-11111-000-00000'
dev_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

model_config = re.sub('\D+', '', model_name)
assert len(model_config) == 13, model_config
model_config = {'atom_global':  model_config[0] == '1',
                'atom_hop': int(model_config[1]),
                'atom_edge':    model_config[2] == '1',
                'atom_atten':   model_config[3] == '1',
                'atom_dense':   model_config[4] == '1',
                'use_frag':     model_config[5] == '1',
                'a2f_atten':    model_config[6] == '1',
                'f2a_atten':    model_config[7] == '1',
                'frag_global':  model_config[8] == '1',
                'frag_hop': int(model_config[9]),
                'frag_edge':    model_config[10] == '1',
                'frag_atten':   model_config[11] == '1',
                'frag_dense':   model_config[12] == '1'}

pt.cuda.set_device(dev_id)
pt.multiprocessing.set_sharing_strategy('file_system')
print('#device:', dev_id, pt.__version__, pt.version.cuda, pt.backends.cuda.matmul.allow_tf32, pt.backends.cudnn.allow_tf32)


batch_size = 512
model = GNN(width=64*4, scale=3, depth=4, config=model_config).cuda()  # base
model.load_state_dict(pt.load('submit/%s.pt' % model_name))
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


y_pred, t0 = [], time()
loader = DataLoader(dataset[dataidx['test-dev']], batch_size=batch_size, shuffle=False, drop_last=False)
for batch_idx, batch in enumerate(loader):
    y_pred.append(model(batch.cuda())[0].detach().cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
input_dict = {'y_pred': y_pred}
dataeval.save_test_submission(input_dict=input_dict, dir_path='submit', mode ='test-dev')
print('#testdev:', y_pred, y_pred.shape, '%.1fs' % (time()-t0))

