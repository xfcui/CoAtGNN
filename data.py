#!/opt/miniconda3/bin/python -Bu

import numpy as np
import torch as pt

from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import Data, HeteroData, Dataset


def preprocess(graph):
    # input
    size = graph.num_nodes
    loop = int(np.ceil(np.log2(size)))
    adj = pt.ones([size, size], dtype=pt.int16) * size
    adj[range(size), range(size)] = pt.arange(size, dtype=adj.dtype)
    adj[graph.edge_index[0], graph.edge_index[1]] = pt.minimum(graph.edge_index[0], graph.edge_index[1]).to(adj.dtype)
    idx = 0; code = 0; x2z_index = []; x2z_attr = []; z = []

    # global node
    x2z_index.extend([[i, idx] for i in range(size)])
    x2z_attr.extend([[code] for i in range(size)])
    z.append([code])
    idx += 1
    code += 1

    # aromatic node (one per aromatic group)
    aromatic = adj.clone()
    msk = graph.x[:, -2] == 0
    aromatic[msk, :] = size
    aromatic[:, msk] = size
    for i in range(loop):
        nanmin = pt.minimum(aromatic[:, :, None], aromatic[None, :, :])
        nanmax = pt.maximum(aromatic[:, :, None], aromatic[None, :, :])
        aromatic = pt.minimum(aromatic, pt.min(pt.where(nanmax>=size, nanmax, nanmin), dim=1)[0])
    aromatic = aromatic.diag()

    for i in sorted(pt.unique(aromatic).tolist()):
        if i == size: continue
        x2z_index.extend([[j, idx] for j in pt.where(aromatic == i)[0].tolist()])
        x2z_attr.extend([[code] for j in pt.where(aromatic == i)[0].tolist()])
        z.append([code])
        idx += 1
    code += 1

    # ring node (one per ring group)
    ring = adj.clone()
    msk = graph.x[:, -1] == 0
    ring[msk, :] = size
    ring[:, msk] = size
    for i in range(loop):
        nanmin = pt.minimum(ring[:, :, None], ring[None, :, :])
        nanmax = pt.maximum(ring[:, :, None], ring[None, :, :])
        ring = pt.minimum(ring, pt.min(pt.where(nanmax>=size, nanmax, nanmin), dim=1)[0])
    ring = ring.diag()

    for i in sorted(pt.unique(ring).tolist()):
        if i == size: continue
        x2z_index.extend([[j, idx] for j in pt.where(ring == i)[0].tolist()])
        x2z_attr.extend([[code] for j in pt.where(ring == i)[0].tolist()])
        z.append([code])
        idx += 1
    code += 1

    # output
    g = HeteroData()
    g['atom', 'bond', 'atom'].edge_index = graph.edge_index
    g['atom', 'bond', 'atom'].edge_attr = graph.edge_attr
    g['atom'].x = graph.x
    g['atom', 'part', 'void'].edge_index = pt.tensor(x2z_index, dtype=graph.edge_index.dtype).T
    g['atom', 'part', 'void'].edge_attr = pt.tensor(x2z_attr, dtype=graph.edge_attr.dtype)
    g['void'].x = pt.tensor(z, dtype=graph.x.dtype)
    g.y = graph.y
    return g


dataset = PygPCQM4Mv2Dataset(root='ogb', pre_transform=preprocess)
dataidx = dataset.get_idx_split()
dataeval = PCQM4Mv2Evaluator()


if __name__=="__main__":
    from torch_geometric.loader import DataLoader
    print('#data:', dataset[0])
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    for b in loader:
        print('#batch:', b)
        print('#bond:', b['bond'].edge_index)
        print('#part:', b['part'].edge_index)
        break
    print("#done!!!")

