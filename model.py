#!/opt/miniconda3/bin/python -Bu

import numpy as np
import torch as pt
import torch.nn as nn
import torch_geometric.nn as gnn


head_size = 64
#0 atomic_num: class120 *
#1 chirality: class4 *
#2 degree: value12
#3 formal_charge: value12[-5]
#4 numH: value10
#5 number_radical_e: value6
#6 hybridization: class6 *
#7 is_aromatic: bool
#8 is_in_ring: bool
node_size = (120, 4, 12, 12, 10, 6, 6, 2, 2)
#0 bond_type: class5 *
#1 bond_stereo: class6 *
#2 is_conjugated: bool
edge_size = (5, 6, 2)


class EmbedBlock(nn.Module):
    def __init__(self, size, width):
        super().__init__()
        self.size = size
        self.width = width

        self.embed0 = nn.Embedding(self.size[0]+1, self.width)
        self.embed1 = nn.ModuleList([nn.Embedding(s+1, self.width, padding_idx=0) for s in self.size[1:]])
        self.zero = nn.parameter.Parameter(pt.zeros(len(self.size) - 1))

    def forward(self, x, x0=None):
        x0 = self.embed0(x[:, 0])
        x1 = pt.concat([e(x[:, i+1])[:, :, None] for i, e in enumerate(self.embed1)], dim=-1)
        xx = x0 + pt.sum(x1 * pt.exp(self.zero), dim=-1)
        if x0 is None: return xx
        else: return (x0 + xx) / 2


class GlobBlock(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.width = width
        self.scale = scale

        self.zero = nn.parameter.Parameter(pt.tensor([0.0]))

        self.pre = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False))
        self.mem = nn.Sequential(nn.Linear(self.width, self.width*self.scale), nn.GELU())
        self.msg = nn.Sequential(nn.Linear(self.width, self.width*self.scale), nn.GELU())
        self.post = nn.Linear(self.width*self.scale, self.width)

    def forward(self, x, batch):
        # novel global node = pool(norm(linear(all nodes)))
        xx = self.pre(x)
        xx = self.mem(xx) + (self.msg(gnn.global_add_pool(xx, batch)))[batch] * pt.exp(self.zero)
        xx = self.post(xx)
        return x + xx, xx

class ConvBlock(gnn.MessagePassing):
    def __init__(self, width, scale, use_edge=False, use_atten=False):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.width = width
        self.scale = scale
        self.use_edge = use_edge
        self.use_atten = use_atten

        self.embed = EmbedBlock(edge_size, width)
        self.zero = nn.parameter.Parameter(pt.tensor([0.0]))

        self.pre = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False))
        self.mem = nn.Sequential(nn.Linear(self.width, self.width*self.scale), nn.GELU())
        self.msg = nn.Sequential(nn.Linear(self.width, self.width*self.scale), nn.GELU())
        self.post = nn.Linear(self.width*self.scale, self.width)

    def forward(self, x, edge_index, edge_attr, edge_embed=None):
        edge_embed = self.embed(edge_attr, edge_embed)
        xx = self.pre(x)
        xx = self.mem(xx) + self.propagate(edge_index, x=xx, edge_embed=edge_embed) * pt.exp(self.zero)
        xx = self.post(xx)
        return x + xx, xx, edge_embed

    def message(self, x_i, x_j, edge_embed):
        if self.use_edge:
            x_i, x_j = x_i + edge_embed, x_j + edge_embed
        if self.use_atten:
            size = len(edge_embed)
            atten = pt.sigmoid(pt.sum((x_i * x_j).reshape(size, -1, head_size), dim=-1, keepdim=True)) * 2
            return (self.msg(x_j).reshape(*atten.shape[:-1], -1) * atten).reshape(size, -1)
        else:
            return self.msg(x_j)

    def update(self, aggr_out):
        return aggr_out

class DenseBlock(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.width = width
        self.scale = scale

        self.block = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False),
                                   nn.Linear(self.width, self.width*self.scale), nn.GELU(),
                                   nn.Linear(self.width*self.scale, self.width))

    def forward(self, x):
        xx = self.block(x)
        return x + xx, xx


class GNN(nn.Module):
    def __init__(self, width, scale, depth, use_edge=False, use_atten=False, use_global=False, use_dense=False):
        super().__init__()
        self.width = width
        self.scale = scale
        self.depth = depth
        self.use_edge = use_edge
        self.use_atten = use_atten
        self.use_global = use_global
        self.use_dense = use_dense
        print('#model:', self.width, self.scale, self.depth, self.use_edge, self.use_atten, self.use_global, self.use_dense)
 
        self.embed = EmbedBlock(node_size, width)

        if use_global:
            self.glob = nn.ModuleList([GlobBlock(self.width, self.scale) for i in range(self.depth)])
        else:
            self.glob = [None] * self.depth
        self.conv = nn.ModuleList([ConvBlock(self.width, self.scale, use_edge, use_atten) for i in range(self.depth)])
        if use_dense:
            self.dense = nn.ModuleList([DenseBlock(self.width, self.scale) for i in range(self.depth)])
        else:
            self.dense = [None] * self.depth
        self.post = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False))

        self.head0 = nn.Linear(self.width, 1)
        self.head1 = nn.Linear(self.width, np.sum(node_size))
        print('#params:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, graph, mask=None):
        xx = graph.x.long().clone(); xx[:, 1:] += 1
        if mask is not None: xx[mask] = 0
        ea = graph.edge_attr.long().clone(); ea += 1

        xx, ee, xlst = self.embed(xx), None, []
        for i, glob, conv, dense in zip(range(self.depth), self.glob, self.conv, self.dense):
            if glob is not None: xx, xres = glob(xx, graph.batch)
            xx, xres, ee = conv(xx, graph.edge_index, ea, ee)
            if dense is not None: xx, xres = dense(xx)
            if i >= self.depth//2: xlst.append(nn.functional.layer_norm(xres, [self.width])[:, :, None])
        # novel global node = pool(norm(linear(all nodes)))
        xnode = self.post(xx)
        xglob = gnn.global_add_pool(xnode, graph.batch)

        x0 = self.head0(xglob)
        x1 = self.head1(xnode)
        x2 = pt.mean(pt.concat(xlst, dim=-1), dim=-1)
        return x0, x1, x2

