import numpy as np
import torch as pt
import torch.nn as nn
import torch_geometric.nn as gnn


head_dim = 64
#0 atomic_num: class120 *
#1 chirality: class4 *
#2 degree: value12
#3 formal_charge: value12[-5]
#4 numH: value10
#5 number_radical_e: value6
#6 hybridization: class6 *
#7 aromatic+ring: class3 *
atom_size = (120, 4, 12, 12, 10, 6, 6, 3)
#0 bond_type: class5 *
#1 bond_stereo: class6 *
#2 is_conjugated: bool
#3 brics: class17 *
bond_size = (5, 6, 2, 17)
pseudo_size = (17,)


class EmbedBlock(nn.Module):
    def __init__(self, size, width):
        super().__init__()
        self.size = size
        self.width = width

        if len(self.size) == 1:
            self.embed = nn.Embedding(self.size[0]+1, self.width)
        else:
            self.init0 = nn.parameter.Parameter(pt.zeros(len(self.size)))
            self.embed = nn.ModuleList([nn.Embedding(self.size[0]+1, self.width)]
                                     + [nn.Embedding(s+1, self.width, padding_idx=0) for s in self.size[1:]])

    def forward(self, x, x_ema=None):
        if len(self.size) == 1:
            xx = self.embed(x.reshape(-1))
        else:
            xx, xw = 0, pt.exp(self.init0) / pt.sqrt(pt.sum(pt.exp(self.init0)))
            for i, e in enumerate(self.embed): xx += e(x[:, i]) * xw[i]
        if x_ema is None: return xx
        else: return (xx + x_ema) / 2  # exponential moving average


class HeteroBlock(gnn.MessagePassing):
    def __init__(self, width, scale, config, res=1.0):
        super().__init__(aggr="add", flow="source_to_target")
        self.width = width
        self.scale = scale
        self.use_frag = config['use_frag']
        self.atom_global = config['atom_global']
        self.frag_global = config['frag_global']
        self.a2f_atten = config['a2f_atten']
        self.f2a_atten = config['f2a_atten']
        self.res = res

        self.init0 = nn.parameter.Parameter(pt.zeros(6))

        self.pre = nn.ModuleList([nn.Sequential(nn.Linear(self.width, self.width),
                                                nn.LayerNorm(self.width, elementwise_affine=False)) for _ in range(2)])
        self.msg = nn.ModuleList([nn.Linear(self.width, self.width*self.scale) for _ in range(10)] + [nn.GELU()])
        self.post = nn.ModuleList([nn.Linear(self.width*self.scale, self.width) for _ in range(2)])

    def forward(self, x, edge_index, batch):
        if self.use_frag:
            xx, zz = self.pre[0](x[0]), self.pre[1](x[1])
            if self.atom_global and self.frag_global:
                xx, zz = self.msg[-1](self.msg[0](xx)) \
                       + self.msg[-1](self.msg[1](gnn.global_add_pool(xx, batch[0])))[batch[0]] * pt.exp(self.init0[0]) \
                       + self.propagate(pt.flip(edge_index, [0]), x=(zz, xx), size=(len(zz), len(xx)), \
                                        msg_idx=2, init0_idx=1, use_atten=self.f2a_atten) \
                       , self.msg[-1](self.msg[5](zz)) \
                       + self.msg[-1](self.msg[6](gnn.global_add_pool(zz, batch[1])))[batch[1]] * pt.exp(self.init0[3]) \
                       + self.propagate(edge_index, x=(xx, zz), size=(len(xx), len(zz)), \
                                        msg_idx=7, init0_idx=4, use_atten=self.a2f_atten)
            elif self.atom_global:
                xx, zz = self.msg[-1](self.msg[0](xx)) \
                       + self.msg[-1](self.msg[1](gnn.global_add_pool(xx, batch[0])))[batch[0]] * pt.exp(self.init0[0]) \
                       + self.propagate(pt.flip(edge_index, [0]), x=(zz, xx), size=(len(zz), len(xx)), \
                                        msg_idx=2, init0_idx=1, use_atten=self.f2a_atten) \
                       , self.msg[-1](self.msg[5](zz)) \
                       + self.propagate(edge_index, x=(xx, zz), size=(len(xx), len(zz)), \
                                        msg_idx=7, init0_idx=4, use_atten=self.a2f_atten)
            elif self.frag_global:
                xx, zz = self.msg[-1](self.msg[0](xx)) \
                       + self.propagate(pt.flip(edge_index, [0]), x=(zz, xx), size=(len(zz), len(xx)), \
                                        msg_idx=2, init0_idx=1, use_atten=self.f2a_atten) \
                       , self.msg[-1](self.msg[5](zz)) \
                       + self.msg[-1](self.msg[6](gnn.global_add_pool(zz, batch[1])))[batch[1]] * pt.exp(self.init0[3]) \
                       + self.propagate(edge_index, x=(xx, zz), size=(len(xx), len(zz)), \
                                        msg_idx=7, init0_idx=4, use_atten=self.a2f_atten)
            else:
                xx, zz = self.msg[-1](self.msg[0](xx)) \
                       + self.propagate(pt.flip(edge_index, [0]), x=(zz, xx), size=(len(zz), len(xx)), \
                                        msg_idx=2, init0_idx=1, use_atten=self.f2a_atten) \
                       , self.msg[-1](self.msg[5](zz)) \
                       + self.propagate(edge_index, x=(xx, zz), size=(len(xx), len(zz)), \
                                        msg_idx=7, init0_idx=4, use_atten=self.a2f_atten)
            xx, zz = self.post[0](xx), self.post[1](zz)
            return x[0] + xx * self.res, x[1] + zz * self.res
        else:
            xx = self.pre[0](x[0])
            if self.atom_global:
                xx = self.msg[-1](self.msg[0](xx)) \
                   + self.msg[-1](self.msg[1](gnn.global_add_pool(xx, batch[0])))[batch[0]] * pt.exp(self.init0[0])
            else: raise Exception('Not implemented model configuration!')
            xx = self.post[0](xx)
            return x[0] + xx * self.res, x[1]

    def message(self, x_i, x_j, msg_idx, init0_idx, use_atten):
        x_q = self.msg[msg_idx+0](x_i)
        x_k = self.msg[msg_idx+1](x_j)
        x_v = self.msg[msg_idx+2](x_j)
        x_v = self.msg[-1](x_v)
        if use_atten:
            # normalized dot product
            att = pt.sum((x_q * x_k).reshape(x_v.shape[0], -1, head_dim), dim=-1, keepdim=True) / np.sqrt(head_dim)
            # unnormalized softmax with scaling and bias
            att = pt.exp(att * self.init0[init0_idx] + self.init0[init0_idx+1])
            return (x_v.reshape(*att.shape[:-1], -1) * att).reshape(x_v.shape)
        else:
            return x_v * pt.exp(self.init0[init0_idx])

class ConvBlock(gnn.MessagePassing):
    def __init__(self, width, scale, edge_size, config, prefix, res=1.0):
        super().__init__(aggr="add", flow="source_to_target")
        self.width = width
        self.scale = scale
        self.hop = config[prefix + 'hop']
        self.use_edge = config[prefix + 'edge']
        self.use_atten = config[prefix + 'atten']
        self.res = res

        self.embed = nn.ModuleList([EmbedBlock(edge_size*i, self.width//2) for i in range(1, self.hop+1)])
        self.remix = nn.ModuleList([nn.Linear(self.width//2, self.width*self.scale) for _ in range(self.hop*3)])
        # 2 for attention of each hop + 2 for edge shifting
        self.init0 = nn.parameter.Parameter(pt.zeros(self.hop*2+2))

        self.pre = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False))
        # 1 for self + 3 for attention of each hop + GELU
        self.msg = nn.ModuleList([nn.Linear(self.width, self.width*self.scale) for _ in range(self.hop*3+1)] + [nn.GELU()])
        self.post = nn.Linear(self.width*self.scale, self.width)

    def forward(self, x, edge_index, edge_attr, edge_embed):
        xx = self.pre(x)
        if self.hop == 1:
            xx = self.msg[-1](self.msg[0](xx)) \
               + self.propagate(edge_index[0], x=xx, edge_embed=self.embed[0](edge_attr[0], edge_embed[0]), edge_type=0)
        elif self.hop == 2:
            xx = self.msg[-1](self.msg[0](xx)) \
               + self.propagate(edge_index[0], x=xx, edge_embed=self.embed[0](edge_attr[0], edge_embed[0]), edge_type=0) \
               + self.propagate(edge_index[1], x=xx, edge_embed=self.embed[1](edge_attr[1], edge_embed[1]), edge_type=1)
        elif self.hop == 3:
            xx = self.msg[-1](self.msg[0](xx)) \
               + self.propagate(edge_index[0], x=xx, edge_embed=self.embed[0](edge_attr[0], edge_embed[0]), edge_type=0) \
               + self.propagate(edge_index[1], x=xx, edge_embed=self.embed[1](edge_attr[1], edge_embed[1]), edge_type=1) \
               + self.propagate(edge_index[2], x=xx, edge_embed=self.embed[2](edge_attr[2], edge_embed[2]), edge_type=2)
        else: raise Exception('Unknown hop number:', self.hop)
        xx = self.post(xx)
        return x + xx * self.res, edge_embed

    def message(self, x_i, x_j, edge_embed, edge_type):
        idx = edge_type * 3
        x_q = self.msg[idx+1](x_i)
        x_k = self.msg[idx+2](x_j)
        x_v = self.msg[idx+3](x_j)
        if self.use_edge:
            # edge shifted query, key, value
            x_q += self.remix[idx+0](edge_embed) * pt.exp(self.init0[-2])
            x_k += self.remix[idx+1](edge_embed) * pt.exp(self.init0[-2])
            x_v += self.remix[idx+2](edge_embed) * pt.exp(self.init0[-1])
        x_v = self.msg[-1](x_v)
        if self.use_atten:
            # normalized dot product
            att = pt.sum((x_q * x_k).reshape(edge_embed.shape[0], -1, head_dim), dim=-1, keepdim=True) / np.sqrt(head_dim)
            # unnormalized softmax with scaling and bias
            att = pt.exp(att * self.init0[edge_type] + self.init0[edge_type+self.hop])
            return (x_v.reshape(*att.shape[:-1], -1) * att).reshape(x_v.shape)
        else:
            return x_v * pt.exp(self.init0[edge_type])

class DenseBlock(nn.Module):
    def __init__(self, width, scale, res=1.0):
        super().__init__()
        self.width = width
        self.scale = scale
        self.res = res

        self.block = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width, elementwise_affine=False),
                                   nn.Linear(self.width, self.width*self.scale), nn.GELU(),
                                   nn.Linear(self.width*self.scale, self.width))

    def forward(self, x):
        xx = self.block(x)
        return x + xx * self.res, xx


class GNN(nn.Module):
    def __init__(self, width, scale, depth, config):
        super().__init__()
        self.width = width
        self.scale = scale
        self.depth = depth
        self.config = config
        assert 1 <= config['atom_hop'] <= 3
        assert 0 <= config['frag_hop'] <= 1
        print('#model:', width, width*scale//head_dim, depth)
 
        self.embed = nn.ModuleList([EmbedBlock(atom_size, width), nn.Embedding(1, self.width)])

        res = 1 / depth
        self.hetero = nn.ModuleList([HeteroBlock(width, scale, config, res=res) for i in range(depth)])
        self.conv_atom = nn.ModuleList([ConvBlock(width, scale, bond_size, config, prefix='atom_', res=res) for i in range(depth)])
        self.conv_frag = nn.ModuleList([ConvBlock(width, scale, pseudo_size, config, prefix='frag_', res=res) for i in range(depth)])
        self.dense_atom = nn.ModuleList([DenseBlock(width, scale, res=res) for i in range(depth)])
        self.dense_frag = nn.ModuleList([DenseBlock(width, scale, res=res) for i in range(depth)])
        self.post = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width, elementwise_affine=False))

        self.head = nn.Linear(width, 1)
        print('#params:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, graph, mask=None):
        xx = graph['atom'].x.long().clone(); xx[:, 1:] += 1
        if mask is not None: xx[mask] = 0
        xx, xlst = self.embed[0](xx), []
        zz = self.embed[1](graph['frag'].x.long().reshape(-1))
        xei = []; xea = []; xee = []
        if self.config['atom_hop'] > 0:
            xei.append(graph['bond'].edge_index.long())
            xea.append(graph['bond'].edge_attr.long().clone()); xea[-1] += 1
            xee.append(None)
        if self.config['atom_hop'] > 1:
            xei.append(graph['2hop'].edge_index.long())
            xea.append(graph['2hop'].edge_attr.long().clone()); xea[-1] += 1
            xee.append(None)
        if self.config['atom_hop'] > 2:
            xei.append(graph['3hop'].edge_index.long())
            xea.append(graph['3hop'].edge_attr.long().clone()); xea[-1] += 1
            xee.append(None)
        if self.config['frag_hop'] == 1:
            zei = [graph['pseudo'].edge_index.long()]
            zea = [graph['pseudo'].edge_attr.long().clone()]; zea[-1] += 1
            zee = [None]
        xzei = graph['part'].edge_index.long()

        for i, hetero, conv_atom, conv_frag, dense_atom, dense_frag in \
                zip(range(self.depth), self.hetero, self.conv_atom, self.conv_frag, self.dense_atom, self.dense_frag):
            if self.config['atom_global'] or self.config['use_frag']:
                xx, zz = hetero((xx, zz), xzei, (graph['atom'].batch, graph['frag'].batch))
            xx, xee = conv_atom(xx, xei, xea, xee)
            if self.config['atom_dense']:
                xx, xres = dense_atom(xx)
            if self.config['use_frag']:
                if self.config['frag_hop'] > 0:
                    zz, zee = conv_frag(zz, zei, zea, zee)
                if self.config['atom_dense']:
                    zz, _ = dense_frag(zz)
            #if i >= self.depth//2: xlst.append(nn.functional.layer_norm(xres, [self.width])[:, :, None])
        # novel global node = pool(norm(linear(all nodes)))
        xglob = gnn.global_add_pool(self.post(xx), graph['atom'].batch)

        return self.head(xglob), None  #pt.mean(pt.concat(xlst, dim=-1), dim=-1)

