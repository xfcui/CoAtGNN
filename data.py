import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
  bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import InMemoryDataset


def hetero_transform(graph):
    # input
    num_nodes = graph.num_nodes
    loop = int(np.ceil(np.log2(num_nodes)))
    graph.edge_index = graph.edge_index.long()
    head = [graph.edge_index[0] == i for i in range(num_nodes)]
    head = [[graph.edge_index[1, i], graph.edge_attr[i]] for i in head]
    tail = [graph.edge_index[1] == i for i in range(num_nodes)]
    tail = [[graph.edge_index[0, i], graph.edge_attr[i]] for i in tail]
    idx = 0; x2z_index = []; z = []

    # two-hop edge
    twohop_index, twohop_attr = [], []
    for i1 in range(num_nodes):
        ei0, ea0 = tail[i1]
        ei1, ea1 = head[i1]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i2, a1 in zip(ei1.tolist(), ea1.tolist()):
                if i0 == i2: continue  # loop
                twohop_index.append([i0, i2])
                twohop_attr.append(a0 + a1)

    # three-hop edge
    threehop_index, threehop_attr = [], []
    for (i1, i2), a1 in zip(graph.edge_index.T.tolist(), graph.edge_attr.tolist()):
        ei0, ea0 = tail[i1]
        ei2, ea2 = head[i2]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i3, a2 in zip(ei2.tolist(), ea2.tolist()):
                if i0 == i2 or i0 == i3 or i1 == i3: continue  # loop
                threehop_index.append([i0, i3])
                threehop_attr.append(a0 + a1 + a2)

    # BRICS node
    adj = torch.ones([num_nodes, num_nodes], dtype=torch.int16) * num_nodes
    adj[range(num_nodes), range(num_nodes)] = torch.arange(num_nodes, dtype=adj.dtype)
    for (i, j), ea in zip(graph.edge_index.T, graph.edge_attr):
        if ea[-1] == 0:  # not BRICS bond
            adj[i, j] = torch.minimum(i, j).to(adj.dtype)
    for i in range(loop):
        nanmin = torch.minimum(adj[:, :, None], adj[None, :, :])
        nanmax = torch.maximum(adj[:, :, None], adj[None, :, :])
        adj = torch.minimum(adj, torch.min(torch.where(nanmax>=num_nodes, nanmax, nanmin), dim=1)[0])
    adj = adj.diag()
    for i in sorted(torch.unique(adj).tolist()):
        if i == num_nodes: continue
        assert i >= idx
        ii = adj == i
        adj[ii] = idx
        x2z_index.extend([[j, idx] for j in torch.where(ii)[0].tolist()])
        z.append([0])
        idx += 1
    buf_index, buf_attr = [], []
    for (i, j), ea in zip(graph.edge_index.T, graph.edge_attr):
        if ea[-1] > 0:  # BRICS bond
            buf_index.append([adj[i].item(), adj[j].item()])
            buf_attr.append([ea[-1].item(),])
    if len(buf_index) > 0:
        pseudo_index = np.array(buf_index, dtype=np.int32)
        pseudo_attr = np.array(buf_attr, dtype=np.int16)
    else:
        pseudo_index = np.empty([0, 2], dtype=np.int32)
        pseudo_attr = np.empty([0, 1], dtype=np.int16)

    # output
    g = HeteroData()
    g.__num_nodes__ = num_nodes
    g['atom', 'bond', 'atom'].edge_index = graph.edge_index.int()
    g['atom', 'bond', 'atom'].edge_attr = graph.edge_attr.byte()
    g['atom', '2hop', 'atom'].edge_index = torch.tensor(twohop_index).T.int()
    g['atom', '2hop', 'atom'].edge_attr = torch.tensor(twohop_attr).byte()
    g['atom', '3hop', 'atom'].edge_index = torch.tensor(threehop_index).T.int()
    g['atom', '3hop', 'atom'].edge_attr = torch.tensor(threehop_attr).byte()
    g['atom'].x = graph.x.byte()
    g['frag', 'pseudo', 'frag'].edge_index = torch.tensor(pseudo_index).T.int()
    g['frag', 'pseudo', 'frag'].edge_attr = torch.tensor(pseudo_attr).byte()
    g['atom', 'part', 'frag'].edge_index = torch.tensor(x2z_index).T.int()
    g['frag'].x = torch.tensor(z).byte()
    g.y = graph.y.float()
    return g


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root = 'dataset', transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
        '''

        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def smiles2graph(self, smiles_string):
        """
        Converts SMILES string to graph Data object
        :input: SMILES string (str)
        :return: graph object
        """

        mol = Chem.MolFromSmiles(smiles_string)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            v = atom_to_feature_vector(atom)
            v = v[:-2] + [v[-2] + v[-1]]
            atom_features_list.append(v)
        x = np.array(atom_features_list, dtype = np.int16)
        num_nodes = len(x)

        # bonds
        num_bond_features = 4  # bond type, bond stereo, is_conjugated
        brics = np.zeros([num_nodes, num_nodes], dtype = np.int16)
        for (i, j), (s, t) in BRICS.FindBRICSBonds(mol):
            try: s, t = int(s), int(t)
            except: s = t = 7  # (s, t) = (7a, 7b)
            brics[i, j] = s
            brics[j, i] = t
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature + [brics[i, j]])
                edges_list.append((j, i))
                edge_features_list.append(edge_feature + [brics[j, i]])

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int32).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int16)
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int32)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int16)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = num_nodes
        return graph 

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).int()
            data.edge_attr = torch.from_numpy(graph['edge_feat']).byte()
            data.x = torch.from_numpy(graph['node_feat']).byte()
            data.y = torch.Tensor([homolumogap]).float()

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


dataset = PygPCQM4Mv2Dataset(root='ogb', pre_transform=hetero_transform)
dataidx = dataset.get_idx_split()
dataeval = PCQM4Mv2Evaluator()


if __name__=="__main__":
    from torch_geometric.loader import DataLoader

    g0 = dataset[0]
    print('#data:', g0)
    for k in ['atom', 'frag']:
        print('#dtype:', k, g0[k].x.dtype)
    for k in ['bond', '2hop', '3hop', 'pseudo']:
        print('#dtype:', k, g0[k].edge_index.dtype, g0[k].edge_attr.dtype)
    print('#dtype:', 'part', g0['part'].edge_index.dtype)
    print('#dtype:', 'y', g0.y.dtype)
    print()

    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    for b in loader:
        print('#batch:'); print(b)
        print('#bond:'); print(b['bond'].edge_index)
        print('#two-hop:'); print(b['2hop'].edge_index)
        print('#three-hop:'); print(b['3hop'].edge_index)
        print('#pseudo-bond:'); print(b['pseudo'].edge_index)
        print('#part:'); print(b['part'].edge_index)
        print()
        break

    print('#done!!!')

