# From Discovering Invariant Rationales for Graph Neural Networks

import os.path as osp
from typing import List, Dict, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.utils import coalesce
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import coalesce
from torch_geometric.utils import to_undirected


from gpl.datasets.spmotif_utils.gen_spmotif import gen_dataset, gen_dataset_general


def data_check_transform(data):
    assert data.edge_index.min() == 0
    assert data.edge_index.max() == data.x.shape[0] - 1
    return data

def reset_node_ids(data):
    node_label = data.node_label
    edge_index = data.edge_index
    fg_node_mask = node_label == 1
    N_nodes = len(node_label)
    curr_idx = N_nodes - 1
    node_id_mapper = {}
    for i in range(N_nodes):
        if node_label[i] == 1:
            node_id_mapper[i] = curr_idx
            curr_idx -= 1
    
    for i in range(N_nodes):
        if node_label[i] == 0:
            node_id_mapper[i] = curr_idx
            curr_idx -= 1
    
    mapper = torch.zeros((N_nodes,), dtype=torch.long)
    for i in range(N_nodes):
        mapper[i] = node_id_mapper[i]
    
    # for i in range(edge_index.shape[1]):
    data.edge_index[0] = mapper[data.edge_index[0]]
    data.edge_index[1] = mapper[data.edge_index[1]]
    
    return data


def remove_foreground_pre_transform(data):
    data = reset_node_ids(data)

    node_label = data.node_label
    edge_index = data.edge_index
    bg_node_mask = node_label == 0
    # fg_node_mask = node_label == 1
    # fg_edge_mask = (fg_node_mask[edge_index[0]] * fg_node_mask[edge_index[1]]).to(torch.bool)
    bg_edge_mask = bg_node_mask[edge_index[0]] * bg_node_mask[edge_index[1]]
    # fg_edge_mask.logical_not()

    x = data.x[bg_node_mask]
    edge_index = data.edge_index[:, bg_edge_mask]
    # import ipdb; ipdb.set_trace()
    edge_index = edge_index - edge_index.min()
    edge_attr = data.edge_attr[bg_edge_mask]
    node_label = data.node_label[bg_node_mask]
    edge_label = data.edge_label[bg_edge_mask]
    y = data.y

    new_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
    return new_data


def remove_background_pre_transform(data):
    data = reset_node_ids(data)

    node_label = data.node_label
    edge_index = data.edge_index
    # bg_node_mask = node_label == 0
    fg_node_mask = node_label == 1
    fg_edge_mask = fg_node_mask[edge_index[0]] * fg_node_mask[edge_index[1]]
    # bg_edge_mask = (1 - fg_edge_mask).to(torch.bool)

    # import ipdb; ipdb.set_trace()
    x = data.x[fg_node_mask]
    edge_index = data.edge_index[:, fg_edge_mask]
    edge_index = edge_index - edge_index.min()
    edge_attr = data.edge_attr[fg_edge_mask]
    node_label = data.node_label[fg_node_mask]
    edge_label = data.edge_label[fg_edge_mask]
    y = data.y

    new_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
    return new_data


def to_undirected_customized(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    edge_label: Optional[Union[Tensor, List[Tensor]]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:


    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0) # 变为双向edge
    edge_index = torch.stack([row, col], dim=0)

    if edge_attr is not None and isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif edge_attr is not None:
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]
    
    edge_label = torch.cat([edge_label, edge_label], dim=0) # edge label也要双倍
    # import ipdb; ipdb.set_trace()
    
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/coalesce.html
    edge_index_new, edge_attr = coalesce(edge_index, edge_attr, num_nodes, reduce)
    edge_index_new, edge_label = coalesce(edge_index, edge_label, num_nodes, reduce='mean')
    # import ipdb; ipdb.set_trace()

    return edge_index_new, edge_attr, edge_label


class MixClassTransform():
    def __init__(self, map_dict) -> None:
        self.map_dict = map_dict
    
    def __call__(self, data):
        data.y = torch.tensor( [self.map_dict[ data.y.item() ], ], dtype=torch.long).reshape(-1)
        
        return data



class SPMotif(InMemoryDataset):
    # splits = ['train', 'val', 'test']
    splits = ['train', 'val', 'test', 'test_foreground', 'test_background', 'test_both']

    def __init__(self, root, b, mode, dataset_name, NUMBER=3000, transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.b = b # bias
        self.mode = mode # train, val, test
        self.dataset_name = dataset_name
        self.NUMBER = NUMBER
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        # return ['train.npy', 'val.npy', 'test.npy', 'test_foreground', 'test_background', 'test_both']
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        # print(f'[INFO] Generating {self.dataset_name} dataset...')
        # gen_dataset(self.b, Path(self.raw_dir), self.dataset_name, NUMBER=self.NUMBER) # 里面有bug，生成的图的ground truth全都是circle
        # return

        label_motif_dict = {
            0: [{'number': 3000, 'list_shapes': [['house']] } ],
            1: [{'number': 3000, 'list_shapes': [['dircycle']]} ],
            2: [{'number': 3000, 'list_shapes': [['crane']]} ],
        }

        print(f'[INFO] Generating {self.dataset_name} dataset...')
        gen_dataset_general(self.b, Path(self.raw_dir), self.dataset_name, label_motif_dict=label_motif_dict)



    def process(self):
        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index).long()
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).reshape(-1)
            
            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            edge_label = torch.tensor(ground_truth, dtype=torch.float)

            edge_index, edge_attr, edge_label = to_undirected_customized(edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label, reduce='mean')

            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])
    

