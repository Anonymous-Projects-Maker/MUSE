import string
import torch
from typing import Dict
import numpy as np
import random
import ipdb
from torch.utils.data import Sampler
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.transforms import Constant
from ogb.graphproppred import PygGraphPropPredDataset

from gpl.datasets import graph_sst2

def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    # if random_state is not None:
    #     np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        n_train_valid = int(splits['train'] * len(idx)) + int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train_valid]
        test_idx = idx[n_train_valid:]
        # test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
        
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

def get_dataset(dataset_name: string, splits: Dict, device='cpu', mutag_x=False, compute_degree=False):
    """
    return train, val, test, dataset
    """
    from gpl import DATA_DIR
    from gpl.datasets.mutag import Mutag
    from gpl.datasets.spmotif import SPMotif, SPMotifMix, SPMotifDiverse, SPMotifDiverseV2, SPMotifDiverseV3
    from gpl.datasets.mnist import MNIST75sp
    

    if dataset_name == 'mutag':
        dataset = Mutag(root=DATA_DIR/'mutag', device=device)
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        train_set = dataset[split_idx['train']]
        val_set = dataset[split_idx['valid']]
        test_set = dataset[split_idx['test']]
    elif 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=DATA_DIR/dataset_name, name=dataset_name)
        dataset.data.x = dataset.data.x.to(torch.float)
        dataset.data.edge_attr = dataset.data.edge_attr.to(torch.float)
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        
        train_set = dataset[split_idx['train']]
        val_set = dataset[split_idx['valid']]
        test_set = dataset[split_idx['test']]
    elif 'spmotif' in dataset_name:
        dataset_name_splitted = dataset_name.split('_')
        b = float(dataset_name_splitted[-1])
        train_set = SPMotif(root=DATA_DIR/dataset_name, b=b, mode='train', dataset_name=dataset_name)
        val_set = SPMotif(root=DATA_DIR/dataset_name, b=b, mode='val', dataset_name=dataset_name)
        test_set = SPMotif(root=DATA_DIR/dataset_name, b=b, mode='test', dataset_name=dataset_name)
        print('[INFO] Using default splits!')
    elif  dataset_name == 'Graph-SST2':
        shift = False if 'noshift' in dataset_name else True
        dataset = graph_sst2.get_dataset(dataset_dir=DATA_DIR, dataset_name=dataset_name, task=None)
        train_set, val_set, test_set = graph_sst2.get_splitted_datasets(dataset, degree_bias=shift, seed=0)
        print('[INFO] Using default splits!')
    elif dataset_name == 'mnist':
        n_train_data, n_val_data = 20000, 5000
        train_val = MNIST75sp(root=DATA_DIR/'mnist', mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
        train_val = train_val[perm_idx]
        train_set, val_set = train_val[:n_train_data], train_val[-n_val_data:]
        test_set = MNIST75sp(root=DATA_DIR/'mnist', mode='test')
        print('[INFO] Using default splits!')
    else: raise ValueError

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    multi_label = False 
    y_unique = torch.cat( [train_set.data.y.unique(), val_set.data.y.unique(), test_set.data.y.unique()] ).unique()
    num_class = y_unique.numel()
    if train_set.data.y.dim() > 1 and train_set.data.y.shape[1] > 1:
        multi_label = True

    if compute_degree:
        batched_train_set = Batch.from_data_list(train_set)
        d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    else:
        d = None

    extra_info = {}
    extra_info['x_dim'] = x_dim
    extra_info['edge_attr_dim'] = edge_attr_dim
    extra_info['num_class'] = num_class
    extra_info['multi_label'] = multi_label
    extra_info['deg'] = d

    return train_set, val_set, test_set, extra_info

class IndexPreserveRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        indexes = list(range(n))
        random.shuffle(indexes)
        self.indexes = indexes

        return iter(indexes)

    def __len__(self):
        return len(self.data_source)

def get_dataloaders(train_set, val_set, test_set, batch_size=16, batch_sampler=None):
    from torch_geometric.loader import DataLoader
    train_sampler = IndexPreserveRandomSampler(train_set)
    val_sampler = IndexPreserveRandomSampler(val_set)
    test_sampler = IndexPreserveRandomSampler(test_set)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True,)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True,)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, pin_memory=True,)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }