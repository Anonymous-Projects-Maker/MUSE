import yaml
import torch
import time
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.utils import degree
from torch import Tensor
from typing import List

def load_config(args):
    from gpl import CONFIG_DIR, LOG_DIR, TBLOG_DIR, TBLOG_HPARAMS_DIR, CKPT_DIR, TMP_DIR
    if args.train:
        exp_name_list = [args.dataset.name, args.model.name, time.strftime('%Y-%m-%d_%H-%M-%S'), f"seed{args.random_seed}", args.exp_note]
        EXP_NAME = '_'.join(exp_name_list)
        local_config = args
        local_config['evaluation']['ckpt_file'] = ''
        
        print('[Training args]')
        
    else:
        from gpl.training import yaml_load
        ckpt_file = CKPT_DIR/args.evaluation.ckpt_file
        EXP_NAME = args.evaluation.ckpt_file.split('/')[0]

        config_fn = TBLOG_HPARAMS_DIR/f'{EXP_NAME}.yml'
        local_config = yaml_load(config_fn)
        local_config = OmegaConf.create(local_config)

        local_config['evaluation']['ckpt_file'] = str(ckpt_file) 
        local_config['train'] = False 

        print(f'[OLD Evaluation args]: {str(config_fn)}')

    
    print(OmegaConf.to_yaml(local_config))
    
    return EXP_NAME, local_config



def get_local_config_name(model_name, dataset_name, override=False, task_id=-1):
    local_config_name = f'{model_name}-{dataset_name}'
    if override:
        assert task_id >= 0
        local_config_name += f'-override-taskid{task_id}'
    else:
        assert task_id == -1
    local_config_name += '.yml'

    return local_config_name

# copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/unbatch.html#unbatch_edge_index
def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)
