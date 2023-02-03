
from json import encoder
import torch.nn as nn
import torch.nn.functional as F

from gpl.models.gin import GIN
from gpl.models.pna import PNA
from gpl.models.mlp import MLP
from gpl.models.gpl import GPLV2
from gpl.models.prediction import Prediction
from gpl.models.gpl_v3 import GPLV3



def get_model(x_dim, edge_attr_dim, num_class, multi_label, deg, config):
    model_config = config['model']

    if model_config['name'] == 'GIN':
        encoder = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['name'] == 'PNA':
        encoder = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config, deg)
    else:
        raise ValueError('[ERROR] Unknown model name!')
    gpl_version = config['framework']['gpl_version']
    assert gpl_version == 'gpl_v3'
    gpl_model = GPLV3(encoder, config)
    gpl_model = gpl_model.to(config['device'])
    
    return gpl_model


def get_mlp(channels, dropout):
    model = MLP(channels, dropout)
    return model