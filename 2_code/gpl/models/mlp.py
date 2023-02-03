import torch.nn as nn
from torch_geometric.nn import InstanceNorm


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class MLPClean(nn.Sequential):
    def __init__(self, channels, dropout, with_softmax=False, bias=True):
        m = []
        for i in range(1, len(channels)): # e.g., [64, 64, 64]
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
        if with_softmax:
            m.append(nn.Softmax(dim=1))
        super(MLPClean, self).__init__(*m)