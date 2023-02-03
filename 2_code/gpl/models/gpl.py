import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

from gpl.models.gin import GIN
from gpl.models.mlp import MLPClean
from gpl.training import get_optimizer

def gaussianKL(mean, std):
    term1 = (mean * mean).sum(dim=1).mean()
    term2 = std.square().sum(dim=1).mean()
    term3 = (std.square() + 1e-6 ).log().sum(dim=1).mean() # log-determiant of a diagonal matrix

    ib_const_loss = 1/2 * (term1 + term2 - term3)
    return ib_const_loss


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label):
        super(Criterion, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        print(f'[INFO] [criterion] Using multi_label: {self.multi_label}')

    def forward(self, logits, targets):
        if self.num_class == 2 and not self.multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float().view(-1, 1))
        elif self.num_class > 2 and not self.multi_label:
            loss = F.cross_entropy(logits, targets.long())
        else:
            is_labeled = targets == targets  # mask for labeled data
            loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss

class AssignerMLP(nn.Module):
    def __init__(self, channels, dropout_p, assign_edge=False):
        super().__init__()
        self.assign_edge = assign_edge

        if self.assign_edge:
            channels[0] = channels[0]*3
            self.feature_extractor = MLPClean(channels=channels, dropout=dropout_p, with_softmax=False)  # here we need to set with_softmax=False!!!
        else:
            self.feature_extractor = MLPClean(channels=channels, dropout=dropout_p, with_softmax=False) # here we need to set with_softmax=False!!!

    def forward(self, emb, edge_index): 
        if self.assign_edge:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            merged = torch.stack([f1, f2], dim=0)
            mean = merged.mean(dim=0)
            max, _ = merged.max(dim=0)
            min, _ = merged.min(dim=0)
            f12 = torch.cat([mean, max, min], dim=-1)

            assign_log_logits = self.feature_extractor(f12)
        else:
            assign_log_logits = self.feature_extractor(emb)
        return assign_log_logits


class GPLV1(nn.Module):
    def __init__(self, encoder: GIN, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.model_config = config['model']
        self.gpl_config = config['framework']
        self.training_config = config['training']
        self.num_class = self.gpl_config['num_class']
        self.multi_label = self.gpl_config['multi_label']
        self.with_assign_matrix = self.gpl_config['with_assign_matrix']
        self.with_ib_constraint = self.gpl_config['with_ib_constraint']
        self.ib_constraint_type = self.gpl_config['ib_constraint_type']
        self.ib_coeff_mask = self.gpl_config['ib_coeff_mask'] 
        self.ib_coeff_vector = self.gpl_config['ib_coeff_vector']
        self.assign_edge = self.gpl_config['assign_edge']
        self.pred_loss_coeff = self.gpl_config['pred_loss_coeff']
        self.fix_r = self.gpl_config['fix_r']
        self.decay_interval = self.gpl_config['decay_interval']
        self.init_r = self.gpl_config['init_r']
        self.final_r = self.gpl_config['final_r']
        self.decay_r = self.gpl_config['decay_r']
        self.criterion = Criterion(self.gpl_config['num_class'], self.gpl_config['multi_label'])
        self.with_reconstruct = self.gpl_config['with_reconstruct']
        self.recon_loss_coeff = self.gpl_config['recon_loss_coeff']
        self.pos_weight = self.gpl_config['pos_weight']
        output_dim = 1 if self.num_class == 2 and not self.multi_label else self.num_class
        assert len(self.model_config['clf_channels']) == 2
        clf_channels = self.model_config['clf_channels'] + [output_dim]
        print('[clf_channels]:', clf_channels)
        self.classifier = MLPClean(clf_channels, dropout=0, with_softmax=False)
        self.initialize()
        self.device = config.device

    def get_r(self):
        if self.fix_r:
            r = self.final_r
        else:
            current_epoch = self.__trainer__.cur_epoch
            r = self.init_r - current_epoch // self.decay_interval * self.decay_r
            if r < self.final_r:
                r = self.final_r
        return r


    def initialize(self):
        self.classifier_encoder = copy.deepcopy(self.encoder)
        if self.assign_edge:
            self.edge_assigner = AssignerMLP(channels=self.model_config['edge_assigner_channels'], dropout_p=self.model_config['dropout_p'], assign_edge=True)
        else:
            self.node_assigner = AssignerMLP(channels=self.model_config['node_assigner_channels'], dropout_p=self.model_config['dropout_p'], assign_edge=False)
        
        self.mean_encoder = MLPClean(self.model_config['mean_encoder_channels'], dropout=self.model_config['dropout_p'], with_softmax=False)
        self.std_encoder = MLPClean(self.model_config['std_encoder_channels'], dropout=self.model_config['dropout_p'], with_softmax=False)


        self.ib_coeff_scheduler = None



    def configure_optimizers(self):
        opt_params = self.training_config['optimizer_params']
        opt_type = opt_params['optimizer_type']
        lr = opt_params['lr']
        l2 = opt_params['l2']
        opt = get_optimizer(self, opt_type, lr, l2)
        return opt
    
    def sampling(self, logits: Tensor, gumbel: bool):
        if self.training and gumbel:
            mask = F.gumbel_softmax(logits, hard=False, tau=1) # [N, K]
        else:
            mask = logits.softmax(dim=1)
        mask = mask[:, 1]
        return mask

    