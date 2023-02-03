import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from torch import Tensor

from gpl.models.gin import GIN
from gpl.models.mlp import MLPClean
from gpl.models.gpl import GPLV1, gaussianKL, maskConstKL
from gpl.models.mcr2 import MaximalCodingRateReduction

class AssignerMLPWithZ(nn.Module):
    def __init__(self, channels, dropout_p, assign_edge=False):
        super().__init__()
        self.assign_edge = assign_edge

        in_dim = channels[0]
        if self.assign_edge:
            channels[0] = in_dim*3 + in_dim 
            self.feature_extractor = MLPClean(channels=channels, dropout=dropout_p, with_softmax=False)  # here we need to set with_softmax=False!!!
        else:
            channels[0] = in_dim + in_dim
            self.feature_extractor = MLPClean(channels=channels, dropout=dropout_p, with_softmax=False) # here we need to set with_softmax=False!!!

    def forward(self, emb, edge_index, batch, Z): 
        if self.assign_edge:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            
            merged = torch.stack([f1, f2], dim=0)
            mean = merged.mean(dim=0)
            max, _ = merged.max(dim=0)
            min, _ = merged.min(dim=0)
            entity_feature = torch.cat([mean, max, min], dim=-1)
            
            edge_batch = batch[row]
            Z_ext = Z[edge_batch]
            
        else:
            entity_feature = emb
            Z_ext = Z[batch]

        embs = torch.cat([entity_feature, Z_ext], dim=-1)
        assign_log_logits = self.feature_extractor(embs)
        return assign_log_logits


class MaskSmoothLayer(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, mask, edge_index, assign_edge):
        if assign_edge:
            avg_node_mask = scatter(mask.reshape(-1), index=edge_index[0], reduce='mean') # 从edge prob来induce node prob
            smoothed_mask = (avg_node_mask[edge_index[0]] + avg_node_mask[edge_index[1]] ) / 2 # edge的两个node求均值得到平均的edge mask
        else:
            avg_edge_mask = (mask[edge_index[0]] + mask[edge_index[1]] ) / 2 # 从node mask得到edge mask
            smoothed_mask = scatter(avg_edge_mask, index=edge_index[0], reduce='mean') # 再从edge mask得到平均的node mask
        
        smoothed_mask = smoothed_mask.reshape((-1, 1)) # [|E|, 1]
        mask = (1-self.gamma) * mask + self.gamma * smoothed_mask
        return mask



def reconstruction_loss(recon_adj, edge_index: Tensor, batch, _slice_dict, _inc_dict, pos_weight):
    
    device = edge_index.device
    recon_loss = 0
    
    node_slice = _slice_dict['x']
    edge_index_slice = _slice_dict['edge_index']
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    recon_mae_total = 0
    recon_pos_total = 0
    recon_neg_total = 0

    batch_size = batch.max()+1
    for i in range(batch_size):
        n = node_slice[i+1] - node_slice[i]

        gt_adj_this = torch.zeros((n, n), dtype=torch.float32, device=device)
        start_idx = edge_index_slice[i]
        end_idx = edge_index_slice[i+1]
        decrement = _inc_dict['edge_index'][i]
        edge_index_single = edge_index[:, start_idx:end_idx]

        edge_index_single = edge_index_single - decrement
        row, col = edge_index_single[0], edge_index_single[1]
        gt_adj_this[row, col] = 1 
        gt_adj_this = gt_adj_this.reshape(-1)

        recon_adj_this = recon_adj[i].reshape(-1)
        recon_loss += F.binary_cross_entropy_with_logits(recon_adj_this, gt_adj_this, pos_weight=pos_weight)

        recon_mae = torch.absolute(recon_adj_this-gt_adj_this).cpu().detach().mean()
        recon_mae_total += recon_mae

        recon_pos = recon_adj_this[gt_adj_this==1].cpu().detach().mean()
        recon_pos_total += recon_pos
        recon_neg = recon_adj_this[gt_adj_this==0].cpu().detach().mean()
        recon_neg_total += recon_neg
    
    recon_loss = recon_loss/batch_size # batch average
    recon_mae_total = recon_mae_total/batch_size 
    recon_pos_total = recon_pos_total/batch_size
    recon_neg_total = recon_neg_total/batch_size

    return recon_loss, recon_mae_total, recon_pos_total, recon_neg_total


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, Z, batch, _slice_dict):
        Z = self.fc1(Z)
        adj_batch = []
        node_slice = _slice_dict['x']
        for i in range(batch.max()+1):
            Z_this = Z[node_slice[i]: node_slice[i+1]]
            inner_prod = torch.matmul(Z_this, Z_this.T)
            adj = self.sigmoid(inner_prod)
            adj_batch.append( adj )
        
        assert sum([adj.shape[0] for adj in adj_batch]) == Z.shape[0]

        return adj_batch

class GPLV3(GPLV1):
    def __init__(self, encoder: GIN, config):
        super().__init__(encoder, config)
        
    def initialize(self, ):
        self.subg_encoder = copy.deepcopy(self.encoder)
        self.mean_encoder = MLPClean(self.model_config['mean_encoder_channels'], dropout=self.model_config['dropout_p'], with_softmax=False)
        self.std_encoder = MLPClean(self.model_config['std_encoder_channels'], dropout=self.model_config['dropout_p'], with_softmax=False)

        if self.assign_edge:
            self.edge_assigner = AssignerMLPWithZ(channels=self.model_config['edge_assigner_channels'], dropout_p=self.model_config['dropout_p'], assign_edge=True)
        else:
            self.node_assigner = AssignerMLPWithZ(channels=self.model_config['node_assigner_channels'], dropout_p=self.model_config['dropout_p'], assign_edge=False)

        if self.with_rate_loss:
            self.rate_loss_ins = MaximalCodingRateReduction()
        
        if self.gpl_config.mask_smooth:
            self.mask_smooth_layer = MaskSmoothLayer(gamma=0.5)
        
        if self.with_reconstruct:
            self.reconstruct_decoder = InnerProductDecoder(64, 64, 0.1)
    

    def __loss__(self, clf_logits, mean, std, node_mask, edge_mask, subg_embs, embs_recon, data):
        step_dict = {}
        clf_labels = data.y
        edge_index = data.edge_index
        batch = data.batch

        if self.pred_loss_coeff == 0.0:
            pred_loss = torch.tensor(0.0)
        else:
            pred_loss = self.criterion(clf_logits, clf_labels)
            pred_loss = pred_loss * self.pred_loss_coeff
        step_dict['pred_loss'] = pred_loss

        mask_value = edge_mask
        if self.with_ib_constraint:
            if self.ib_constraint_type == 'vector':
                ib_const_loss_gaussian = gaussianKL(mean, std)
                ib_const_loss_gaussian = ib_const_loss_gaussian * self.ib_coeff_vector
                ib_const_loss = ib_const_loss_gaussian
                step_dict['vib_loss'] = ib_const_loss_gaussian.item()

            elif self.ib_constraint_type == 'mask':
                ib_const_loss_mask = maskConstKL(mask_value, self.get_r())
                ib_const_loss_mask = ib_const_loss_mask * self.ib_coeff_mask
                ib_const_loss = ib_const_loss_mask
                step_dict['eib_loss'] = ib_const_loss_mask.item()
            
            elif self.ib_constraint_type == 'both':
                ib_const_loss_gaussian = gaussianKL(mean, std) * self.ib_coeff_vector
                ib_const_loss_mask = maskConstKL(mask_value, self.get_r()) * self.ib_coeff_mask
                ib_const_loss = ib_const_loss_gaussian + ib_const_loss_mask
                step_dict['vib_loss'] = ib_const_loss_gaussian.item()
                step_dict['eib_loss'] = ib_const_loss_mask.item()
        else:
            ib_const_loss = torch.tensor(0.0)
        
        if self.with_reconstruct:
            adj_batch = self.reconstruct_decoder(embs_recon, batch, _slice_dict=data._slice_dict)
            recon_loss, recon_mae, recon_pos_mean, recon_neg_mean = reconstruction_loss(adj_batch, edge_index, batch, 
                        _slice_dict=data._slice_dict, _inc_dict=data._inc_dict, pos_weight=self.pos_weight)

            recon_loss = recon_loss * self.recon_loss_coeff

        loss = pred_loss + ib_const_loss

        if self.with_reconstruct:
            loss += recon_loss
            step_dict['recon_loss'] = recon_loss.item()
            step_dict['recon_mae'] = recon_mae.item()
            step_dict['recon_pos_mean'] = recon_pos_mean.item()
            step_dict['recon_neg_mean'] = recon_neg_mean.item()

        step_dict['loss'] = loss
        step_dict['pred_loss'] = pred_loss.item()
        step_dict['ib_loss'] = ib_const_loss.item()

        return step_dict


    def forward_pass(self, data, batch_idx, compute_loss=True):
        return_dict = self.get_embs(data)

        # classification
        clf_logits = self.classifier(return_dict['subg_embs'])

        data = data.to(self.device)
        if compute_loss:
            loss_dict = self.__loss__(clf_logits=clf_logits, 
                                    mean=return_dict['mean'],
                                    std=return_dict['std'],
                                    node_mask=return_dict['node_mask'],
                                    edge_mask=return_dict['edge_mask'],
                                    subg_embs=return_dict['subg_embs'],
                                    embs_recon=return_dict['embs_recon_node'],
                                    data=data,
                                )
        else:
            loss_dict = {}
        
        loss_dict['clf_logits'] = clf_logits
        loss_dict['node_mask'] = return_dict['node_mask']
        loss_dict['edge_mask'] = return_dict['edge_mask']
        loss_dict['mean'] = return_dict['mean']

        loss_dict['y'] = data.y
        loss_dict['batch'] = data.batch
        loss_dict['edge_index'] = data.edge_index

        if not hasattr(data, 'edge_label'):
            loss_dict['exp_labels'] = torch.zeros((data.edge_index.shape[1]), device=data.edge_index.device)
        else:
            loss_dict['exp_labels'] = data.edge_label

        return loss_dict

    def get_mask(self, N, embs, edge_index, batch, sampled_Z):
        edge_assign_logits = self.edge_assigner(embs, edge_index, batch, sampled_Z) # [N, 2], N is the number of edges
        edge_mask = self.sampling(edge_assign_logits, gumbel=True)
        node_mask = torch.ones((embs.shape[0],), device=embs.device)
        edge_mask = edge_mask.reshape(-1, 1)
        node_mask= node_mask.reshape(-1, 1)

        return edge_mask, node_mask
    
    def get_subg_encoder_embs(self, data):
        if isinstance(data, Batch):
            data = data.to(self.device)
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            new_embs = self.subg_encoder.get_emb(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr) # node-level embeddings
            new_embs = scatter(new_embs, batch, dim=0, reduce='sum') # [B, dim]
        elif isinstance(data, Data):
            new_embs = self.subg_encoder.get_emb(x=data.x, edge_index=data.edge_index, batch=None, edge_attr=data.edge_attr) # node-level embeddings
            new_embs = new_embs.sum(dim=0)
        else: raise ValueError

        return new_embs

    def get_embs(self, data):
        data = data.to(self.device)
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        assert self.encoder.graph_pooling is False, 'Should obtain node embeddings now'
        N = x.shape[0]

        assert self.with_assign_matrix is True
        assert self.with_ib_constraint is True
        assert self.assign_edge is True

        hidden_size = self.encoder.hidden_size 
        embs = self.encoder.get_emb(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr) # node-level embeddings.  
        
        
        embs_mask = embs[:, :hidden_size] 
        embs_recon = embs[:, hidden_size:] 

        Z_mask = scatter(embs_mask, batch, dim=0, reduce='sum') # [B, hidden_dim]
        embs_recon_graph = scatter(embs_recon, batch, dim=0, reduce='sum') # [B, hidden_dim]

        mean = self.mean_encoder(Z_mask)
        std = F.relu( self.std_encoder(Z_mask) )
        sampled_Z = mean
        edge_mask, node_mask = self.get_mask(N, embs_mask, edge_index, batch, sampled_Z)
        new_embs = self.subg_encoder.get_emb(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, edge_atten=edge_mask) # node-level embeddings
        new_embs = new_embs * node_mask
        new_embs = scatter(new_embs, batch, dim=0, reduce='sum')
        
        return_dict = {
            'subg_embs': new_embs,
            'mean': mean,
            'std': std,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'embs_recon_node': embs_recon,
            'embs_recon_graph': embs_recon_graph,
        }
        return return_dict

    