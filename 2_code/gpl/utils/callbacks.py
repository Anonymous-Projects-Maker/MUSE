import copy
import torch
from torch_geometric.data import Batch

class ModelFunction():
    def __init__(self, model, instance) -> None:
        self.model = model
        self.instance = instance

    def mask_instance(self, mask):
        masked_ins = None
        return masked_ins

    def __call__(self, masked_ins):
        # masked_ins = self.mask_instance(mask)
        # print('masked_ins:', masked_ins)
        # print('masked_ins.shape:', masked_ins.shape)
        # print('instance:', self.instance)
        # import ipdb; ipdb.set_trace()
        masked_instances = []
        for i in range(masked_ins.shape[0]):
            ins_tmp = copy.deepcopy(self.instance)
            ins_tmp.edge_index = ins_tmp.edge_index[:, torch.tensor(masked_ins[i], dtype=torch.bool)] # 去掉mask为0的边
            # import ipdb; ipdb.set_trace()
            ins_tmp.edge_attr = ins_tmp.edge_attr[torch.tensor(masked_ins[i], dtype=torch.bool), :] # 去掉相应的边的feature
            masked_instances.append( ins_tmp  )
        
        batch_data = Batch.from_data_list(masked_instances)
        with torch.no_grad():
            self.model.eval()
            return_dict = self.model.forward_pass(batch_data, batch_idx=0, compute_loss=False)
        # self.return_dict = return_dict
        
        prob = return_dict['clf_logits'].softmax(dim=1)
        prob = prob[:, self.instance.y]
        prob = prob.detach().cpu().flatten().tolist()

        # import ipdb; ipdb.set_trace()

        return prob 

    def get_edge_mask(self):
        batch_data = Batch.from_data_list([self.instance,])
        with torch.no_grad():
            self.model.eval()
            return_dict = self.model.forward_pass(batch_data, batch_idx=0, compute_loss=False)
        self.__return_dict__ = return_dict
        return return_dict['edge_mask']

def SHAPmarker(mask, model_arg):
    # print('mask:', mask)
    # print('mask.shape:', mask.shape)
    # print('model_arg:', model_arg)
    model_arg = model_arg * mask
    return model_arg.reshape((1, -1))


def save_train_embeds_callback(test_results, **kwargs):
    from gpl import TMP_DIR
    from gpl.utils.evaluate import embedding_evaluate
    model = kwargs['model']
    dataloaders = kwargs['dataloaders']
    exp_name = kwargs['__trainer__'].EXP_NAME
    emb_acc, emb_auc, (train_embs, train_y) = embedding_evaluate(model, dataloaders)
    emb_save_dir = TMP_DIR/'embs'
    emb_save_dir.mkdir(parents=True, exist_ok=True)
    emb_save_name = emb_save_dir/f'{exp_name}.pt'
    torch.save({'train_embs': train_embs, 'train_y': train_y}, f=emb_save_name)
    print(f'saved at {emb_save_name}')

