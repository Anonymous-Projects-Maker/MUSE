
import numpy as np
from collections import OrderedDict
import torch
from sklearn.metrics import roc_auc_score
import wandb


def compute_auc(preds, labels, binary=True):
    # preds are logits
    if binary is True:
        auc = roc_auc_score(y_true=labels, y_score=preds)
    else:
        probabilities = torch.tensor(preds).softmax(dim=1).numpy()
        auc = roc_auc_score(y_true=labels, y_score=probabilities, multi_class='ovr')
    
    return auc

def compute_acc_auc(clf_logits, y):
    if clf_logits.shape[1] == 1: # binary classificarion
        preds = (clf_logits >= 0).flatten()
        y = y.flatten()
        acc = (preds == y).mean() # it may be >=2 classes here.
        auc = compute_auc(clf_logits, y, binary=True)

    else: # multiple classificarion (not multi-label)
        acc = (np.argmax(clf_logits, axis=1) == y).mean()
        auc = compute_auc(clf_logits, y, binary=False)
    return acc, auc


def process_one_set(results):
    loss = np.mean(results['loss'] )
    # acc and auc
    clf_logits = np.concatenate(results['clf_logits'])
    y = np.concatenate(results['y'])
    acc, auc = compute_acc_auc(clf_logits, y)
    return loss, acc, auc

def KNNEvaluate(proto_embs, proto_y, test_embs, test_y):
    classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    classifier.fit(proto_embs, proto_y)
    num_classes = len(np.unique(test_y))

    pred = classifier.predict_proba(test_embs)
    pred_hard = np.argmax(pred, axis=1)
    acc = (pred_hard == test_y).mean()

    if pred.shape[1] == 1:
        auc = roc_auc_score(y_true=test_y, y_score=pred)
        print(f'pred shape is {pred.shape}')
    elif num_classes == 2:
        auc = roc_auc_score(y_true=test_y, y_score=pred[:, 1])
    elif num_classes > 2:
        auc = roc_auc_score(y_true=test_y, y_score=pred, multi_class='ovr')
    else:
        raise NotImplementedError

    return acc, auc


def get_embeddings(model, dataloader, key):
    model.eval()
    if isinstance(key, str):
        embs_all = []
    elif isinstance(key, list):
        embs_all = {}
        for k in key:
            embs_all[k] = []
    else: raise ValueError

    y_all = []
    with torch.no_grad():
        for batch in dataloader:
            results_dict = model.get_embs(batch)
            if isinstance(key, str):
                embs = results_dict[key]
                embs_all.append(embs)
            elif isinstance(key, list):
                for k in key:
                    embs_all[k].append(results_dict[k])
            else: raise ValueError
            y_all.append(batch.y)
        
        indices = np.array(dataloader.sampler.indexes)

    if isinstance(key, str):
        embs_all = torch.cat(embs_all, dim=0).cpu().numpy()
    elif isinstance(key, list):
        for k in key:
            embs_all[k] = torch.cat(embs_all[k], dim=0).cpu().numpy()

    y_all = torch.cat(y_all, dim=0).cpu().numpy()

    return embs_all, y_all, indices



def prototype_performance(**kwargs):
    from gpl.utils.mmd.mmd_critic import select_prototypes_criticisms

    # prepare
    dataloaders = kwargs['dataloaders']
    model = kwargs['model']
    train_dataloader = dataloaders.train_dataloader
    test_dataloader = dataloaders.test_dataloader
    device = next(model.parameters()).device

    # prototype info
    embs_dict, train_y, _ = get_embeddings(model, train_dataloader, key=['subg_embs', 'embs_recon_graph'])
    train_embs = np.concatenate([embs_dict['subg_embs'], embs_dict['embs_recon_graph']], axis=1)
    train_y = train_y.reshape(-1)

    embs_dict, test_y, _ = get_embeddings(model, test_dataloader, key=['subg_embs', 'embs_recon_graph'])
    test_embs = np.concatenate([embs_dict['subg_embs'], embs_dict['embs_recon_graph']], axis=1)
    test_y = test_y.reshape(-1)

    # prototypes
    max_prototypes = 100
    prototyes_dataset = select_prototypes_criticisms(train_embs, train_y, num_prototypes=max_prototypes)
    prototype_indices = prototyes_dataset.prototype_indices.numpy()
    criticism_indices = prototyes_dataset.criticism_indices.numpy()
    
    prototypes = prototyes_dataset.prototypes
    prototype_labels = prototyes_dataset.prototype_labels
    criticisms = prototyes_dataset.criticisms
    criticism_labels = prototyes_dataset.criticism_labels

    auc_list = []
    acc_list = []

    for proto_num in [20, 40, 60, 80, 100]:
        print(f'######### prot_num: {proto_num}')
        prototypes_part = prototypes[:proto_num]
        prototype_labels_part = prototype_labels[:proto_num]
        acc, auc = KNNEvaluate(prototypes_part, prototype_labels_part, test_embs, test_y)
        print(f'prototype acc: {acc:.4f}, auc: {auc:.4f}')
        auc_list.append(auc)
        acc_list.append(acc)
    
    print('auc: ', [f'{val:.4f}' for val in auc_list])
    print('acc: ', [f'{val:.4f}' for val in acc_list])
   



def explain_precision_at_k(results, k):
    """
    att, exp_labels, k, batch, edge_index
    exp_labels: explanation labels
    """
    att_all = results['edge_mask']
    exp_labels_all = results['exp_labels']
    batch_all = results['batch']
    edge_index_all = results['edge_index']

    precision_at_k = []
    for att, exp_labels, batch, edge_index in zip(att_all, exp_labels_all, batch_all, edge_index_all): # 所有的batch
        att = att.flatten()
        exp_labels = exp_labels.flatten()
        batch = batch.flatten()

        for i in range(batch.max()+1): 
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
    
    return np.mean(precision_at_k)


def explain_auc(results):
    att_all = np.concatenate( results['edge_mask'], axis=0)
    exp_labels_all = np.concatenate( results['exp_labels'], axis=0)
    att_all = att_all.flatten()
    exp_labels_all = exp_labels_all.flatten()

    if np.unique(exp_labels_all).shape[0] > 1:
        att_auroc = roc_auc_score(exp_labels_all, att_all)
    else:
        att_auroc = 0
    return att_auroc

def write_logs_to_tensorboard(writer, logging_dict, cur_epoch):
    for key, val in logging_dict.items():
        writer.add_scalar(key, val, cur_epoch)

def write_logs_to_wandb(logging_dict, cur_epoch):
    logging_dict['epoch'] = cur_epoch
    wandb.log(logging_dict)


def log_metrics(**kwargs):
    has_val_set = kwargs['val_results'] is not None

    logger = kwargs['logger']
    cur_epoch = kwargs['cur_epoch']

    # loss
    loss_train, acc_train, auc_train = process_one_set(kwargs['train_results'])
    loss_test, acc_test, auc_test = process_one_set(kwargs['test_results'])
    if has_val_set:
        loss_val, acc_val, auc_val = process_one_set(kwargs['test_results'])
    
    y_test = np.concatenate(kwargs['test_results']['y'])
    assert y_test.ndim == 1 or y_test.shape[1] == 1, 'only care about non-multi-label case now'

    # explanation metrics
    evaluation_config = kwargs['__trainer__'].hparams_save['evaluation']
    k = evaluation_config['precision_k']

    exp_precision_k_train = explain_precision_at_k(kwargs['train_results'], k)
    exp_precision_k_test = explain_precision_at_k(kwargs['test_results'], k)
    exp_auc_train = explain_auc(kwargs['train_results'])
    exp_auc_test = explain_auc(kwargs['test_results'])

    logger.info('###############################')
    logger.info(f'[epoch {cur_epoch+1}]')
    logger.info(f'Loss_train: {loss_train:.4f}, Acc_train: {acc_train:.4f}, Auc_train: {auc_train:.4f}')
    logger.info(f'Loss_test : {loss_test:.4f}, Acc_test : {acc_test:.4f}, Auc_test : {auc_test:.4f}')
    logger.info(f'exp_precision@{k}_train : {exp_precision_k_train:.4f}, exp_precision@{k}_test : {exp_precision_k_test:.4f}')
    logger.info(f'exp_auc_train : {exp_auc_train:.4f}, exp_auc_test : {exp_auc_test:.4f}')
    logger.info('###############################')
    
    ############################### 写到log（以及可能的wandb&tensorboard中）
    LOGGING_DICT = OrderedDict()
    # prediction loss
    loss_train_pred = np.mean( kwargs['train_results']['pred_loss'] )
    loss_test_pred = np.mean( kwargs['test_results']['pred_loss'] )
    LOGGING_DICT['overall_loss/train'] = loss_train
    LOGGING_DICT['overall_loss/test'] = loss_test
    LOGGING_DICT['pred_loss/train'] = loss_train_pred
    LOGGING_DICT['pred_loss/test'] = loss_test_pred

    LOGGING_DICT['acc/train'] = acc_train
    LOGGING_DICT['acc/test'] = acc_test
    LOGGING_DICT['auc/train'] = auc_train
    LOGGING_DICT['auc/test'] = auc_test
    if has_val_set:
        LOGGING_DICT['acc/val'] = acc_val
        LOGGING_DICT['auc/val'] = auc_val

    # explanation metrics
    LOGGING_DICT[f'exp_pre@{k}/train'] = exp_precision_k_train
    LOGGING_DICT[f'exp_pre@{k}/test'] = exp_precision_k_test
    LOGGING_DICT[f'exp_auc/train'] = exp_auc_train
    LOGGING_DICT[f'exp_auc/test'] = exp_auc_test

    # ib loss
    if kwargs['model'].with_ib_constraint:
        loss_ib_train = np.mean( kwargs['train_results']['ib_loss'] )
        loss_ib_test = np.mean( kwargs['test_results']['ib_loss'] )
        LOGGING_DICT['ib_loss/train'] = loss_ib_train
        LOGGING_DICT['ib_loss/test'] = loss_ib_test
        if 'vib_loss' in kwargs['train_results']:
            vib_loss_train = np.mean( kwargs['train_results']['vib_loss'] )
            vib_loss_test = np.mean( kwargs['test_results']['vib_loss'] )
            LOGGING_DICT['vib_loss/train'] = vib_loss_train
            LOGGING_DICT['vib_loss/test'] = vib_loss_test
        if 'eib_loss' in kwargs['train_results']:
            eib_loss_train = np.mean( kwargs['train_results']['eib_loss'] )
            eib_loss_test = np.mean( kwargs['test_results']['eib_loss'] )
            LOGGING_DICT['eib_loss/train'] = eib_loss_train
            LOGGING_DICT['eib_loss/test'] = eib_loss_test
    
    # reconstruction loss
    if kwargs['model'].with_reconstruct:
        loss_recon_train = np.mean( kwargs['train_results']['recon_loss'] )
        loss_recon_test = np.mean( kwargs['test_results']['recon_loss'] )

        recon_mae_train = np.mean( kwargs['train_results']['recon_mae'] )
        recon_mae_test = np.mean( kwargs['test_results']['recon_mae'] )

        recon_pos_mean_train = np.mean( kwargs['train_results']['recon_pos_mean'] )
        recon_pos_mean_test = np.mean( kwargs['test_results']['recon_pos_mean'] )
        recon_neg_mean_train = np.mean( kwargs['train_results']['recon_neg_mean'] )
        recon_neg_mean_test = np.mean( kwargs['test_results']['recon_neg_mean'] )
        

        LOGGING_DICT['recon_loss/train'] = loss_recon_train
        LOGGING_DICT['recon_loss/test'] = loss_recon_test

        LOGGING_DICT['recon_mae/train'] = recon_mae_train
        LOGGING_DICT['recon_mae/test'] = recon_mae_test

        LOGGING_DICT['recon_pos_mean/train'] = recon_pos_mean_train
        LOGGING_DICT['recon_pos_mean/test'] = recon_pos_mean_test

        LOGGING_DICT['recon_neg_mean/train'] = recon_neg_mean_train
        LOGGING_DICT['recon_neg_mean/test'] = recon_neg_mean_test


    # some monitoring values
    weight_node = np.mean( np.concatenate(kwargs['train_results']['node_mask']) )
    weight_edge = np.mean( np.concatenate(kwargs['train_results']['edge_mask']) )
    curr_r = kwargs['model'].get_r()
    LOGGING_DICT['mask_weight/node'] = weight_node
    LOGGING_DICT['mask_weight/edge'] = weight_edge
    LOGGING_DICT['mask_weight/curr_r'] = curr_r

    for key, val in LOGGING_DICT.items():
        logger.info(f"{key}: {val}")
    logger.info('###############################')
        

    # write to wandb and tensorboard
    if kwargs['__trainer__'].training_mode and not kwargs['__trainer__'].debug:
        write_logs_to_wandb(LOGGING_DICT, cur_epoch)
        if kwargs['__trainer__'].log2tensorboard:
            write_logs_to_tensorboard(kwargs['tb_writer'], LOGGING_DICT, cur_epoch)