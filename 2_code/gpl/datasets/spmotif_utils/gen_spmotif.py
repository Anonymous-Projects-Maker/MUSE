# From Discovering Invariant Rationales for Graph Neural Networks

from .BA3_loc import *
from pathlib import Path
import random
from tqdm import tqdm
import itertools


def gen_one_label(dataset_name, number, bias, label, train_val_test, label_num=3, list_shapes=[[]]):
    """
    生成label为某个值的所有graphs。
    """
    
    # print(locals())
    # import ipdb; ipdb.set_trace()

    edge_index_list = []
    label_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    # bias = float(global_b)
    NUMBER = number
    base_bias = [] # 每一种base的被选中概率
    assert label_num >= 2
    assert label <= label_num-1

    for i in range(label_num):
        if i == label:
            base_bias.append( bias )
        else:
            base_bias.append( (1-bias)/(label_num-1) ) 

    # if label == 0:
    #     base_bias = [bias, (1-bias)/2, (1-bias)/2]
    # elif label == 1:
    #     base_bias = [(1-bias)/2, bias, (1-bias)/2]
    # elif label == 2:
    #     base_bias = [(1-bias)/2, (1-bias)/2, bias]
    # else:
    #     raise NotImplementedError

    e_mean = []
    n_mean = []

    base_choices = ['tree', 'ladder', 'wheel']
    base_choices = base_choices[:label_num]

    for _ in tqdm(range(NUMBER), desc=f"dataset {dataset_name}, {train_val_test}, label {label}"):
        if train_val_test == 'train':
            # base_num = np.random.choice([1,2,3], p=base_bias)
            base_type = np.random.choice(base_choices, p=base_bias)
        else:
            # base_num = np.random.choice([1,2,3])
            base_type = np.random.choice(base_choices)
        
        
        if 'simple' in dataset_name: # a simple version of this spmotif dataset
            # base_num = 1
            base_type = 'tree'


        # if base_num == 1:
        # 确定base图的大小
        if base_type == 'tree':
            if train_val_test in ['train', 'val']:
                width_basis=np.random.choice(range(3,4))
            else:
                width_basis=np.random.choice(range(3,6))
        elif base_type == 'ladder':
            if train_val_test in ['train', 'val']:
                width_basis=np.random.choice(range(8,12))
            else:
                width_basis=np.random.choice(range(30,50))
        elif base_type == 'wheel':
            if train_val_test in ['train', 'val']:
                width_basis=np.random.choice(range(15,20)) # 原始wheel的宽度
                # width_basis=np.random.choice(range(10,15))
            else:
                width_basis=np.random.choice(range(60,80))
        
        
        # if base_num == 2:
        #     base = 'ladder'
        #     if train_val_test in ['train', 'val']:
        #         width_basis=np.random.choice(range(8,12))
        #     else:
        #         width_basis=np.random.choice(range(30,50))
        #         # width_basis=np.random.choice(range(8,12))

        # if base_num == 3:
        #     base = 'wheel'
        #     if train_val_test in ['train', 'val']:
        #         width_basis=np.random.choice(range(15,20)) # 原始wheel的宽度
        #         # width_basis=np.random.choice(range(10,15))
        #     else:
        #         width_basis=np.random.choice(range(60,80))
        #         # width_basis=np.random.choice(range(15,20))
        #         # width_basis=np.random.choice(range(10,15))

        G, role_id, name = get_general(basis_type=base_type, width_basis=width_basis, list_shapes=list_shapes, 
                            feature_generator=None, draw=False)
        
        # get_crane(basis_type=base, nb_shapes=1,
        #                                 width_basis=width_basis, feature_generator=None, m=3, draw=False)

        # if label == 0:
        #     G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
        #                                 width_basis=width_basis, feature_generator=None, m=3, draw=False)
        # elif label == 1:
        #     G, role_id, name = get_house(basis_type=base, nb_shapes=1,
        #                                 width_basis=width_basis, feature_generator=None, m=3, draw=False)
        # elif label == 2:
        #     G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
        #                                 width_basis=width_basis, feature_generator=None, m=3, draw=False)
        # else:
        #     NotImplementedError
        

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        # row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(label) # NOTE
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(f'#nodes mean: {np.mean(n_mean):.1f}, #edges mean: {np.mean(e_mean):.1f}')
    return edge_index_list, label_list, ground_truth_list, role_id_list, pos_list


def gen_one_set_general(global_b, data_path, train_val_test, dataset_name, label_motif_dict):
    assert train_val_test in ['train', 'val', 'test']

    # NUMBER = 3000
    # Training/val/test Dataset
    edge_index_list_all_label = []
    label_list_all_label = []
    ground_truth_list_all_label = []
    role_id_list_all_label = []
    pos_list_all_label = []

    bias = float(global_b)
    label_num = len(set(label_motif_dict.keys())) # label的个数，也就是类的个数
    for label, scheme_list in label_motif_dict.items():
        for sub_scheme in scheme_list:
            number = sub_scheme['number']
            list_shapes = sub_scheme['list_shapes']

            if train_val_test != 'train': # 对于val和test set，样本只有label_motif_dict中指示的数量的1/5。
                number = int(number * 0.2)
        
            print('train_val_test:', train_val_test)
            print('label:', label)
            print('list_shapes:', list_shapes)

            edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = gen_one_label(dataset_name, 
                                                                                    number, bias, label, train_val_test, label_num=label_num, list_shapes=list_shapes)
            edge_index_list_all_label.extend(edge_index_list)
            label_list_all_label.extend(label_list)
            ground_truth_list_all_label.extend(ground_truth_list)
            role_id_list_all_label.extend(role_id_list)
            pos_list_all_label.extend(pos_list)

    # for label in [0, 1, 2]:
    #     edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = gen_one_label(dataset_name, NUMBER, bias, label, train_val_test)
    #     edge_index_list_all_label.extend(edge_index_list)
    #     label_list_all_label.extend(label_list)
    #     ground_truth_list_all_label.extend(ground_truth_list)
    #     role_id_list_all_label.extend(role_id_list)
    #     pos_list_all_label.extend(pos_list)
    
    np.save(data_path / f'{train_val_test}.npy', (edge_index_list_all_label, label_list_all_label, ground_truth_list_all_label, role_id_list_all_label, pos_list_all_label))



def gen_one_set(global_b, data_path, train_val_test, dataset_name, NUMBER):
    assert train_val_test in ['train', 'val', 'test']

    # NUMBER = 3000
    # Training/val/test Dataset
    edge_index_list_all_label = []
    label_list_all_label = []
    ground_truth_list_all_label = []
    role_id_list_all_label = []
    pos_list_all_label = []

    bias = float(global_b)
    
    for label in [0, 1, 2]:
        # import ipdb; ipdb.set_trace()
        if label == 0:
            list_shapes = [['dircycle']]
        elif label == 1:
            list_shapes = [['crane']] # !!!! 写成==了
        elif label == 2:
            list_shapes = [['house']]
        
        # import ipdb; ipdb.set_trace()

        edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = gen_one_label(dataset_name, NUMBER, bias, label, 
                                                                                train_val_test, label_num=3, list_shapes=list_shapes)
        edge_index_list_all_label.extend(edge_index_list)
        label_list_all_label.extend(label_list)
        ground_truth_list_all_label.extend(ground_truth_list)
        role_id_list_all_label.extend(role_id_list)
        pos_list_all_label.extend(pos_list)
    
    np.save(data_path / f'{train_val_test}.npy', (edge_index_list_all_label, label_list_all_label, ground_truth_list_all_label, role_id_list_all_label, pos_list_all_label))



def gen_dataset_general(global_b, data_path, dataset_name, label_motif_dict):
    """
    label_motif_dict确定label对应的motif有哪些。
    e.g.,
    label_motif_dict = {
        0: [
            {'number': 1500, 'list_shapes': [['house'], ['crane']] },
            {'number': 1500, 'list_shapes': [['house']] },
        ],
        1: [{'number': 1500, 'list_shapes': [['crane']]} ]
    }

    """
    gen_one_set_general(global_b, data_path, 'train', dataset_name, label_motif_dict)
    gen_one_set_general(global_b, data_path, 'val', dataset_name, label_motif_dict)
    gen_one_set_general(global_b, data_path, 'test', dataset_name, label_motif_dict)


def gen_dataset(global_b, data_path, dataset_name, NUMBER):
    # n_node = 0
    # n_edge = 0
    # for _ in range(500):
    #     # small:
    #     width_basis=np.random.choice(range(3,4))     # tree    #Node 32.55 #Edge 35.04
    #     # width_basis=np.random.choice(range(8,12))  # ladder  #Node 24.076 #Edge 34.603
    #     # width_basis=np.random.choice(range(15,20)) # wheel   #Node 21.954 #Edge 40.264
    #     # large:
    #     # width_basis=np.random.choice(range(3,6))   # tree    #Node 111.562 #Edge 117.77
    #     # width_basis=np.random.choice(range(30,50)) # ladder  #Node 83.744 #Edge 128.786
    #     # width_basis=np.random.choice(range(60,80)) # wheel   #Node 83.744 #Edge 128.786
    #     G, role_id, name = get_crane(basis_type="tree", nb_shapes=1,
    #                                         width_basis=width_basis,
    #                                         feature_generator=None, m=3, draw=False)
    #     role_id = np.array(role_id)
    #     edge_index = np.array(G.edges, dtype=int).T
    #     row, col = edge_index
    #     ground_truth = find_gd(edge_index, role_id)

    # #     pos = nx.spring_layout(G)
    # #     nx.draw_networkx_nodes(G, pos=pos, nodelist=range(len(G.nodes())), node_size=150,
    # #                            node_color=role_id, cmap='bwr',
    # #                            linewidths=.1, edgecolors='k')

    # #     nx.draw_networkx_labels(G, pos,
    # #                             labels={i: str(role_id[i]) for i in range(len(G.nodes))},
    # #                             font_size=10,
    # #                             font_weight='bold', font_color='k'
    # #                             )
    # #     nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), edge_color='black')
    # #     plt.show()

    #     n_node += len(role_id)
    #     n_edge += edge_index.shape[1]
    # print("#Node", n_node/1000, "#Edge", n_edge/1000)

    gen_one_set(global_b, data_path, 'train', dataset_name, NUMBER)
    gen_one_set(global_b, data_path, 'val', dataset_name, NUMBER)
    gen_one_set(global_b, data_path, 'test', dataset_name, NUMBER)


def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_general(basis_type, list_shapes, width_basis=8, feature_generator=None, draw=False):
    """
    向一个basis graph添加多个shape，由list_shapes所决定。例如
    [['dircycle'], ['crane'], ['house']]，这里需要是一个两层列表，以兼容synthetic_structsim.build_graph里面的调用
    """
    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str( itertools.chain.from_iterable(list_shapes) )

    return G, role_id, name



def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name
