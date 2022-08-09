import os
import json
import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric.utils import to_undirected

class New_Dataset:
    def __init__(self, root, name, setting=None):

        self.root = root
        self.dataset = name

        self.features, self.labels, self.n_nodes = self.load_feature_label()
        self.adj = self.load_adj()
        # self.idx_train, self.idx_val, self.idx_test = self.load_idx()

    def load_feature_label(self):
        graph_node_features_and_labels_file_path = os.path.join(self.root, f'{self.dataset}_node_feature_label.txt')

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if self.dataset == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features = sp.csr_matrix(np.stack(list(graph_node_features_dict.values())))
        labels = np.array(list(graph_labels_dict.values()))

        n_nodes = features.shape[0]

        return features, labels, n_nodes

    def load_adj(self):
        graph_adjacency_list_file_path = os.path.join(self.root, f'{self.dataset}_graph_edges.txt')

        edge_index = []
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                edge_index.append([int(line[0]), int(line[1])])

        edge_index = torch.LongTensor(edge_index).T
        edge_index = to_undirected(edge_index)

        adj = sp.csr_matrix((np.ones_like(edge_index[0]), 
                            (edge_index[0], edge_index[1])), shape=(self.n_nodes, self.n_nodes))

        return adj

    def load_idx(self):
        with open(f'./pertubed_data/{self.dataset}_metattacked_nodes.json', 'r') as f:
            idx = json.loads(f.read())
        idx_train, idx_val, idx_test = tuple(np.array(x) for x in idx.values())

        return idx_train, idx_val, idx_test

    def load_preptbdata(self, attack_method, ptb_rate):
        self.adj = sp.load_npz(f'./pertubed_data/{self.dataset}_{attack_method}_adj_{ptb_rate}.npz')
        if attack_method == 'nettack':
            with open(f'./pertubed_data/{self.dataset}_nettacked_nodes.json', 'r') as f:
                idx = json.loads(f.read())
            self.target_nodes = idx['attacked_test_nodes']
    




