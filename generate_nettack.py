import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--ptb_rate', type=int, default=5.0, help='Perturbation rate.')
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data = Dataset(root='./data/', name=args.dataset)
# adj, features, labels = data.adj, data.features, data.labels

# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
from utils import load_data
features, edge_index, labels = load_data()
n = features.shape[0]
indices = np.random.permutation(n)
num = int(n * 0.1)
idx_train, idx_val, idx_test = indices[:num], indices[num: num*5], indices[num*5:]
adj = csr_matrix((np.ones_like(edge_index[0]), (edge_index[0], edge_index[1])), shape=(n, n))

idx_unlabeled = np.union1d(idx_val, idx_test)

features = csr_matrix(features)
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

# Setup Attack Model
target_node = 0
assert target_node in idx_unlabeled

def attack_all():
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = np.intersect1d(np.where(degrees>=5)[0], idx_test) # obtain the nodes to be attacked
    num = len(node_list)
    print('=== Attacking %s nodes sequentially ===' % num)
    modified_adj = adj
    for target_node in tqdm(node_list):
        n_perturbations = 5
        model = Nettack(surrogate, nnodes=modified_adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
    
    root = './pertubed_data'
    name = '{}_nettack_adj_{}'.format(args.dataset, args.ptb_rate)
    name = name + '.npz'

    if type(modified_adj) is torch.Tensor:
        modified_adj = to_scipy(modified_adj)
    if sp.issparse(modified_adj):
        modified_adj = modified_adj.tocsr()
    sp.save_npz(os.path.join(root, name), modified_adj)

    nodes = {"idx_train":idx_train.tolist(), "attacked_test_nodes":node_list.tolist()}
    with open(f'./pertubed_data/{args.dataset}_nettacked_nodes.json', 'w') as f:
        json.dump(nodes, f)

if __name__ == '__main__':
    attack_all()
