import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils

class RSGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=1e-3, lr_adj=1e-3, weight_decay=5e-4, threshold=0.8, alpha=3, beta=0.3, device=None) -> None:
        super().__init__()

        self.n_class = nclass
        self.nhid = nhid
        self.nfeat = nfeat

        self.lr = lr
        self.lr_adj = lr_adj
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

        self.best_graph = None
        self.weights = None

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=1000, outer_steps=1, inner_steps=2):
        edge_index = adj.nonzero().T
        self.x = x

        self.estimator = EstimateAdj(edge_index, x, hid_dim=self.nhid, device=self.device).to(self.device)
        self.model = GCN(self.nfeat, self.nhid, self.n_class).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_adj = torch.optim.Adam(self.estimator.parameters(), lr=self.lr_adj, weight_decay=self.weight_decay)

        best_val_acc = 0
        for epoch in range(epochs):
            for i in range(int(outer_steps)):
                optimizer_adj.zero_grad()
                rec_loss = self.estimator(edge_index, x)
                output = self.model(x, self.estimator.estimated_index, self.estimator.estimated_weights)
                loss_gcn = F.cross_entropy(output[idx_train], labels[idx_train])

                loss_label_smooth = self.label_smoothing(self.estimator.estimated_index,
                                                         self.estimator.estimated_weights.detach(),
                                                         output, idx_train, self.threshold)
                total_loss = loss_gcn + self.alpha * rec_loss

                total_loss.backward()
                optimizer_adj.step()


                self.model.eval()
                output = self.model(x, self.estimator.estimated_index, self.estimator.estimated_weights.detach())
                loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
                
                if epoch % 20 == 0:
                    accs = []
                    for phase, mask in enumerate([idx_train, idx_val, idx_test]):
                        pred = output[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                    if accs[1] > best_val_acc:
                        best_val_acc = accs[1]
                        self.best_graph = self.estimator.estimated_weights.detach()
                        self.best_edge_index = self.estimator.estimated_index
                        self.weights = copy.deepcopy(self.model.state_dict())

                    # print(epoch, accs, best_val_acc, accs[2])

            for i in range(int(inner_steps)):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(x, self.estimator.estimated_index, self.estimator.estimated_weights.detach())

                loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
                loss_label_smooth = self.label_smoothing(self.estimator.estimated_index, 
                                                         self.estimator.estimated_weights.detach(),
                                                         output, idx_train, self.threshold)
                loss = loss_train + self.beta * loss_label_smooth
                loss.backward()
                optimizer.step()

                self.model.eval()
                output = self.model(x, self.estimator.estimated_index, self.estimator.estimated_weights.detach())

                if epoch % 20 == 0:
                    accs = []
                    for phase, mask in enumerate([idx_train, idx_val, idx_test]):
                        pred = output[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                    if accs[1] > best_val_acc:
                        best_val_acc = accs[1]
                        self.best_graph = self.estimator.estimated_weights.detach()
                        self.best_edge_index = self.estimator.estimated_index
                        self.weights = copy.deepcopy(self.model.state_dict())

                    print(epoch, accs, best_val_acc, accs[2])

            self.model.load_state_dict(self.weights)

    def label_smoothing(self, edge_index, edge_weight, representations, idx_train, threshold):

        num_nodes = representations.shape[0]
        n_mask = torch.ones(num_nodes, dtype=torch.bool).to(self.device)
        n_mask[idx_train] = 0

        mask = n_mask[edge_index[0]] \
                & (edge_index[0] < edge_index[1])\
                & (edge_weight >= threshold)\
                | torch.bitwise_not(n_mask)[edge_index[1]]

        unlabeled_edge = edge_index[:,mask]
        unlabeled_weight = edge_weight[mask]

        Y = F.softmax(representations, -1)

        loss_smooth_label = unlabeled_weight@ torch.pow(Y[unlabeled_edge[0]] - Y[unlabeled_edge[1]], 2).sum(dim=1)/num_nodes

        return loss_smooth_label

    def forward(self, x, adj):
        if x == None and adj == None:
            x = self.x
            if self.best_graph is None:
                edge_weight = self.estimator.estimated_weights
                edge_index = self.estimator.estimated_index
            else:
                edge_weight = self.best_graph
                edge_index = self.best_edge_index
        else:
            edge_index = adj.nonzero().T
            edge_weight = adj[tuple(edge_index)]

        output = self.model(x, edge_index, edge_weight)
        
        return output

    def predict(self, x=None, adj=None):
        return self.forward(x, adj)

    def get_embed(self, x, adj):
        # edge_index = adj.nonzero().T
        # edge_weight = adj[tuple(edge_index)]
        self.model.eval()
        s_embeds = self.model.get_embeds(self.x, self.best_edge_index, self.best_graph)
        
        return s_embeds

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.get_embeds(x, edge_index, edge_weight)
        return x

    def get_embeds(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        return self.conv2(x, edge_index, edge_weight)

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, edge_index, features, hid_dim=16, model_name='MLP', device='cuda'):
        super(EstimateAdj, self).__init__()

        self.model_name = model_name
        if self.model_name == 'MLP':
            self.estimator = nn.Sequential(nn.Linear(features.shape[1],hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(hid_dim, hid_dim))
        else:
            self.estimator = GCN(features.shape[1], hid_dim, hid_dim)
        self.device = device
        self.poten_edge_index = self.get_poten_edge(edge_index, features, 100)
        self.features_diff = torch.cdist(features,features,2)
        self.estimated_weights = None

        self.n_n = 100
        self.t_small = 0.1
        self.sigma = 100

    def get_poten_edge(self, edge_index, features, n_p):

        if n_p == 0:
            return edge_index

        poten_edges = []
        for i in range(len(features)):
            sim = torch.div(torch.matmul(features[i],features.T), features[i].norm()*features.norm(dim=1))
            _,indices = sim.topk(n_p)
            poten_edges.append([i,i])
            indices = set(indices.cpu().numpy())
            indices.update(edge_index[1,edge_index[0]==i])
            for j in indices:
                if j > i:
                    pair = [i,j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

        return poten_edges
    

    def forward(self, edge_index, features):

        if self.model_name=='MLP':
            representations = self.estimator(features)
        else:
            representations = self.estimator(features, edge_index)
        rec_loss = self.reconstruct_loss(edge_index, representations)

        x0 = representations[self.poten_edge_index[0]]
        x1 = representations[self.poten_edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        estimated_weights = F.relu(output)
        mask = estimated_weights >= self.t_small
        self.estimated_weights = estimated_weights[mask]
        self.estimated_index = self.poten_edge_index[:, mask]

        return rec_loss
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0],randn[1]]/self.sigma,2)) @ F.mse_loss(neg,torch.zeros_like(neg), reduction='none')
        pos_loss = torch.exp(-torch.pow(self.features_diff[edge_index[0],edge_index[1]]/self.sigma,2)) @ F.mse_loss(pos, torch.ones_like(pos), reduction='none')

        rec_loss = (pos_loss + neg_loss) * num_nodes/(randn.shape[1] + edge_index.shape[1]) 

        return rec_loss