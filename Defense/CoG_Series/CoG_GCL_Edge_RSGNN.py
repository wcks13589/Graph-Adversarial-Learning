import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from tqdm import trange

from utils import _similarity, knn_fast, add_edges

class CoG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None) -> None:
        super().__init__()

        self.n_class = nclass
        self.nhid = nhid
        self.nfeat = nfeat

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.pseudo_nodes_list = []
        
        self.k = 20
        self.init_models()

    def init_models(self):

        self.graph_learner = MLP_learner(2, self.nfeat, self.k)
        self.gcl = GCL(self.nfeat, 512, 128)
        self.model_s = GCN(self.nfeat, self.nhid, self.n_class)

    def init_label_ratio(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def delete_edges(self, real_edge_index, embeddings):
        z = embeddings[:self.n_real]
        sim_matrix = _similarity(z, z)
        edge_weight = sim_matrix[tuple(real_edge_index)]

        edge_mask = edge_weight > 0.1

        return real_edge_index[:, edge_mask], edge_weight[edge_mask]

    def mask_feature(self, x, drop_prob):
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0

        return x

    def contrast(self, real_edge_index, real_edge_weight, fake_edge_index, fake_edge_weight):

        z1 = self.gcl(self.mask_feature(self.x, 0.2), fake_edge_index, fake_edge_weight)[:self.n_real]
        # edge_index  = torch.LongTensor([[i,i] for i in range(self.n_real)]).T.cuda()
        z2 = self.gcl(self.mask_feature(self.x[:self.n_real], 0.2), real_edge_index, real_edge_weight)

        return self.gcl.calc_loss(z1, z2)

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=20):

        train_mask = torch.LongTensor(idx_train)
        training_labels = labels.clone()

        self.init_label_ratio(labels, idx_train)
        
        optimizer = torch.optim.Adam(list(self.model_s.parameters())+
                                     list(self.graph_learner.parameters())+
                                     list(self.gcl.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)
        self.n_real = x.shape[0]
        self.x = torch.cat([x, x[idx_train]])
        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        train_mask = torch.cat([train_mask, torch.arange(len(idx_train))+self.n_real])
        training_labels = torch.cat([training_labels, labels[idx_train]])

        self.edge_index = real_edge_index
        self.edge_weight = adj[tuple(real_edge_index)]

        best_acc = 0
        for i in trange(iteration):
            for epoch in range(epochs):
                optimizer.zero_grad()

                fake_edge_index, fake_edge_weight, embeddings = self.graph_learner(self.x)

                loss = self.contrast(real_edge_index, real_edge_weight, fake_edge_index, fake_edge_weight)*0.5
                
                fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                              training_labels, train_mask, mode='class')

                new_real_edge_index,  new_edge_weight = self.delete_edges(real_edge_index, embeddings)
                
                labels_ = torch.cat([labels, labels[idx_train]]) # ?????????
                T = labels_[fake_edge_index[0]] == labels_[fake_edge_index[1]] # ?????????
                F = labels_[fake_edge_index[0]] != labels_[fake_edge_index[1]] # ?????????

                self.edge_index = torch.cat([new_real_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([new_edge_weight, fake_edge_weight])
                
                s_pred, loss_s = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight), 
                                                       training_labels, train_mask)

                loss += loss_s
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    s_pred = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight))
                    accs = []
                    logits = s_pred
                    for phase, mask in enumerate([idx_train, idx_test, idx_val]):
                        pred = logits[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                        if phase == 2 and acc > best_acc:
                            best_acc = acc
                            best_test_acc = accs[1]
                            best_model_s_wts = copy.deepcopy(self.model_s.state_dict())
                            best_model_g_wts = copy.deepcopy(self.graph_learner.state_dict())
                    
                    print(accs, self.edge_weight.shape[0], new_real_edge_index.shape[1], T.sum().item(), F.sum().item(), best_acc, best_test_acc)
            
            # # update pseudo label
            # for node in self.pseudo_nodes_list:
            #     f_idx = f_pred[node].max(-1)
            #     s_idx = s_pred[node].max(-1)
            #     if f_idx[0] > s_idx[0]:
            #         training_labels[node] = f_idx[1]
            #         training_labels_with_fake[node] = f_idx[1]
            #     else:
            #         training_labels[node] = s_idx[1]
            #         training_labels_with_fake[node] = s_idx[1]
            
            add_nodes_s, pseudo_labels_s = self.add_nodes(train_mask)
            training_labels[add_nodes_s] = pseudo_labels_s

            pseudo_nodes = add_nodes_s
            train_mask = torch.cat([train_mask, pseudo_nodes])

            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

        self.restore_all(best_model_s_wts, best_model_g_wts, real_edge_index, real_edge_weight, training_labels, train_mask)

    def restore_all(self, model_s_wts, model_g_wts, real_edge_index, real_edge_weight, training_labels, train_mask):
        
        self.model_s.load_state_dict(model_s_wts)
        self.graph_learner.load_state_dict(model_g_wts)

        self.graph_learner.eval()
        self.model_s.eval()
        
        fake_edge_index, fake_edge_weight, embeddings = self.graph_learner(self.x)
        fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                      training_labels, train_mask, mode='class')

        new_real_edge_index,  new_edge_weight = self.delete_edges(real_edge_index, embeddings)
        self.edge_index = torch.cat([new_real_edge_index, fake_edge_index], -1)
        self.edge_weight = torch.cat([new_edge_weight, fake_edge_weight])

    def add_nodes(self, train_mask, n=100):
        mask = torch.isin(torch.arange(self.n_real), train_mask)
        unlabel_nodes = torch.where(~mask)[0]

        new_nodes_s = []
        new_labels_s = []

        s_pred = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight))
        unlabel_logit_s, unlabel_pseudo_s = s_pred[unlabel_nodes].max(-1)

        for c, r in self.label_ratio.items():
            n_class = int(r*n)
            idx_class_s = torch.where(unlabel_pseudo_s == c)[0]

            if len(idx_class_s) < n_class:
                s_idx = idx_class_s[unlabel_logit_s[idx_class_s].topk(len(idx_class_s))[1]]
            else:
                s_idx = idx_class_s[unlabel_logit_s[idx_class_s].topk(n_class)[1]]

            new_nodes_s.append(unlabel_nodes[s_idx])
            new_labels_s.append(unlabel_pseudo_s[s_idx])

        new_nodes_s = torch.cat(new_nodes_s)
        new_labels_s = torch.cat(new_labels_s)

        return new_nodes_s, new_labels_s
        
    def forward(self, x, adj):
        if x == None and adj == None:
            x = self.x
            edge_index = self.edge_index
            edge_weight = self.edge_weight
        else:
            edge_index = adj.nonzero().T
            edge_weight = adj[tuple(edge_index)]

        s_pred = self.forward_classifier(self.model_s, (x, edge_index, edge_weight))

        return s_pred

    def predict(self, x=None, adj=None):
        return self.forward(x, adj)

    def get_embed(self, x, adj):
        # edge_index = adj.nonzero().T
        # edge_weight = adj[tuple(edge_index)]
        self.model_s.eval()
        s_embeds = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight)
        
        return s_embeds

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None, T=0.2):
        x = self.get_embeds(x, edge_index, edge_weight)

        return F.log_softmax(x/T, dim=1)

    def get_embeds(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        return self.conv2(x, edge_index, edge_weight)

class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k):
        super().__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.k = k
        self.param_init()

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        embeddings = self.internal_forward(features)
        edge_index, edge_weight = knn_fast(embeddings, self.k, 1000, device=embeddings.device)

        return edge_index, edge_weight, embeddings

class GCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim)
        self.proj_head = nn.Linear(out_dim, out_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight != None:
            edge_weight = F.dropout(edge_weight, training=self.training)
        x = self.encoder.get_embeds(x, edge_index, edge_weight)
        x = self.proj_head(x)

        return x

    def calc_loss(self, x, x_aug, temperature=0.2, sym=True):
        batch_size = x.shape[0]
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1