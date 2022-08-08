import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from tqdm import trange

class CoG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.001, weight_decay=5e-4, device=None) -> None:
        super().__init__()

        self.n_class = nclass
        self.nfeat = nfeat

        self.model_s = GCN(nfeat, nhid, nclass)
        self.model_f = MLP(nfeat, nhid, nclass)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.pseudo_nodes_list = []

    def init_label_ratio(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)  
    
    def forward_classifier(self, model, x, labels=None, mask=None, train=True):
        if train:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])
        else:
            model.eval()
            logit = model(*x)
            return logit

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=10):
        self.x = x
        self.edge_index = adj.nonzero().T
        self.edge_weight = adj[tuple(self.edge_index)]
        train_mask = torch.LongTensor(idx_train)
        training_labels = labels.clone()

        self.init_label_ratio(labels, idx_train)

        optimizer = torch.optim.Adam(list(self.model_f.parameters())+\
                                     list(self.model_s.parameters()), 
                                     lr=self.lr, weight_decay=self.weight_decay)
        best_acc = 0
        for i in trange(iteration):
            for epoch in range(epochs):
                optimizer.zero_grad()
                f_pred, loss_f = self.forward_classifier(self.model_f, (x,), training_labels, train_mask, train=True)
                s_pred, loss_s = self.forward_classifier(self.model_s, (x, self.edge_index, self.edge_weight), training_labels, train_mask, train=True)

                loss = loss_f+loss_s
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    f_pred = self.forward_classifier(self.model_f, (x,), labels, train_mask, train=False)
                    s_pred = self.forward_classifier(self.model_s, (x, self.edge_index), labels, train_mask, train=False)
                    
                    accs = []
                    logits = f_pred+s_pred
                    for phase, mask in enumerate([train_mask, idx_val, idx_test]):
                        pred = logits[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                    if accs[1] > best_acc:
                        best_acc = accs[1]
                        best_model_f_wts = copy.deepcopy(self.model_f.state_dict())
                        best_model_s_wts = copy.deepcopy(self.model_s.state_dict())

                    # print(accs, train_mask.shape[0], best_acc)
                
            add_nodes_f, add_nodes_s, pseudo_labels_f, pseudo_labels_s = self.add_nodes(x, self.edge_index, labels, train_mask)
            training_labels[add_nodes_f] = pseudo_labels_f
            training_labels[add_nodes_s] = pseudo_labels_s

            if sum(torch.isin(add_nodes_f, add_nodes_s)) > 0:
                same_nodes = np.intersect1d(add_nodes_f.cpu(), add_nodes_s.cpu())
                for node in same_nodes:
                    f_idx = f_pred[node].max(-1)
                    s_idx = s_pred[node].max(-1)
                    if f_idx[0] > s_idx[0]:
                        training_labels[node] = f_idx[1]
                    else:
                        training_labels[node] = s_idx[1]

            pseudo_nodes = torch.unique(torch.cat([add_nodes_f, add_nodes_s]))
            train_mask = torch.cat([train_mask, pseudo_nodes])

            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

        self.model_f.load_state_dict(best_model_f_wts)
        self.model_s.load_state_dict(best_model_s_wts)

    def add_nodes(self, x, edge_index, labels, train_mask, n=15):
        mask = torch.isin(torch.arange(x.shape[0]), train_mask)
        unlabel_nodes = torch.where(~mask)[0]

        new_nodes_f = []
        new_nodes_s = []

        new_labels_f = []
        new_labels_s = []

        f_pred = self.forward_classifier(self.model_f, (x,), labels, train_mask, train=False)
        s_pred = self.forward_classifier(self.model_s, (x, edge_index), labels, train_mask, train=False)
        unlabel_logit_f, unlabel_pseudo_f = f_pred[unlabel_nodes].max(-1)
        unlabel_logit_s, unlabel_pseudo_s = s_pred[unlabel_nodes].max(-1)

        for c, r in self.label_ratio.items():
            n_class = int(r*n)

            idx_class_f = torch.where(unlabel_pseudo_f == c)[0]
            idx_class_s = torch.where(unlabel_pseudo_s == c)[0]

            if len(idx_class_f) < n_class:
                f_idx = idx_class_f[unlabel_logit_f[idx_class_f].topk(len(idx_class_f))[1]]

            else:     
                f_idx = idx_class_f[unlabel_logit_f[idx_class_f].topk(n_class)[1]]

            if len(idx_class_s) < n_class:
                s_idx = idx_class_s[unlabel_logit_s[idx_class_s].topk(len(idx_class_s))[1]]
            else:
                s_idx = idx_class_s[unlabel_logit_s[idx_class_s].topk(n_class)[1]]

            new_nodes_f.append(unlabel_nodes[f_idx])
            new_nodes_s.append(unlabel_nodes[s_idx])

            new_labels_f.append(unlabel_pseudo_f[f_idx])
            new_labels_s.append(unlabel_pseudo_s[s_idx])

        new_nodes_f = torch.cat(new_nodes_f)
        new_nodes_s = torch.cat(new_nodes_s)
        new_labels_f = torch.cat(new_labels_f)
        new_labels_s = torch.cat(new_labels_s)

        return new_nodes_f, new_nodes_s, new_labels_f, new_labels_s
        
    def forward(self, x, adj):
        if x == None and adj == None:
            x = self.x
            edge_index = self.edge_index
            edge_weight = self.edge_weight
        else:
            edge_index = adj.nonzero().T
            edge_weight = adj[tuple(edge_index)]

        f_pred = self.forward_classifier(self.model_f, (x,), labels=None, mask=None, train=False)
        s_pred = self.forward_classifier(self.model_s, (x, edge_index, edge_weight), labels=None, mask=None, train=False)
        
        return (f_pred+s_pred)/2

    def predict(self, x=None, adj=None):
        return self.forward(x, adj)

    def get_embed(self, x, adj):
        edge_index = adj.nonzero().T
        edge_weight = adj[tuple(edge_index)]
        self.model_f.eval()
        self.model_s.eval()
        f_embeds = self.model_f.get_embeds(x)
        s_embeds = self.model_s.get_embeds(x, edge_index, edge_weight)
        
        return (f_embeds+s_embeds)/2

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

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.layers = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(hid_dim, out_dim))

    def forward(self, x, T=0.2):
        x = self.get_embeds(x)
        return F.log_softmax(x/T, dim=1)

    def get_embeds(self, x):
        return self.layers(x)