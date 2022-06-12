import copy
from hashlib import new
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv
from tqdm import trange

from utils import _similarity, knn_fast, recons_loss, add_edges

from torch_geometric.utils import negative_sampling

from matplotlib import pyplot as plt
import seaborn as sns

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
        self.encoder = GCN(self.nfeat, self.nhid, self.nhid, mask=True)
        self.decoder = GCN(self.nhid, self.nhid, self.nfeat, mask=True)
        self.model_s = GCN(self.nfeat, self.nhid, self.n_class)
        self.encoder_to_decoder = nn.Linear(self.nfeat//8, self.nhid, bias=False)

    def init_label_dict(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

        self.training_ratio = labels.shape[0] / idx_train.shape[0]

    def create_fake_nodes(self, num_fakes, x, labels, idx_train):
        if type(num_fakes) == int:
            num_fakes = [num_fakes] * self.n_class
        
        elif type(num_fakes) == list:
            assert self.nclass == len(num_fakes), "the length of 'num_fakes' must equal to nclass"

        x = x[idx_train]
        labels = labels[idx_train]
        x_fake = []
        labels_fake = []

        for c, n_fake in enumerate(num_fakes):
            x_class = x[labels==c]
            w = F.softmax(torch.rand(n_fake, x_class.size(0)), 1).to(self.device)
            x_fake.append(torch.mm(w, x_class))
            labels_fake.append(torch.LongTensor([c]*n_fake))

        return torch.cat(x_fake), torch.cat(labels_fake).to(self.device)

    def recons_loss(self, z, edge_index, stepwise=True):
        # Version 1 MSE
        randn = negative_sampling(edge_index, num_nodes=self.n_real, num_neg_samples=5*self.n_real)
        randn = randn[:,randn[0]<randn[1]]
        
        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        if stepwise:
            edge_index = edge_index[:, self.edge_mask]

        neg0 = z[randn[0]]
        neg1 = z[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = z[edge_index[0]]
        pos1 = z[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') + \
                    F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * self.n_real/(randn.shape[1] + edge_index.shape[1])

        return rec_loss

    def delete_edges(self, real_edge_index, embeddings, threshold=0.2):
        z = embeddings[:self.n_real]
        sim_matrix = _similarity(z)
        edge_weight = sim_matrix[tuple(real_edge_index)]

        edge_mask = edge_weight > threshold

        return real_edge_index[:, edge_mask], edge_weight[edge_mask]

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def mask_feature_loss(self, real_features, pred_features, mask_nodes, loss_type='cos', alpha=3):
        if loss_type == 'mse':
            loss = F.mse_loss(real_features[mask_nodes], pred_features[mask_nodes])
        else:
            x = F.normalize(real_features[mask_nodes], p=2, dim=-1)
            y = F.normalize(pred_features[mask_nodes], p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            loss = loss.mean()
        return loss

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=20):
        real_mask = torch.load('./image/citeseer_real_mask')
        train_mask = torch.LongTensor(idx_train)
        training_labels = labels.clone()
        self.init_label_dict(labels, idx_train)

        optimizer = torch.optim.Adam(list(self.model_s.parameters())+
                                     list(self.decoder.parameters())+
                                     list(self.encoder_to_decoder.parameters())+
                                     list(self.graph_learner.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)

        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        single_mask = real_edge_index[0]<real_edge_index[1]
        self.edge_mask = torch.zeros(single_mask.sum().item()).bool()
        self.n_real = x.size(0)

        fake_x, fake_labels = self.create_fake_nodes(20, x, labels, idx_train)
        self.x = torch.cat([x, fake_x])
        training_labels = torch.cat([training_labels, fake_labels])
        train_mask = torch.cat([train_mask, torch.arange(len(fake_labels))+self.n_real]).to(self.device)

        # train_mask = torch.cat([train_mask, torch.arange(len(idx_train))+self.n_real]).to(self.device)
        # training_labels = torch.cat([training_labels, labels[idx_train]])

        fake_edges_index = []
        for i, label in enumerate(fake_labels):
            same_class = idx_train[labels[idx_train]==label]
            i += self.n_real
            for j in same_class:
                fake_edges_index.append([i, j])
                fake_edges_index.append([j, i])
        fake_edges_index = torch.LongTensor(fake_edges_index).T.to(self.device)
        real_edge_index = torch.cat([real_edge_index, fake_edge_index], -1)

        best_acc = 0
        for i in trange(iteration):
            for epoch in range(epochs):
                optimizer.zero_grad()
                embeddings = self.graph_learner.internal_forward(self.x)
                loss_lp = self.recons_loss(embeddings, real_edge_index)
                # loss_lp = self.recons_loss(embeddings, real_edge_index)

                # embeddings = self.encoder_to_decoder(embeddings)
                # reconst = self.decoder.get_embeds(embeddings, real_edge_index, real_edge_weight, mask_rate=0.5)
                # loss_mask = self.mask_feature_loss(x, reconst, self.decoder.mask_nodes, loss_type='mse')

                # reconst = self.decoder.get_embeds(embeddings, real_edge_index, real_edge_weight, training=False)
                fake_edge_index, fake_edge_weight = knn_fast(embeddings, self.k, 1000, self.device)

                fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                              training_labels, train_mask, mode='threshold')

                new_edge_index, new_edge_weight = self.delete_edges(real_edge_index, embeddings) # 選項一 embeddings 選項二 reconst

                # self.edge_index = real_edge_index
                # self.edge_weight = real_edge_weight
                self.edge_index = torch.cat([new_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([new_edge_weight, fake_edge_weight])

                s_pred, loss_s = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight), 
                                                         training_labels, train_mask)

                loss = loss_s + loss_lp
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    s_pred = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight))
                    accs = []
                    logits = s_pred
                    for mask in [train_mask[train_mask<self.n_real], idx_val, idx_test]:
                        pred = logits[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                    if accs[1] > best_acc:
                        best_acc = accs[1]
                        best_test_acc = accs[2]
                        best_model_s_wts = copy.deepcopy(self.model_s.state_dict())
                        best_model_e_wts = copy.deepcopy(self.encoder.state_dict())
                    
                    print(accs, train_mask.shape[0], real_edge_index.shape[1], self.edge_index.shape[1], loss.item(), best_acc, best_test_acc)
                    
            try:
                fake_adj = _similarity(embeddings)
                plt.figure()
                sns.histplot(data=fake_adj[tuple(real_edge_index[:, real_mask])].detach().cpu(), bins=30, color='skyblue', stat='count')
                sns.histplot(data=fake_adj[tuple(real_edge_index[:, ~real_mask])].detach().cpu(), bins=30, color='red', stat='count', alpha=0.6)
                # a = torch.isin(torch.arange(self.n_real), torch.unique(self.edge_index).cpu())
                # disconnect = torch.arange(self.n_real)[~a]
                # connect = torch.arange(self.n_real)[a]
                # if len(disconnect) != 0:
                #     acc_disconnect = (logits[disconnect].max(1)[1]).eq(labels[disconnect]).sum().item() / len(disconnect)
                # else:
                #     acc_disconnect = 0
                # acc_connect = (logits[connect].max(1)[1]).eq(labels[connect]).sum().item() / len(connect)
                # plt.title(f'{len(disconnect)}, {acc_disconnect*100:.2f}%, {len(connect)} {acc_connect*100:.2f}, {torch.isin(disconnect, train_mask.cpu()).sum().item()}\n\
                #             {(torch.isin(real_edge_index[0, fake_adj[tuple(real_edge_index)]<0.2].cpu(), disconnect)+torch.isin(real_edge_index[1, fake_adj[tuple(real_edge_index)]<0.2].cpu(), disconnect)).sum().item()}')
                plt.savefig(f'./image/testplt_{i}.jpg')
            except:
                plt.close()

            add_nodes_s, pseudo_labels_s = self.add_nodes(train_mask)
            training_labels[add_nodes_s] = pseudo_labels_s

            pseudo_nodes = add_nodes_s
            train_mask = torch.cat([train_mask, pseudo_nodes])

            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

            sim_mat = _similarity(x)
            edge_weight = sim_mat[tuple(real_edge_index[:, single_mask])]
            unlabel_mask = torch.where(self.edge_mask == False)[0]
            try:
                _, train_edges = edge_weight[unlabel_mask].topk(150)

                # self.edge_mask[unlabel_mask[add_edges]] = True
                for idx in unlabel_mask[train_edges]:
                    if edge_weight[idx] > 0.2:
                        self.edge_mask[idx] = True
            except:
                pass

        self.restore_all(x, best_model_s_wts, best_model_e_wts, real_edge_index, real_edge_weight, training_labels, train_mask, idx_train)

    def restore_all(self,x,  model_s_wts, model_e_wts, real_edge_index, real_edge_weight, training_labels, train_mask, idx_train):
        
        self.model_s.load_state_dict(model_s_wts)
        self.encoder.load_state_dict(model_e_wts)

        self.encoder.eval()
        self.model_s.eval()

        embeddings = self.encoder.get_embeds(x, real_edge_index, real_edge_weight, training=False)
        fake_edge_index, fake_edge_weight = knn_fast(torch.cat([embeddings, embeddings[idx_train]]), self.k, 1000, self.device)

        fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                      training_labels, train_mask, mode='mix')
        
        self.edge_index = real_edge_index
        self.edge_weight = real_edge_weight
        # self.edge_index = torch.cat([real_edge_index, fake_edge_index], -1)
        # self.edge_weight = torch.cat([real_edge_weight, fake_edge_weight])

    def add_nodes(self, train_mask, n=100):
        mask = torch.isin(torch.arange(self.n_real).to(self.device), train_mask)
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
    def __init__(self, in_dim, hid_dim, out_dim, mask=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        if mask:
            self.mask_feature = torch.nn.Parameter(torch.zeros(1,in_dim))
            self.mask_nodes = None

    def forward(self, x, edge_index, edge_weight=None, T=0.2):
        x = self.get_embeds(x, edge_index, edge_weight)

        return F.log_softmax(x/T, dim=1)

    def get_embeds(self, x, edge_index, edge_weight=None, mask_rate=None, training=True):
        if mask_rate!=None or self.mask_nodes!=None:
            if mask_rate:
                num_nodes = x.size(0)
                perm = torch.randperm(num_nodes, device=x.device)
                num_mask_nodes = int(mask_rate * num_nodes)

                # random masking
                num_mask_nodes = int(mask_rate * num_nodes)
                self.mask_nodes = perm[:num_mask_nodes]
                out_x = x.clone()
            else:
                out_x = x

            if training:
                out_x[self.mask_nodes] = 0

            # if mask_rate:
            #     out_x[self.mask_nodes] += self.mask_feature
            
        else:
            out_x = x
            
        x = F.relu(self.conv1(out_x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        return self.conv2(x, edge_index, edge_weight)

class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k):
        super().__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize//4))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize//4, isize//8))

        self.input_dim = isize
        self.k = k
        # self.param_init()

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