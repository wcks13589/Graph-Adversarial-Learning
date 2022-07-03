import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import trange

from utils import _similarity, knn_fast, recons_loss, add_edges

from torch_geometric.utils import negative_sampling, degree, structured_negative_sampling

from matplotlib import pyplot as plt
import seaborn as sns

from torch_geometric.utils import k_hop_subgraph, to_undirected

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
        
        self.k = 100
        self.init_models()
        
    def init_models(self):

        # self.graph_learner = MLP_learner(2, self.nfeat, self.k)
        self.graph_learner = MLP(self.nfeat, self.nfeat, self.nfeat)
        self.decoder = GCN(self.nhid, self.nhid, self.nfeat, self.nhid, mask=True)
        self.model_s = GCN(self.nfeat, self.nhid, self.nhid, self.n_class, mask=True)
        self.class_embedding = nn.Embedding(self.n_class, self.nfeat)

    def init_label_dict(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

        self.training_ratio = labels.shape[0] / idx_train.shape[0]

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
    
    def recons_loss_2(self, z, edge_index, edge_mask, stepwise=True):
        # Version 1 MSE
        # randn = negative_sampling(edge_index, num_nodes=self.n_real, num_neg_samples=5*self.n_real)
        edge_0, edge_1, edge_k = structured_negative_sampling(edge_index, num_nodes=self.n_real)
        # perm = torch.randperm(self.neg_edge_index.shape[1], device=self.device)
        # num_mask_nodes = perm[:5*self.n_real]
        # randn = self.neg_edge_index[:, num_mask_nodes]
        randn = torch.stack([edge_0, edge_k])
        randn = randn[:,randn[0]<randn[1]]
        
        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        if stepwise:
            edge_index = edge_index[:, edge_mask]

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

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def mask_feature_loss(self, real_features, pred_features, loss_type='cos', alpha=3):
        if loss_type == 'mse':
            loss = F.mse_loss(real_features, pred_features)
        else:
            x = F.normalize(real_features, p=2, dim=-1)
            y = F.normalize(pred_features, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            loss = loss.mean()
        return loss

    def create_fake_nodes(self, num_fakes, x, labels, idx_train):
        if type(num_fakes) == int:
            num_fakes = [num_fakes] * self.n_class
        
        elif type(num_fakes) == list:
            assert self.nclass == len(num_fakes), "the length of 'num_fakes' must equal to nclass"

        x = x[idx_train]
        labels = labels[idx_train].cpu()
        x_fake = []
        labels_fake = []
        edge_fake = []

        i = self.n_real
        for c, n_fake in enumerate(num_fakes):
            x_class = x[labels==c]
            w = F.softmax(torch.rand(n_fake, x_class.size(0)), 1).to(self.device)
            x_fake.append(torch.mm(w, x_class))
            labels_fake.append(torch.LongTensor([c]*n_fake))

            for node in idx_train[labels==c]:
                edge_fake.append([i, node])
                edge_fake.append([node, i])
                i += 1

        return torch.cat(x_fake), torch.cat(labels_fake).to(self.device), torch.LongTensor(edge_fake).T.to(self.device)

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=25):
        # real_mask = torch.load('./image/cora_real_mask_0.2')
        train_mask = torch.LongTensor(idx_train).to(self.device)
        training_labels = labels.clone()
        self.init_label_dict(labels, idx_train)

        self.n_real = x.size(0)
        
        optimizer = torch.optim.Adam(list(self.model_s.parameters())+
                                     list(self.decoder.parameters())+
                                     list(self.graph_learner.parameters())+
                                     list(self.class_embedding.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)

        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        single_mask = real_edge_index[0]<real_edge_index[1]
        self.edge_mask = torch.zeros(single_mask.sum().item()).bool()

        self.x = x
        # fake_x, fake_labels, fake_edge_index_0 = self.create_fake_nodes(10, x, labels, idx_train)
        fake_labels = torch.arange(self.n_class).to(self.device)
        train_mask = torch.cat([train_mask, fake_labels+self.n_real]) # torch.cat([train_mask, train_mask+self.n_real]) 
        training_labels = torch.cat([training_labels, fake_labels]) # torch.cat([training_labels, training_labels])

        # label_dict = {}
        # for i in range(self.n_class):
        #     label_dict[i] = torch.arange(len(fake_x)).to(self.device)[fake_labels == i]+self.n_real
            # label_dict[i] = torch.LongTensor(idx_train).to(self.device)[labels[idx_train] == i]

        # self.neg_edge_index = self.negative_sampling(real_edge_index, num_nodes=self.n_real, k=3)

        best_acc = 0
        for i in trange(iteration):
            for epoch in range(200):
                optimizer.zero_grad()
                self.graph_learner.train()
                fake_nodes = self.class_embedding(fake_labels)
                self.x = torch.cat([x, fake_nodes])

                # embeddings = self.graph_learner.get_embeds(self.x)
                # loss_lp = self.recons_loss(embeddings[:self.n_real], real_edge_index, stepwise=False)

                fake_edge_index, fake_edge_weight = knn_fast(self.x, self.k, 1000, self.device)
                fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                            training_labels, train_mask, self.n_real, mode='threshold', threshold=0.7)

                self.edge_index = real_edge_index # torch.cat([real_edge_index, fake_edge_index], -1)
                self.edge_weight = real_edge_weight # torch.cat([real_edge_weight, fake_edge_weight])

                s_pred, loss_s = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight), 
                                                         training_labels, train_mask)
                loss = loss_s # + loss_lp
                loss.backward()
                optimizer.step()

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
                    best_model_g_wts = copy.deepcopy(self.graph_learner.state_dict())
                    best_edge_index = self.edge_index.clone()
                    best_edge_weight = self.edge_weight.clone()

                if epoch % 20 == 0:
                   print(accs, train_mask.shape[0], self.edge_index.shape[1], real_edge_index.shape[1], loss.item(), best_acc, best_test_acc)

            # mask = train_mask[train_mask<self.n_real]
            # pred = logits[mask].max(1)[1]
            # misclass_nodes = torch.where(pred != labels[mask])[0]
            
            # same = []
            # diff = []
            # if misclass_nodes.size(0) > 0 :
            #     for misclass_node in misclass_nodes:
            #         fake_adj = _similarity(z[label_dict[pred[misclass_node].item()]], z[misclass_node].unsqueeze(0))
            #         diff.append(fake_adj.detach().cpu().flatten().mean().item())

            #         fake_adj = _similarity(z[label_dict[labels[misclass_node].item()]], z[misclass_node].unsqueeze(0))
            #         same.append(fake_adj.detach().cpu().flatten().mean().item())

            # plt.figure()
            # sns.histplot(data=same, bins=30, color='skyblue', stat='count')
            # sns.histplot(data=diff, bins=30, color='red', stat='count', alpha=0.6)

            # plt.savefig(f'./image/testplt_{i}.jpg')

            # fake_adj = _similarity(embeddings)
            # plt.figure()
            # sns.histplot(data=fake_adj[tuple(real_edge_index[:, real_mask])].detach().cpu(), bins=30, color='skyblue', stat='count')
            # sns.histplot(data=fake_adj[tuple(real_edge_index[:, ~real_mask])].detach().cpu(), bins=30, color='red', stat='count', alpha=0.6)
            # a = torch.isin(torch.arange(self.n_real), torch.unique(new_edge_index).cpu())
            # disconnect = torch.arange(self.n_real)[~a]
            # connect = torch.arange(self.n_real)[a]
            # if len(disconnect) != 0:
            #     acc_disconnect = (logits[disconnect].max(1)[1]).eq(labels[disconnect]).sum().item() / len(disconnect)
            # else:
            #     acc_disconnect = 0
            # acc_connect = (logits[connect].max(1)[1]).eq(labels[connect]).sum().item() / len(connect)
            # plt.title(f'{len(disconnect)}, {acc_disconnect*100:.2f}%, {len(connect)} {acc_connect*100:.2f}, {torch.isin(disconnect, train_mask.cpu()).sum().item()}\n\
            #             {(torch.isin(real_edge_index[0, fake_adj[tuple(real_edge_index)]<0.2].cpu(), disconnect)+torch.isin(real_edge_index[1, fake_adj[tuple(real_edge_index)]<0.2].cpu(), disconnect)).sum().item()}')
            # plt.savefig(f'./image/testplt_{i}.jpg')

            # except:
            #     plt.close()

            # update pseudo label
            # for node in self.pseudo_nodes_list:
            #     s_idx = s_pred[node].max(-1)
            #     training_labels[node] = s_idx[1]

            add_nodes_s, pseudo_labels_s = self.add_nodes(train_mask)
            training_labels[add_nodes_s] = pseudo_labels_s

            pseudo_nodes = add_nodes_s
            train_mask = torch.cat([train_mask, pseudo_nodes])

            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

            # sim_mat = _similarity(x)
            # edge_weight = sim_mat[tuple(real_edge_index[:, single_mask])]
            # unlabel_mask = torch.where(self.edge_mask == False)[0]
            # try:
            #     _, train_edges = edge_weight[unlabel_mask].topk(150)

            #     # self.edge_mask[unlabel_mask[add_edges]] = True
            #     for idx in unlabel_mask[train_edges]:
            #         if edge_weight[idx] > 0.2:
            #             self.edge_mask[idx] = True
            # except:
            #     pass

        # from sklearn.manifold import TSNE
        # colors = [
        #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        #     '#ffd700'
        # ]
        # embeddings = self.model_s.get_embeds(x, real_edge_index)
        # z = TSNE(n_components=2).fit_transform(embeddings.detach().cpu().numpy())
        # plt.figure(figsize=(8, 8))
        # for i in range(self.n_class):
        #     plt.scatter(z[labels.cpu() == i, 0], z[labels.cpu() == i, 1], s=20, color=colors[i])
        # plt.savefig(f'./image/tsne.png')
        # torch.save(train_acc, './image/train_acc_mask_update')

        self.restore_all(x, best_model_s_wts, best_model_g_wts, best_edge_index, best_edge_weight)

    def restore_all(self, x, model_s_wts, model_g_wts, edge_index, edge_weight):
        
        self.model_s.load_state_dict(model_s_wts)
        self.graph_learner.load_state_dict(model_g_wts)

        self.graph_learner.eval()
        self.model_s.eval()

        self.edge_index = edge_index
        self.edge_weight = edge_weight

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

    def negative_sampling(self, edge_index, num_nodes, k, num_neg_samples=None):
        neg_edge_index = []
        for node in range(num_nodes):
            one_hop_nodes = edge_index[1][edge_index[0] == node]
            k_hop_nodes = k_hop_subgraph(node, k, edge_index, num_nodes=num_nodes)[0]
            mask = torch.isin(k_hop_nodes, one_hop_nodes)

            edge_1 = k_hop_nodes[~mask]
            edge_0 = torch.ones_like(edge_1) * node

            neg_edge_index.append(torch.stack([edge_0, edge_1]))

        neg_edge_index = torch.cat(neg_edge_index, -1)

        return to_undirected(neg_edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hid_dim)
        self.conv2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, T=1):
        x = self.get_embeds(x)

        return F.log_softmax(x/T, dim=1)

    def get_embeds(self, x):
        x = F.relu(self.conv1(x))

        return self.conv2(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, hid_dim1, out_dim, mask=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim1)
        self.output = nn.Linear(hid_dim1, out_dim)
        if mask:
            self.mask_feature = torch.nn.Parameter(torch.zeros(1,in_dim))

    def forward(self, x, edge_index, edge_weight=None, T=0.2):
        x = self.get_embeds(x, edge_index, edge_weight)
        x = self.output(x)

        return F.log_softmax(x/T, dim=1)

    def process_mask(self, x, mask_rate, mask_nodes):
        if mask_rate!=None or mask_nodes!=None:
            if mask_rate:
                num_nodes = x.size(0)
                if mask_nodes != None:
                    all_nodes = torch.arange(num_nodes, device=x.device)
                    all_mask = torch.isin(all_nodes, mask_nodes)
                    num_nodes -= mask_nodes.size(0)
                    perm = torch.randperm(num_nodes, device=x.device)
                    num_mask_nodes = int(mask_rate * num_nodes)
                    mask_nodes = torch.cat([mask_nodes, all_nodes[~all_mask][perm[:num_mask_nodes]]])
                else:
                    perm = torch.randperm(num_nodes, device=x.device)
                    num_mask_nodes = int(mask_rate * num_nodes)
                    mask_nodes = perm[:num_mask_nodes]
                out_x = x.clone()
                self.mask_nodes = mask_nodes
            else:
                out_x = x

            if mask_rate!=None or mask_nodes!=None:
                out_x[mask_nodes] = 0

            if mask_rate:
                out_x[mask_nodes] += self.mask_feature
        else:
            out_x = x

        return out_x

    def get_embeds(self, x, edge_index, edge_weight=None, mask_rate=None, mask_nodes=None):
        x = self.process_mask(x, mask_rate, mask_nodes)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

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

    def get_embeds(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        embeddings = self.get_embeds(features)
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