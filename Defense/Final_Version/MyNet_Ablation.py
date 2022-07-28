import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import trange

from utils import _similarity, add_edges, knn_fast
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import negative_sampling

from matplotlib import pyplot as plt
import seaborn as sns

class CoG(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None, verbose=False) -> None:
        super().__init__()

        self.n_class = nclass
        self.nhid = nhid
        self.nfeat = nfeat

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.pseudo_nodes_list = []
        
        self.k = args.k
        self.threshold = args.threshold
        self.add_labels = args.add_labels
        self.iteration = args.iteration
        self.num_fakes = args.fake_nodes
        self.verbose = args.verbose

        self.init_models()
        
    def init_models(self):
        self.graph_learner = MLP(self.nfeat, self.nhid, self.nhid)
        self.model_s = GCN(self.nfeat, self.nhid,  self.nhid, self.n_class, mask=True)
        self.output = nn.Linear(self.nhid, self.n_class)

        self.decoder = GCN(self.nhid, self.nhid, self.nfeat, self.nhid, mask=True)
        self.encoder_to_decoder = nn.Linear(self.nhid, self.nhid)

    def init_label_dict(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

        self.training_ratio = labels.shape[0] / idx_train.shape[0]

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def create_fake_nodes(self, num_fakes, x, labels, idx_train, mode='normalize'):
        if type(num_fakes) == int:
            num_fakes = [num_fakes] * self.n_class
        
        elif type(num_fakes) == list:
            assert self.n_class == len(num_fakes), "the length of 'num_fakes' must equal to n_class"

        x = x[idx_train]
        labels = labels[idx_train].cpu()
        x_fake = []
        labels_fake = []
        edge_fake = []

        i = self.n_real
        for c, n_fake in enumerate(num_fakes):
            x_class = x[labels==c]
            random_weight = torch.rand(n_fake, x_class.size(0)).to(self.device)
            if mode == 'normalize':
                w = F.normalize(random_weight, p=1)
            elif mode == 'softmax':
                w = F.softmax(random_weight, 1)
            x_fake.append(torch.mm(w, x_class))
            labels_fake.append(torch.LongTensor([c]*n_fake))

        x_fake = torch.cat(x_fake)
        label_fake = torch.cat(labels_fake).to(self.device)

        return x_fake, label_fake, torch.arange(x_fake.size(0)).to(self.device)+self.n_real

    def create_mask_nodes(self, n, mask_rate):
        perm = torch.randperm(n)
        if mask_rate > 1:
            num_mask_nodes = mask_rate
        else:
            num_mask_nodes = int(self.n_real * mask_rate)

        return perm[:num_mask_nodes], perm[num_mask_nodes:]

    def sim_loss(self, real_features, pred_features, loss_type='cos', alpha=1):
        if loss_type == 'mse':
            loss = F.mse_loss(real_features, pred_features)
        else:
            x = F.normalize(real_features, p=2, dim=-1)
            y = F.normalize(pred_features, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            loss = loss.mean()
        return loss

    def calc_loss(self, embeddings, train_nodes, train_labels, fake_nodes, fake_labels):
        loss = 0
        for c in range(self.n_class):
            same_class_nodes = train_nodes[train_labels[train_nodes] == c]
            pos_nodes = fake_nodes[fake_labels == c]
            neg_nodes = fake_nodes[fake_labels != c]

            n = neg_nodes.size(0)
            perm = torch.randperm(n)
            if n > pos_nodes.size(0):
                n = pos_nodes.size(0)

            neg_nodes = neg_nodes[perm[:n]]

            pos = _similarity(embeddings[same_class_nodes], embeddings[pos_nodes], normalize=True).flatten()
            loss += F.mse_loss(pos, torch.ones_like(pos), reduction='mean')

            neg = _similarity(embeddings[same_class_nodes], embeddings[neg_nodes], normalize=True).flatten()
            loss += F.mse_loss(neg, torch.zeros_like(neg), reduction='mean')

        return loss

    def mask_loss(self, mask_rate, real_edge_index, reconst_features=False):
       
        mask_nodes, unmask_nodes = self.create_mask_nodes(n=self.n_real, mask_rate=mask_rate)
        # hop_nodes = k_hop_subgraph(mask_nodes, 1, self.edge_index)[0]
        # hop_nodes = hop_nodes[~torch.isin(hop_nodes, mask_nodes.to(self.device))]
        z1 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=mask_nodes, mask_embedding=True)
        if reconst_features:
            self.encoder_to_decoder.train()
            self.decoder.train()
            z1 = self.encoder_to_decoder(z1)
            reconst = self.decoder.get_embeds(z1, real_edge_index, None, mask_nodes=mask_nodes)
            loss = self.sim_loss(self.x[mask_nodes], reconst[mask_nodes])
        else:
            z2 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight)
            loss = self.sim_loss(z1[mask_nodes], z2[mask_nodes])

        return loss

    def recons_loss(self, z, edge_index, stepwise=True):
        # Version 1 MSE
        randn = negative_sampling(edge_index, num_nodes=self.n_real, num_neg_samples=10*self.n_real)
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

        # neg_loss = torch.exp(torch.pow(self.features_diff[randn[0],randn[1]]/1,2)) @ F.mse_loss(neg,torch.zeros_like(neg), reduction='none')
        # pos_loss = torch.exp(-torch.pow(self.features_diff[edge_index[0],edge_index[1]]/100,2)) @ F.mse_loss(pos, torch.ones_like(pos), reduction='none')

        # rec_loss = (pos_loss + neg_loss) \
        #             * self.n_real/(randn.shape[1] + edge_index.shape[1]) 

        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') + \
                    F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * self.n_real/(randn.shape[1] + edge_index.shape[1])
        return rec_loss

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200):
        idx_train = torch.LongTensor(idx_train).to(self.device)
        self.init_label_dict(labels, idx_train)

        self.n_real = x.size(0)
        optimizer = torch.optim.Adam(list(self.model_s.parameters())+
                                     list(self.graph_learner.parameters())+
                                     list(self.output.parameters())+
                                     list(self.decoder.parameters())+
                                     list(self.encoder_to_decoder.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)

        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        
        fake_x, fake_labels, fake_nodes = self.create_fake_nodes(self.num_fakes, x, labels, idx_train)

        train_nodes = idx_train.clone()
        train_labels = labels.clone()

        self.x = torch.cat([x, fake_x])
        train_labels= torch.cat([train_labels, fake_labels])
        train_nodes_with_fake = torch.cat([train_nodes, fake_nodes])
        
        best_acc = 0
        for i in trange(self.iteration):
            for epoch in range(epochs):
                optimizer.zero_grad()
                self.graph_learner.train()
                embeddings = self.graph_learner.get_embeds(self.x)

                mask_nodes, unmask_nodes = self.create_mask_nodes(n=self.n_real, mask_rate=0.2)
                self.encoder_to_decoder.train()
                self.decoder.train()
                z1 = self.encoder_to_decoder(embeddings)
                reconst = self.decoder.get_embeds(z1, real_edge_index, None, mask_nodes=mask_nodes)
                loss_sim = self.sim_loss(self.x[mask_nodes], reconst[mask_nodes], loss_type='cos')

                # loss_sim = self.calc_loss(embeddings, train_nodes, train_labels, fake_nodes ,fake_labels)
                logit = F.log_softmax(self.output(embeddings), -1)
                loss_sim += F.nll_loss(logit[train_nodes_with_fake], train_labels[train_nodes_with_fake])
                # loss_sim += self.recons_loss(embeddings, real_edge_index, stepwise=False) * 0.01

                knn_edge_index, knn_edge_weight = self.knn(embeddings, self.k, 1000, self.device)
                knn_edge_index_1, knn_edge_weight_1 = knn_fast(embeddings, self.k, 1000, self.device)
                knn_edge_index = torch.cat([knn_edge_index, knn_edge_index_1], -1)
                knn_edge_weight = torch.cat([knn_edge_weight, knn_edge_weight_1])
                fake_edge_index, fake_edge_weight, edge_mask = add_edges(knn_edge_index, knn_edge_weight, train_labels, train_nodes, 
                                                                         self.n_real, mode='threshold', threshold=self.threshold)
                
                # print(knn_edge_index.shape, fake_edge_index.shape)

                self.edge_index = torch.cat([real_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([real_edge_weight, fake_edge_weight])

                # self.edge_index = real_edge_index
                # self.edge_weight = real_edge_weight

                # loss_mask = self.mask_loss(0.5, real_edge_index, reconst_features=True)

                s_pred, loss_c = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight), 
                                                         train_labels, train_nodes_with_fake)
                loss = loss_c + loss_sim# + loss_mask * 0.001
                loss.backward()
                optimizer.step()

                s_pred = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight))
                accs = []
                for mask in [train_nodes[train_nodes<self.n_real], idx_val, idx_test]:
                    pred = s_pred[mask].max(1)[1]
                    acc = pred.eq(labels[mask]).sum().item() / len(mask)
                    accs.append(acc)

                if accs[1] > best_acc:
                    best_acc = accs[1]
                    best_test_acc = accs[2]
                    best_model_s_wts = copy.deepcopy(self.model_s.state_dict())
                    best_model_g_wts = copy.deepcopy(self.graph_learner.state_dict())
                    best_edge_index = self.edge_index.clone()
                    best_edge_weight = self.edge_weight.clone()

                if self.verbose:
                    if epoch % 20 == 0:
                        accs = [round(acc, 4) for acc in accs]
                        test_nodes = torch.LongTensor(idx_test).to(self.device)
                        correct_mask = s_pred.max(1)[1][test_nodes] == labels[test_nodes]
                        correct_nodes = test_nodes[correct_mask]
                        mistake_nodes = test_nodes[~correct_mask]
                        correct_mask = torch.isin(fake_edge_index[0], correct_nodes) + torch.isin(fake_edge_index[1], correct_nodes)
                        mistake_mask = torch.isin(fake_edge_index[0], mistake_nodes) + torch.isin(fake_edge_index[1], mistake_nodes)
                        
                        new_labels = torch.cat([labels, fake_labels])
                        new_mask_1 = new_labels[fake_edge_index[0][correct_mask]] == new_labels[fake_edge_index[1][correct_mask]] 
                        new_mask_2 = new_labels[fake_edge_index[0][mistake_mask]] == new_labels[fake_edge_index[1][mistake_mask]] 

                        print(accs, train_nodes.shape[0],
                            f'real:{real_edge_index.shape[1]}',
                            f'total:{self.edge_index.shape[1]}',
                            f'fake:{fake_edge_index.shape[1]}',
                            f'correct:{correct_nodes.size(0)}/{correct_mask.sum().item()}/{new_mask_1.sum().item()}',
                            f'wrong:{mistake_nodes.size(0)}/{mistake_mask.sum().item()}/{new_mask_2.sum().item()}',
                            f'{best_acc:.4f}', f'{best_test_acc:.4f}',
                            f'{loss.item()}')

            pseudo_nodes, pseudo_labels = self.add_nodes(train_nodes, s_pred)
            train_labels[pseudo_nodes] = pseudo_labels

            # pseudo_nodes_mlp, pseudo_labels_mlp = self.add_nodes(train_nodes, logit)
            # train_labels[pseudo_nodes_mlp] = pseudo_labels_mlp

            # if sum(torch.isin(pseudo_nodes, pseudo_nodes_mlp)) > 0:
            #     same_nodes = np.intersect1d(pseudo_nodes.cpu(), pseudo_nodes_mlp.cpu())
            #     for node in same_nodes:
            #         f_idx = logit[node].max(-1)
            #         s_idx = s_pred[node].max(-1)
            #         if f_idx[0] > s_idx[0]:
            #             train_labels[node] = f_idx[1]
            #         else:
            #             train_labels[node] = s_idx[1]

            # pseudo_nodes = torch.unique(torch.cat([pseudo_nodes, pseudo_nodes_mlp]))

            train_nodes = torch.cat([train_nodes, pseudo_nodes])
            train_nodes_with_fake = torch.cat([train_nodes_with_fake, pseudo_nodes])
            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

            if self.verbose:
                fake_adjs = []
                for n in test_nodes:
                    same_nodes = torch.where(train_labels != labels[n])[0]
                    same_nodes = same_nodes[same_nodes>=self.n_real]
                    fake_adjs.append(_similarity(embeddings[n].unsqueeze(0), embeddings[same_nodes]))
                fake_adjs = torch.cat(fake_adjs)
                plt.figure()
                sns.histplot(data=fake_adjs.flatten().detach().cpu(), bins=30, color='red', stat='count', alpha=0.6)

                fake_adjs = []
                for n in test_nodes:
                    same_nodes = torch.where(train_labels == labels[n])[0]
                    same_nodes = same_nodes[same_nodes>=self.n_real]

                    fake_adjs.append(_similarity(embeddings[n].unsqueeze(0), embeddings[same_nodes]))
                fake_adjs = torch.cat(fake_adjs)
                
                sns.histplot(data=fake_adjs.flatten().detach().cpu(), bins=30, color='skyblue', stat='count')

                plt.savefig(f'./image/testplt_101.jpg')
                plt.close('all')

        self.restore_all(x, best_model_s_wts, best_model_g_wts, best_edge_index, best_edge_weight)
        # print('correct labels',(train_labels[train_nodes[train_nodes<self.n_real]] == labels[train_nodes[train_nodes<self.n_real]]).sum()/len(train_nodes[train_nodes<self.n_real]))

    def restore_all(self, x, model_s_wts, model_g_wts, edge_index, edge_weight):
        self.model_s.load_state_dict(model_s_wts)
        self.graph_learner.load_state_dict(model_g_wts)

        self.graph_learner.eval()
        self.model_s.eval()

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
    def add_nodes(self, train_nodes, s_pred):
        n = self.add_labels
        mask = torch.isin(torch.arange(self.n_real).to(self.device), train_nodes)
        unlabel_nodes = torch.where(~mask)[0]

        new_nodes_s = []
        new_labels_s = []

        # s_pred = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight))
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

        # #######
        # hop_nodes = k_hop_subgraph(new_nodes_s, 1, self.edge_index)[0]
        # hop_nodes = hop_nodes[~torch.isin(hop_nodes, new_nodes_s)]

        # embeddings_1 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=hop_nodes, mask_embedding=True)
        # pred_1 = self.model_s.output(embeddings_1)
        # embeddings_2 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=new_nodes_s, mask_embedding=True)
        # pred_2 = self.model_s.output(embeddings_2)
        # mask = pred_1[new_nodes_s].max(1)[1] == pred_2[new_nodes_s].max(1)[1]
        # mask = pred_1[new_nodes_s].max(1)[1] == new_labels_s

        # mlp_logit_s, mlp_pseudo_s = logit[new_nodes_s].max(-1)
        # mask = new_labels_s == mlp_pseudo_s
        
        # return new_nodes_s[mask], new_labels_s[mask]
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

    def knn(self, Z, k, b, device, normalize=True):
        X = Z[:self.n_real] # Real Nodes
        Y = Z[self.n_real:] # Fake Nodes 
        if normalize:
            X = F.normalize(X, dim=1, p=2)
            Y = F.normalize(Y, dim=1, p=2)

        index = 0
        values = torch.zeros(X.shape[0] * (k + 1), device=device)
        rows = torch.zeros(X.shape[0] * (k + 1), device=device)
        cols = torch.zeros(X.shape[0] * (k + 1), device=device)

        while index < X.shape[0]:
            if (index + b) > (X.shape[0]):
                end = X.shape[0]
            else:
                end = index + b
            sub_tensor = X[index:index + b]
            similarities = torch.mm(sub_tensor, Y.t())
            vals, inds = similarities.topk(k=k + 1, dim=-1)
            values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
            cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
            rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
            index += b


        rows = rows.long()
        cols = cols.long() + self.n_real

        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        edge_index = torch.stack([rows_, cols_])
        edge_weight = torch.cat((values, values)).relu()

        return edge_index, edge_weight

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hid_dim)
        self.conv2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, T=0.2):
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
            self.mask_embedding = torch.nn.Parameter(torch.zeros(1,in_dim))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.get_embeds(x, edge_index, edge_weight)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

    def process_mask(self, x, mask_nodes, mask_embedding):
        if mask_nodes != None:
            if mask_embedding:
                out_x = x.clone()
            else:
                out_x = x
            out_x[mask_nodes] = 0
            if mask_embedding:
                out_x[mask_nodes] += self.mask_embedding
        else:
            out_x = x

        return out_x

    def get_embeds(self, x, edge_index, edge_weight=None, mask_nodes=None, mask_embedding=False):
        x = self.process_mask(x, mask_nodes, mask_embedding)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x