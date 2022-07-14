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

import itertools

class CoG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, verbose=False, device=None) -> None:
        super().__init__()

        self.n_class = nclass
        self.nhid = nhid
        self.nfeat = nfeat

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.pseudo_nodes_list = []
        
        self.k = 200
        self.init_models()

        self.verbose = verbose
        
    def init_models(self):

        self.graph_learner = Subtab(self.nfeat, self.nhid, self.nhid, n_subsets=3, overlap=0.75, masking_ratio=0.3, device=self.device)
        self.decoder = GCN(self.nhid, self.nhid, self.nfeat, self.nhid)
        self.model_s = GCN(self.nfeat, self.nhid, self.nhid, self.n_class, mask=True)
        self.encoder_to_decoder = nn.Linear(self.nhid, self.nhid)

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

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def mask_feature_loss(self, real_features, pred_features, loss_type='cos', alpha=1):
        if loss_type == 'mse':
            loss = F.mse_loss(real_features, pred_features)
        else:
            x = F.normalize(real_features, p=2, dim=-1)
            y = F.normalize(pred_features, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            loss = loss.mean()
        return loss
    
    def create_fake_nodes(self, num_fakes, x, labels, idx_train, mode='softmax'):
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
            random_weight = torch.rand(n_fake, x_class.size(0)).to(self.device)
            if mode == 'normalize':
                w = F.normalize(random_weight, p=1)
            elif mode == 'softmax':
                w = F.softmax(random_weight, 1)
            x_fake.append(torch.mm(w, x_class))
            labels_fake.append(torch.LongTensor([c]*n_fake))

            for j in range(n_fake):
                for node in idx_train[labels==c]:
                    edge_fake.append([i, node])
                    edge_fake.append([node, i])
                i += 1

        return torch.cat(x_fake), torch.cat(labels_fake).to(self.device), torch.LongTensor(edge_fake).T.to(self.device)

    def create_mask_nodes(self, mask_rate):
        perm = torch.randperm(self.n_real, device=self.device)
        if mask_rate > 1:
            num_mask_nodes = mask_rate
        else:
            num_mask_nodes = int(self.n_real * mask_rate)

        return perm[:num_mask_nodes], perm[num_mask_nodes:]

    def delete_edges(self, real_edge_index, embeddings, threshold=0.2):
        z = embeddings[:self.n_real]
        sim_matrix = _similarity(z)
        edge_weight = sim_matrix[tuple(real_edge_index)]

        edge_mask = edge_weight > threshold

        return real_edge_index[:, edge_mask], edge_weight[edge_mask], edge_mask

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=25):
        # real_mask = torch.load('./image/cora_real_mask_0.2')
        train_mask = torch.LongTensor(idx_train).to(self.device)
        training_labels = labels.clone()
        self.init_label_dict(labels, idx_train)

        self.n_real = x.size(0)
        
        optimizer = torch.optim.Adam(list(self.model_s.parameters())+
                                     list(self.decoder.parameters())+
                                     list(self.graph_learner.parameters())+
                                     list(self.encoder_to_decoder.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)

        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        single_mask = real_edge_index[0]<real_edge_index[1]
        self.edge_mask = torch.zeros(single_mask.sum().item()).bool()

        # self.x = x
        
        fake_x, fake_labels, fake_edge_index_0 = self.create_fake_nodes(50, x, labels, idx_train)
        self.x =  torch.cat([x, fake_x])
        train_mask = torch.cat([train_mask, torch.arange(len(fake_x)).to(self.device)+self.n_real])
        training_labels = torch.cat([training_labels, fake_labels])
        
        # fake_labels = labels[idx_train] # torch.arange(self.n_class).to(self.device)
        # fake_x = x[idx_train] # self.embed(fake_labels)
        # self.x =  torch.cat([x, fake_x]) # torch.cat([x, x])
        # train_mask = torch.cat([train_mask, torch.arange(len(idx_train)).to(self.device)+self.n_real])
        # training_labels = torch.cat([training_labels, fake_labels])
        batch_size = 1244
        best_acc = 0

        n_step_train = self.n_real // batch_size
        # if self.n_real % batch_size != 0:
        #     n_step_train += 1
        
        n_step_test = self.x.size(0) // batch_size
        if self.x.size(0) % batch_size != 0:
            n_step_test += 1

        for i in trange(iteration):
            for epoch in trange(100):
                optimizer.zero_grad()
                j = 0
                for j in range(n_step_train):
                    self.graph_learner.train()
                    start = j*batch_size
                    end = (j+1)*batch_size
                    if end > self.n_real:
                        end = self.n_real
                    loss_lp = self.graph_learner.calc_loss(self.x[start:end])

                embeddings_list = []
                for j in range(n_step_test):
                    start = j*batch_size
                    end = (j+1)*batch_size
                    if end > self.x.size(0):
                        end = self.x.size(0)
                    embeddings, z = self.graph_learner.get_embeds(self.x[start:end])
                    embeddings_list.append(embeddings)
                    embeddings = torch.cat(embeddings_list)
                # loss_lp = self.recons_loss(embeddings[:self.n_real], real_edge_index, stepwise=False)

                fake_edge_index, fake_edge_weight = knn_fast(embeddings, self.k, 1000, self.device)
                fake_edge_index, fake_edge_weight = add_edges(fake_edge_index, fake_edge_weight, 
                                                              training_labels, train_mask, self.n_real, mode='mix', threshold=0.8)

                # self.edge_index = real_edge_index
                # self.edge_weight = real_edge_weight
                new_edge_index, new_edge_weight, edge_mask = self.delete_edges(real_edge_index, embeddings, 0.5)
                self.edge_index = torch.cat([new_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([new_edge_weight, fake_edge_weight])

                mask_nodes, unmask_nodes = self.create_mask_nodes(0.5)
                # hop_nodes = k_hop_subgraph(mask_nodes, 2, real_edge_index)[0]
                # hop_nodes = hop_nodes[~torch.isin(hop_nodes, mask_nodes)]

                z1 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=mask_nodes, mask_embedding=True)
                z2 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight)
                # z3 = self.model_s.get_embeds(self.x[torch.randperm(self.x.shape[0])], self.edge_index, self.edge_weight)

                # loss_sim = (1-(F.normalize(z1[unmask_nodes], p=2, dim=-1) * F.normalize(z2[unmask_nodes], p=2, dim=-1)).sum(dim=-1)).mean().pow_(3)
                loss_sim = (F.normalize(z1[unmask_nodes], p=2, dim=-1) * F.normalize(z2[unmask_nodes], p=2, dim=-1)).sum(dim=-1).mean() * -1
                # loss_sim += (F.normalize(z1[unmask_nodes], p=2, dim=-1) * F.normalize(z3[unmask_nodes], p=2, dim=-1)).sum(dim=-1).mean()
                # loss_sim += (F.normalize(z2[unmask_nodes], p=2, dim=-1) * F.normalize(z3[unmask_nodes], p=2, dim=-1)).sum(dim=-1).mean()
                z1 = self.encoder_to_decoder(z1)
                reconst = self.decoder.get_embeds(z1, real_edge_index, None, mask_nodes=mask_nodes)
                loss_mask = self.mask_feature_loss(self.x[mask_nodes], reconst[mask_nodes])
                # loss_mask = self.mask_feature_loss(self.x[:self.n_real], reconst[:self.n_real])

                s_pred, loss_s = self.forward_classifier(self.model_s, (self.x, self.edge_index, self.edge_weight), 
                                                         training_labels, train_mask)
                loss = loss_s + loss_lp + loss_mask + loss_sim * 0.1
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

                if self.verbose:
                    if epoch % 20 == 0:
                        accs = [round(acc, 4) for acc in accs]
                        print(accs, (~edge_mask).sum().item(), train_mask.shape[0], self.edge_index.shape[1], real_edge_index.shape[1], loss.item(), f'{best_acc:.4f}', f'{best_test_acc:.4f}')

            # update pseudo label
            # for node in self.pseudo_nodes_list:
            #     s_idx = s_pred[node].max(-1)
            #     training_labels[node] = s_idx[1]

            pseudo_nodes, pseudo_labels_s = self.add_nodes(train_mask, embeddings, real_edge_index, labels)
            training_labels[pseudo_nodes] = pseudo_labels_s

            train_mask = torch.cat([train_mask, pseudo_nodes])
            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

            # sim_mat = _similarity(x)
            # edge_weight = sim_mat[tuple(real_edge_index[:, single_mask])]
            # unlabel_mask = torch.where(self.edge_mask == False)[0]
            # try:
            #     _, train_edges = edge_weight[unlabel_mask].topk(300)

            #     # self.edge_mask[unlabel_mask[add_edges]] = True
            #     for idx in unlabel_mask[train_edges]:
            #         if edge_weight[idx] > 0:
            #             self.edge_mask[idx] = True
            # except:
            #     pass

        self.restore_all(x, best_model_s_wts, best_model_g_wts, best_edge_index, best_edge_weight)
        print('correct labels',(training_labels[train_mask[train_mask<self.n_real]] == labels[train_mask[train_mask<self.n_real]]).sum()/len(train_mask[train_mask<self.n_real]))

    def restore_all(self, x, model_s_wts, model_g_wts, edge_index, edge_weight):
        
        self.model_s.load_state_dict(model_s_wts)
        self.graph_learner.load_state_dict(model_g_wts)

        self.graph_learner.eval()
        self.model_s.eval()

        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def add_nodes(self, train_mask, embeddings, real_edge_index, labels, n=100):
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

        ######
        # mask = torch.isin(torch.arange(self.n_real).to(self.device), new_nodes_s)
        # unlabel_nodes = torch.where(~mask)[0]

        hop_nodes = k_hop_subgraph(new_nodes_s, 1, real_edge_index)[0]
        hop_nodes = hop_nodes[~torch.isin(hop_nodes, new_nodes_s)]

        embeddings_1= self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=hop_nodes, mask_embedding=True)
        pred_1 = self.model_s.output(embeddings_1)
        # # embeddings_2 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight, mask_nodes=new_nodes_s, mask_embedding=True)
        # # pred_2 = self.model_s.output(embeddings_2)
        # # mask = pred_1[new_nodes_s].max(1)[1] == pred_2[new_nodes_s].max(1)[1]
        mask = pred_1[new_nodes_s].max(1)[1] == new_labels_s

        # sim_mat = _similarity(embeddings) - torch.eye(embeddings.size(0), device=self.device)

        # max_sim = []
        # for nodes in new_nodes_s:
        #     max_sim.append(sim_mat[nodes].max().item())
        # max_sim = torch.Tensor(max_sim).to(self.device)
        # mask = (max_sim < 0.8) + mask
        # print(mask.shape, mask.sum(), (~mask).sum(), (new_labels_s[~mask]==labels[new_nodes_s][~mask]).sum())

        # uncertain_nodes = new_nodes_s[~mask]


        # plt.figure()
        # max_sim = []
        # for nodes in new_labels_s[new_labels_s==labels[new_nodes_s]]:
        #     max_sim.append(sim_mat[nodes].max().item())
        # sns.histplot(data=max_sim, bins=30, color='skyblue', stat='count')
        # max_sim = []
        # for nodes in new_labels_s[new_labels_s!=labels[new_nodes_s]]:
        #     max_sim.append(sim_mat[nodes].max().item())
        # sns.histplot(data=max_sim, bins=30, color='red', stat='count', alpha=0.6)
        # true_mask = new_labels_s==labels[new_nodes_s]
        # embeddings_1 = self.model_s.get_embeds(self.x, self.edge_index, self.edge_weight)

        # 
        # plt.figure()
        # sns.histplot(data=max_sim, bins=30, color='skyblue', stat='count')
        # sns.histplot(data=sim[~true_mask].detach().cpu(), bins=30, color='red', stat='count', alpha=0.6)
        # plt.savefig(f'./image/testplt_100.jpg')
        # plt.close('all')

        return new_nodes_s[mask], new_labels_s[mask]
        ######

        # return new_nodes_s, new_labels_s
        
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

class Subtab(nn.Module):
    def __init__(self, nfeat, nhid, out_dim, n_subsets, overlap, masking_ratio, add_noise=True, noise_type='swap_noise', noise_level=0.1, device=None):
        super().__init__()
        self.n_subsets = n_subsets
        self.n_column_subset = int(nfeat / n_subsets)
        self.n_overlap = int(overlap * self.n_column_subset)
        self.column_idx = list(range(nfeat))

        in_dim = self.n_column_subset+self.n_overlap
        self.encoder = MLP(in_dim, nhid, out_dim)
        self.decoder = MLP(out_dim, nhid, in_dim)
        
        self.linear_layer1 = nn.Linear(nhid, nhid)
        self.linear_layer2 = nn.Linear(nhid, nhid)

        self.add_noise = add_noise
        self.masking_ratio = masking_ratio
        self.noise_type = noise_type
        self.noise_level = noise_level

        self.joint_loss = JointLoss(batch_size=1244, temperature=0.2, device=device)
        self.device = device

        
    def subset_generator(self, x, combination=True):
        perm = torch.randperm(self.n_subsets)
        x_tilde_list = []
        for i in perm:
            if i == 0:
                start_idx = 0
                stop_idx = self.n_column_subset + self.n_overlap
            else:
                start_idx = i * self.n_column_subset - self.n_overlap
                stop_idx = (i + 1) * self.n_column_subset
            x_bar = x[:, self.column_idx[start_idx:stop_idx]]

            if self.add_noise:
                x_bar_noisy = self.generate_noisy_xbar(x_bar).to(self.device)
                # Generate binary mask
                mask = torch.LongTensor(np.random.binomial(1, self.masking_ratio, x_bar.shape)).to(self.device)
                # Replace selected x_bar features with the noisy ones
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
            # Add the subset to the list
            x_tilde_list.append(x_bar)

        if combination:
            x_tilde_list = self.combination(x_tilde_list)

        return x_tilde_list

    def generate_noisy_xbar(self, x):
        n, dim = x.shape
        # Initialize corruption array
        x_bar = torch.zeros([n, dim])

        # Randomly (and column-wise) shuffle data
        if self.noise_type == "swap_noise":
            for i in range(dim):
                idx = torch.randperm(n)
                x_bar[:, i] = x[idx, i]

        elif self.noise_type == "gaussian_noise":
            x_bar = x + torch.normal(0, self.noise_level, size=x.size())

        else:
            x_bar = x_bar

        return x_bar

    def combination(self, x_tilde_list):
        # np.random.choice(self.n_subsets, 2, )
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        # List to store the concatenated subsets
        concatenated_subsets_list = []
        # Go through combinations
        for (xi, xj) in subset_combinations:
            # Concatenate xi, and xj, and turn it into a tensor
            Xbatch = torch.cat([xi, xj])
            # Add it to the list
            concatenated_subsets_list.append(Xbatch)
            # Return the list of combination of subsets

        return concatenated_subsets_list

    def get_embeds(self, x, normalize=True):
        x_tilde_list = self.subset_generator(x, combination=False)
        latent_list = []
        for xi in x_tilde_list:
            latent = self.encoder.get_embeds(xi)
            z = F.leaky_relu(self.linear_layer1(latent))
            z = self.linear_layer2(z)
            z = F.normalize(z, p=2, dim=1) if normalize else z

        return latent, z

    def calc_loss(self, x, normalize=True):
        X_all = torch.cat([x, x])
        x_tilde_list = self.subset_generator(x, combination=True)

        loss = 0
        for xi in x_tilde_list:
            # If we are using combination of subsets use xi since it is already a concatenation of two subsets. 
            # Else, concatenate subset with itself just to make the computation of loss compatible with the case, 
            # in which we use the combinations. Note that Xorig is already concatenation of two copies of original input.
            X_sub = xi if xi.size(0) > x.size(0) else torch.cat([xi, xi])

            latent = self.encoder.get_embeds(X_sub)
            z = F.leaky_relu(self.linear_layer1(latent))
            z = self.linear_layer2(z)
            z = F.normalize(z, p=2, dim=1) if normalize else z

            X_recon = self.decoder.get_embeds(latent)

            tloss, closs, rloss, zloss = self.joint_loss(z, X_recon, X_sub, loss_types=['contrastive','distance'])
            loss += tloss

        return loss

class JointLoss(nn.Module):
    """
    Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
    """

    def __init__(self, batch_size, temperature, device, cosine=True):
        super(JointLoss, self).__init__()
        self.batch_size = batch_size
        # Temperature to use scale logits
        self.tau = temperature
        # Device to use: GPU or CPU
        self.device = device
        # initialize softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if cosine else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = torch.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(torch.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = torch.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        l_pos = torch.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = torch.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = torch.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.tau
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation, xrecon, xorig, loss_types=[], reconstruction=True):
        """
        Args:
            representation (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):
        """
        # recontruction loss
        recon_loss = getMSEloss(xrecon, xorig) if reconstruction else getBCELoss(xrecon, xorig)

        # Initialize contrastive and distance losses with recon_loss as placeholder
        closs, zrecon_loss = recon_loss, recon_loss

        # Start with default loss i.e. reconstruction loss
        loss = recon_loss

        if 'contrastive' in loss_types:
            closs = self.XNegloss(representation)
            loss += closs

        if 'distance' in loss_types:
            # recontruction loss for z
            zi, zj = torch.split(representation, self.batch_size)
            zrecon_loss = getMSEloss(zi, zj)
            loss += zrecon_loss

        # Return
        return loss, closs, recon_loss, zrecon_loss

def getMSEloss(recon, target):
    """
    Args:
        recon (torch.FloatTensor):
        target (torch.FloatTensor):
    """
    dims = list(target.size())
    bs = dims[0]
    loss = torch.sum(torch.square(recon - target)) / bs
    return loss

def getBCELoss(prediction, label):
    """
    Args:
        prediction (torch.FloatTensor):
        label (torch.FloatTensor):
    """
    dims = list(prediction.size())
    bs = dims[0]
    return F.binary_cross_entropy(prediction, label, reduction='sum') / bs