import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from tqdm import trange

class CoG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None, use_gan=False) -> None:
        super().__init__()

        self.n_class = nclass
        self.nhid = nhid
        self.nfeat = nfeat

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.pseudo_nodes_list = []      
        
        self.init_models()
        self.use_gan = use_gan

    def init_models(self):

        self.generator = Generator(128, self.nfeat, self.n_class)
        self.discriminator = Discriminator(self.nfeat, self.n_class)
        
        self.graph_learner = MLP_learner(2, self.nfeat, 20)
        self.gcl = GCL(self.nfeat, 512, 128)
        self.model_s = GCN(self.nfeat, self.nhid, self.n_class)
        self.model_f = MLP(self.nfeat, self.nhid, self.n_class)

    def train_gan(self, real_features, labels, generator, discriminator, optimizer_G, optimizer_D):

        n_real = real_features.shape[0]
        n_fakes = n_real

        valid = Variable(torch.FloatTensor(n_real, 1).fill_(1.0), requires_grad=False).to(self.device)
        fake = Variable(torch.FloatTensor(n_fakes, 1).fill_(0.0), requires_grad=False).to(self.device)
            
        optimizer_G.zero_grad()
        z = Variable(torch.Tensor(np.random.normal(0, 1, (n_fakes, 128)))).to(self.device)
        
        fake_features = generator(z, labels)
        logits = discriminator(fake_features, labels)
        g_loss = F.mse_loss(logits, valid)
        g_loss += F.nll_loss(discriminator.classify(fake_features), labels)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        validity_real = discriminator(real_features, labels)
        d_real_loss = F.mse_loss(validity_real, valid) + F.nll_loss(discriminator.classify(real_features), labels)
        
        validity_fake = discriminator(fake_features.detach(), labels)
        d_fake_loss = F.mse_loss(validity_fake, fake) + F.nll_loss(discriminator.classify(fake_features.detach()), labels)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # 檢驗用
        # sim = []
        # for i in range(self.n_class):
        #     idx = labels == i
        #     sim.append(round(_similarity(fake_features, real_features[idx]).max().item(),2))

        return fake_features.detach()

    def add_edges(self, fake_edge_index, fake_edge_weight, training_labels, train_mask):
        train_label_mask = torch.logical_and(torch.isin(fake_edge_index[0], train_mask.cuda()), torch.isin(fake_edge_index[1], train_mask.cuda()))
        edge_mask = (fake_edge_index[0] >= self.n_real) + (fake_edge_index[1] >= self.n_real)

        sim_mask = torch.logical_and(edge_mask, fake_edge_weight >= 0.9)
        sim_mask = torch.logical_and(sim_mask, ~train_label_mask)

        label_mask = torch.logical_and(edge_mask, training_labels[fake_edge_index[0]] == training_labels[fake_edge_index[1]])
        label_mask = torch.logical_and(label_mask, train_label_mask)

        edge_mask = label_mask + sim_mask
        
        fake_edge_index = fake_edge_index[:, edge_mask]
        fake_edge_weight = fake_edge_weight[edge_mask]

        return fake_edge_index, fake_edge_weight

    def forward_classifier(self, model, x, labels, mask, train=True):
        if train:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])
        else:
            model.eval()
            logit = model(*x)
            return logit

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200, iteration=20):

        train_mask = torch.LongTensor(idx_train)
        training_labels = labels.clone()

        train_y = labels[train_mask].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.01, weight_decay=self.weight_decay)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.01, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(list(self.model_f.parameters())+\
                                     list(self.model_s.parameters())+\
                                     list(self.graph_learner.parameters())+
                                     list(self.gcl.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)
        self.n_real = x.shape[0]
        self.x = x
        train_mask_with_fake = torch.cat([train_mask, torch.arange(len(train_mask))+self.n_real])
        training_labels_with_fake = torch.cat([training_labels, labels[train_mask]])

        real_edge_index = adj.nonzero().T

        best_acc = 0
        for i in trange(iteration):
            if self.use_gan:
                for epoch in range(epochs*5):
                    x_fake = self.train_gan(x[idx_train], labels[idx_train], 
                                        self.generator, self.discriminator, 
                                        optimizer_G, optimizer_D)
                    # print(f'Epoch:{epoch}, g_loss:{g_loss}, d_loss:{d_loss}, {sim}')
            else:
                x_fake = x[idx_train].clone()

            self.x_new = torch.cat([x, x_fake])
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                fake_edge_index, fake_edge_weight = self.graph_learner(self.x_new)

                # z1 = self.gcl(self.x_new, fake_edge_index, fake_edge_weight)
                # #+ self.gcl.calc_loss(z1[self.n_real:], z2[idx_train]) # V2
                # z2 = self.gcl(self.x, real_edge_index)
                # # loss = self.gcl.calc_loss(z1[:self.n_real], z2) # V1
                # loss = self.gcl.calc_loss(z1[:self.n_real], z2)

                fake_edge_index, fake_edge_weight = self.add_edges(fake_edge_index, fake_edge_weight, 
                                                                   training_labels_with_fake, train_mask_with_fake)

                labels_ = torch.cat([labels, labels[idx_train]]) # 監控用
                T = labels_[fake_edge_index[0]] == labels_[fake_edge_index[1]] # 監控用
                F = labels_[fake_edge_index[0]] != labels_[fake_edge_index[1]] # 監控用

                self.edge_index = torch.cat([real_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([adj[tuple(real_edge_index)], fake_edge_weight])

                f_pred, loss_f = self.forward_classifier(self.model_f, (x,), 
                                                         training_labels, train_mask, train=True)
                s_pred, loss_s = self.forward_classifier(self.model_s, (self.x_new, self.edge_index, self.edge_weight), 
                                                         training_labels_with_fake, train_mask_with_fake, train=True)

                # loss = loss_s
                loss = loss_f + loss_s
                loss.backward()
                optimizer.step()

                
                if epoch % 20 == 0:
                    f_pred = self.forward_classifier(self.model_f, (x,), labels, train_mask, train=False)
                    s_pred = self.forward_classifier(self.model_s, (self.x_new, self.edge_index, self.edge_weight), 
                                                    labels, train_mask, train=False)[:self.n_real]
                    accs = []
                    logits = f_pred+s_pred
                    # logits = s_pred
                    for phase, mask in enumerate([train_mask, idx_test, idx_val]):
                        pred = logits[mask].max(1)[1]
                        acc = pred.eq(labels[mask]).sum().item() / len(mask)
                        accs.append(acc)

                        if phase == 2 and acc > best_acc:
                            best_acc = acc
                            best_test_acc = accs[1]
                            best_model_f_wts = copy.deepcopy(self.model_f.state_dict())
                            best_model_s_wts = copy.deepcopy(self.model_s.state_dict())
                            best_model_g_wts = copy.deepcopy(self.graph_learner.state_dict())
                    
                    print(accs, fake_edge_weight.shape, train_mask_with_fake.shape, T.sum().item(), F.sum().item(), best_acc, best_test_acc)
            
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
            
            add_nodes_f, add_nodes_s, pseudo_labels_f, pseudo_labels_s = self.add_nodes(x, self.x_new, self.edge_index, labels, train_mask)
            training_labels[add_nodes_f] = pseudo_labels_f
            training_labels[add_nodes_s] = pseudo_labels_s

            training_labels_with_fake[add_nodes_f] = pseudo_labels_f
            training_labels_with_fake[add_nodes_s] = pseudo_labels_s

            if sum(torch.isin(add_nodes_f, add_nodes_s)) > 0:
                same_nodes = np.intersect1d(add_nodes_f.cpu(), add_nodes_s.cpu())
                for node in same_nodes:
                    f_idx = f_pred[node].max(-1)
                    s_idx = s_pred[node].max(-1)
                    if f_idx[0] > s_idx[0]:
                        training_labels[node] = f_idx[1]
                        training_labels_with_fake[node] = f_idx[1]
                    else:
                        training_labels[node] = s_idx[1]
                        training_labels_with_fake[node] = s_idx[1]

            pseudo_nodes = torch.unique(torch.cat([add_nodes_f, add_nodes_s]))
            # pseudo_nodes = add_nodes_s
            train_mask = torch.cat([train_mask, pseudo_nodes])
            train_mask_with_fake = torch.cat([train_mask_with_fake, pseudo_nodes])

            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

        self.restore_all(best_model_f_wts, best_model_s_wts, best_model_g_wts,
                         adj, real_edge_index, labels, training_labels_with_fake, train_mask_with_fake,idx_train)

        # print(fake_edge_index.shape, fake_edge_weight.shape)

    def restore_all(self, model_f_wts, model_s_wts, model_g_wts, adj, real_edge_index, labels, training_labels_with_fake, train_mask_with_fake, idx_train):
        
        self.model_f.load_state_dict(model_f_wts)
        self.model_s.load_state_dict(model_s_wts)
        self.graph_learner.load_state_dict(model_g_wts)
        
        self.model_f.eval()
        self.model_s.eval()
        self.graph_learner.eval()

        n_real = idx_train.shape[0]
        if self.use_gan:
            z = Variable(torch.Tensor(np.random.normal(0, 1, (n_real, 128)))).to(self.device)
            x_fake = self.generator(z, labels[idx_train]).detach()
        else:
            x_fake = self.x[idx_train].clone()
        
        self.x_new = torch.cat([self.x, x_fake])

        fake_edge_index, fake_edge_weight = self.graph_learner(self.x_new)
        fake_edge_index, fake_edge_weight = self.add_edges(fake_edge_index, fake_edge_weight, training_labels_with_fake, train_mask_with_fake)

        self.edge_index = torch.cat([real_edge_index, fake_edge_index], -1)
        self.edge_weight = torch.cat([adj[tuple(real_edge_index)], fake_edge_weight])

    def add_nodes(self, x, x_new, edge_index, labels, train_mask, n=50):
        mask = torch.isin(torch.arange(x.shape[0]), train_mask)
        unlabel_nodes = torch.where(~mask)[0]

        new_nodes_f = []
        new_nodes_s = []

        new_labels_f = []
        new_labels_s = []

        f_pred = self.forward_classifier(self.model_f, (x,), labels, train_mask, train=False)
        s_pred = self.forward_classifier(self.model_s, 
                                         (x_new, self.edge_index, self.edge_weight), 
                                         labels, train_mask, train=False)[:self.n_real]
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
        s_pred = self.forward_classifier(self.model_s, (self.x_new, edge_index, edge_weight), 
                                         labels=None, mask=None, train=False)[:self.n_real]
        
        return (f_pred+s_pred)/2
        # return s_pred

    def predict(self, x=None, adj=None):
        return self.forward(x, adj)

    def get_embed(self, x, adj):
        # edge_index = adj.nonzero().T
        # edge_weight = adj[tuple(edge_index)]
        self.model_f.eval()
        self.model_s.eval()
        f_embeds = self.model_f.get_embeds(x)
        s_embeds = self.model_s.get_embeds(self.x_new, self.edge_index, self.edge_weight)[:self.n_real]
        
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

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, nclass):
        super().__init__()
        self.label_emb = nn.Embedding(nclass, nclass)

        self.model = nn.Sequential(
            nn.Linear(in_dim + nclass, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, labels):
        label_embed = self.label_emb(labels)
        gen_input = torch.cat((x, label_embed), -1)
        output = self.model(gen_input)
        
        return output

class Discriminator(nn.Module):
    def __init__(self, in_dim, nclass):
        super().__init__()
        self.label_emb = nn.Embedding(nclass, nclass)

        self.model = nn.Sequential(
            nn.Linear(in_dim+nclass, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, nclass)
        )

    def forward(self, x, labels):
        # Concatenate label embedding and image to produce input
        label_embed = self.label_emb(labels)
        d_in = torch.cat((x, label_embed), -1)
        # d_in = x
        validity = self.model(d_in)
        return validity

    def classify(self,x):
        x = self.classifier(x)
        return F.log_softmax(x, -1)

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

        return edge_index, edge_weight


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



def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

def knn_fast(X, k, b, device):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(device)
    norm_row = torch.zeros(X.shape[0]).to(device)
    norm_col = torch.zeros(X.shape[0]).to(device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    # values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))

    rows_ = torch.cat((rows, cols))
    cols_ = torch.cat((cols, rows))
    edge_index = torch.stack([rows_, cols_])
    edge_weight = torch.cat((values, values)).relu()
    return edge_index, edge_weight