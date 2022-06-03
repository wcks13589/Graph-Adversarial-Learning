import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _similarity
from torch_cluster import knn_graph

from torch_geometric.nn import GCNConv
from deeprobust.graph.utils import accuracy

class SLAPS(nn.Module):
    def __init__(self, nfeat, nclass, k, ratio=10, nr=5, lr=0.01, weight_decay=5e-4, dropout=0.5, l=10) -> None:
        super().__init__()
        self.k = k
        self.r = ratio
        self.nr = nr
        self.l = l

        self.lr = lr
        self.weight_decay =weight_decay
        self.doupout = dropout
        self.adj_generator = MLP(nfeat, nfeat)
        self.encoder = GCN(nfeat, 512, nfeat)
        self.clssifier = GCN(nfeat, 32, nclass)

    def load_adj(self, x):
        edge_index, edge_weight = self.adj_generator(x, self.k)
        #　Adj Processor
        edge_weight = F.dropout(edge_weight, self.doupout, training=self.training)

        return edge_index, edge_weight

    def load_noised_x(self, x):
        #　create mask
        nones = torch.sum(x > 0.0).float()
        nzeros = x.shape[0] * x.shape[1] - nones
        pzeros = nones / nzeros / self.r * self.nr
        probs = torch.zeros(x.shape).cuda()
        probs[x == 0.0] = pzeros
        probs[x > 0.0] = 1 / self.r
        self.mask = torch.bernoulli(probs)

        return x * (1-self.mask)

    def get_loss_masked_features(self, x):
        noised_x = self.load_noised_x(x)
        self.edge_index, self.edge_weight = self.load_adj(x)
        logits = self.encoder.get_embeds(noised_x, self.edge_index, self.edge_weight)
        mask_idx = self.mask > 0
        loss = F.binary_cross_entropy_with_logits(logits[mask_idx], x[mask_idx])
        return loss

    def fit(self, x, adj, labels, idx_train, idx_val=None):
        optimizer_adj = torch.optim.Adam(list(self.adj_generator.parameters())+ \
                                         list(self.encoder.parameters()), 
                                         lr=0.001, 
                                         weight_decay=0.0005)
        optimizer_cls = torch.optim.Adam(self.clssifier.parameters(), lr=0.01,
                                         weight_decay=0.0005)
        self.x = x
        loss2 = 0
        for epoch in range(2000):
            self.train()
            optimizer_adj.zero_grad()
            optimizer_cls.zero_grad()
            loss1 = self.get_loss_masked_features(x) * self.l
            
            if epoch >= 2000 // 5:
                logits = self.clssifier(x, self.edge_index, self.edge_weight)
                loss2 = F.nll_loss(logits[idx_train], labels[idx_train])

            loss = loss1 + loss2
            loss.backward()
            optimizer_adj.step()
            optimizer_cls.step()

            if epoch % 40 == 0:
                self.eval()
                logits = self.clssifier(x, self.edge_index, self.edge_weight)
                acc = accuracy(logits[idx_val], labels[idx_val]).cpu().item()
                print("Epoch {:05d} | Train Loss {:.4f}, {:.4f} Val Acc {:.4f}".format(epoch, loss1.item(), float(loss2), acc))

    def forward(self):
        return self.clssifier(self.x, self.edge_index, self.edge_weight)

    def predict(self, x=None, adj=None):
        return self.forward()

    def get_embed(self, features, adj):
        return self.clssifier.get_embeds(self.x, self.edge_index, self.edge_weight)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.get_embeds(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    def get_embeds(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        return self.conv2(x, edge_index, edge_weight)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.layers = nn.Sequential(nn.Linear(in_dim, in_dim),
                                    nn.ReLU(),
                                    nn.Linear(in_dim, out_dim))
        self.weight_initinal()

    def weight_initinal(self):
        #　Version 1
        for layer in self.layers:
            if layer._get_name() == 'Linear':
                layer.weight = nn.Parameter(torch.eye(self.in_dim))
        # Version 2
        # optimizer = torch.optim.Adam(self.parameters(), 0.01)
        #     labels = torch.from_numpy(nearest_neighbors(self.features.cpu(), self.k, self.knn_metric)).cuda()

        #     for epoch in range(1, self.mlp_epochs):
        #         self.train()
        #         logits = self.forward()
        #         loss = F.mse_loss(logits, labels, reduction='sum')
        #         if epoch % 10 == 0:
        #             print("MLP loss", loss.item())
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        
    def forward(self, x, k):
        embeds = self.layers(x)

        sim_mat = _similarity(embeds)
        row, col, edge_weight = knn_fast(embeds, k, 1000)
        edge_index = torch.stack([row, col])
        edge_index = torch.cat([edge_index, torch.stack([col, row])])
        edge_weight = torch.cat([edge_weight, edge_weight])

        return edge_index, edge_weight

def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
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
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values