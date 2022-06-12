import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import negative_sampling

def _similarity(h1, h2=None):
    if h2 == None:
        h2 = h1
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def resplit_data(idx_train, idx_val, idx_test, labels, ratio=0.01):
    n_nodes = idx_train.shape[0] + idx_val.shape[0] + idx_test.shape[0]
    n_train_old = idx_train.shape[0]
    n_train_new = int(n_nodes * ratio)
    train_ratio = n_train_new / n_train_old
    
    classes, counts = np.unique(labels[idx_train], return_counts=True)

    idx_train = idx_train.copy()
    idx_val = idx_val.copy()

    idx_train_new = []
    for c, count in zip(classes, counts):
        n_samples = round(count * train_ratio)
        new_idx = idx_train[labels[idx_train]==c]
        np.random.shuffle(new_idx)
        new_train_idx = new_idx[:n_samples]
        new_val_idx = new_idx[n_samples:]

        idx_train_new.append(new_train_idx)
        # idx_val = np.concatenate([idx_val, new_val_idx])
        
    idx_train_new = np.concatenate(idx_train_new)
    
    return idx_train_new, idx_val, idx_test

from sklearn.model_selection import train_test_split
def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data
    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed
    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    idx_train = idx_train[:int(0.01 * nnodes)]

    return idx_train, idx_val, idx_test

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def knn_fast(X, k, b, device):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1), device=device)
    rows = torch.zeros(X.shape[0] * (k + 1), device=device)
    cols = torch.zeros(X.shape[0] * (k + 1), device=device)
    # norm_row = torch.zeros(X.shape[0]).to(device)
    # norm_col = torch.zeros(X.shape[0]).to(device)
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
        # norm_row[index: end] = torch.sum(vals, dim=1)
        # norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    # norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    # values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))

    rows_ = torch.cat((rows, cols))
    cols_ = torch.cat((cols, rows))
    edge_index = torch.stack([rows_, cols_])
    edge_weight = torch.cat((values, values)).relu()
    return edge_index, edge_weight

def recons_loss(z, edge_index, edge_mask=None, sample_ratio=5):
    # Version 1 MSE
    n_nodes = z.shape[0]
    randn = negative_sampling(edge_index, num_nodes=n_nodes, num_neg_samples=sample_ratio*n_nodes)
    randn = randn[:,randn[0]<randn[1]]
    
    edge_index = edge_index[:, edge_index[0]<edge_index[1]]

    if edge_mask != None:
        edge_index = edge_index[edge_mask]

    neg0 = z[randn[0]]
    neg1 = z[randn[1]]
    neg = torch.sum(torch.mul(neg0,neg1),dim=1)

    pos0 = z[edge_index[0]]
    pos1 = z[edge_index[1]]
    pos = torch.sum(torch.mul(pos0,pos1),dim=1)

    rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') \
                + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                * n_nodes/(randn.shape[1] + edge_index.shape[1])

    # Version 2
    # EPS = 1e-15
    # pos_values = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1).sigmoid()
    # pos_loss = -torch.log(pos_values + EPS).mean()
    # neg_edge_index = negative_sampling(edge_index, num_nodes=self.n_real, num_neg_samples=5*self.n_real)

    # neg_values = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).sigmoid()
    # neg_loss = -torch.log(1 - neg_values + EPS).mean()

    # rec_loss = pos_loss + neg_loss

    return rec_loss

def add_edges(fake_edge_index, fake_edge_weight, training_labels, train_mask, n_real, threshold=0.7, mode='mix'):
    edge_mask = (fake_edge_index[0] >= n_real) + (fake_edge_index[1] >= n_real)
    if mode == 'mix':
        train_label_mask = torch.logical_and(torch.isin(fake_edge_index[0], train_mask), torch.isin(fake_edge_index[1], train_mask))
        
        sim_mask = torch.logical_and(edge_mask, fake_edge_weight >= threshold)
        sim_mask = torch.logical_and(sim_mask, ~train_label_mask)

        label_mask = torch.logical_and(edge_mask, training_labels[fake_edge_index[0]] == training_labels[fake_edge_index[1]])
        label_mask = torch.logical_and(label_mask, train_label_mask)

        edge_mask = label_mask + sim_mask

    elif mode == 'threshold':
        edge_mask = torch.logical_and(edge_mask, fake_edge_weight >= threshold)

    elif mode == 'class':
        train_label_mask = torch.logical_and(torch.isin(fake_edge_index[0], train_mask), torch.isin(fake_edge_index[1], train_mask))
        edge_mask = torch.logical_and(edge_mask, training_labels[fake_edge_index[0]] == training_labels[fake_edge_index[1]])
        edge_mask = torch.logical_and(edge_mask, train_label_mask)
    
    fake_edge_index = fake_edge_index[:, edge_mask]
    fake_edge_weight = fake_edge_weight[edge_mask]

    return fake_edge_index, fake_edge_weight
