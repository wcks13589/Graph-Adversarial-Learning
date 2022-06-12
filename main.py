import argparse
import numpy as np
import torch

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

from model import Defender
from utils import resplit_data, get_train_val_test, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=18, help='Random seed')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr'])
parser.add_argument('--ptb_rate', type=float, default=0.15, choices=[0.05, 0.1, 0.15, 0.2, 0.25], help='Pertubation rate')
parser.add_argument('--attacker', type=str, default='meta', choices=['PGD', 'meta', 'Label', 'Class'])
parser.add_argument('--defender', type=str, default='NewCoG', choices=['gcn', 'prognn', 'MyGCN', 'SLAPS', 'CoG', 'NewCoG', 'RSGNN'])

# model training setting
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') # 記得train GCN的時候lr要改成0.01，其餘為0.001
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

# PGD setting
parser.add_argument('--epochs_pgd', type=int,  default=200, help='Number of epochs to train on PGD')
parser.add_argument('--loss_type', type=str, default='tanhMarginMCE', choices=['CE', 'CW', 'tanhMarginMCE', 'CL'])
parser.add_argument('--attack_graph', action="store_false", default=True)
parser.add_argument('--attack_feat', action="store_true", default=False)

# ProGNN setting
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true', default=False, help='test the performance of gcn without other components')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train on ProGNN.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0.001, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False, help='whether use symmetric matrix')

# NewCoG setting
parser.add_argument('--use_gan', action='store_true', default=False, help='use gan to generate fake node features')
parser.add_argument('--iterations', type=int,  default=20, help='Number of iteration to add pseudo label to training set.')

# Argument Initialization
args = parser.parse_args([])

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    # Prepare Data
    data = Dataset(root='./data/', name=args.dataset, setting='prognn')
    adj, features, labels = preprocess(data.adj, data.features, data.labels, preprocess_feature=True, device=device)
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train, idx_val, idx_test = resplit_data(data.idx_train, data.idx_val, data.idx_test, data.labels)
    # idx_train, idx_val, idx_test = get_train_val_test(features.shape[0], stratify=data.labels, seed=args.seed)
    classes, counts = np.unique(data.labels[idx_train], return_counts=True)
    print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], classes.shape[0], dict(zip(classes, counts)))
    n_perturbations = int(args.ptb_rate * torch.div(adj.sum(), 2))

    # Model Initialization
    model = Defender(args, device)

    # (1) Evaluate on Clean Data
    print('=== (1) Evaluate on Clean Data ===')
    # model.fit(features, adj, labels, idx_train, idx_val, idx_test)
    # acc_clean = model.test(labels, idx_test, args.attacker)
    acc_clean = 0 
    print(acc_clean)
    # (2) Evaluate on Perturbed Data - Evasion Attack
    print('=== (2) Evaluate on Perturbed Data - Evasion Attack ===')

    # Generate Bad Graph
    if args.attacker == 'meta':
        from deeprobust.graph.data import PrePtbDataset
        perturbed_data = PrePtbDataset(root='./pertubed_data/',
                                       name=args.dataset,
                                       attack_method=args.attacker,
                                       ptb_rate=args.ptb_rate)
        modified_adj = perturbed_data.adj
        modified_adj, features, labels = preprocess(modified_adj, data.features, data.labels, preprocess_feature=True, device=device)
        noise_labels = labels
        
    elif args.attacker == 'PGD':
        from Attack.PGD import NewPGDAttack
        n_samples, n_features = features.shape

        if args.defender == 'prognn':
            surrogate = Defender(args, device, surrogate=True)
            surrogate.fit(features, adj, labels, idx_train, idx_test)
        else:
            surrogate = model

        attacker = NewPGDAttack(model=surrogate, n_samples=n_samples, n_features=n_features,
                                attack_structure=args.attack_graph, attack_features=args.attack_feat,
                                loss_type=args.loss_type, device=device).to(device)

        fake_labels = torch.argmax(model.predict(), 1)
        idx_fake = np.concatenate([idx_train,idx_test])
        # idx_others = list(set(np.arange(len(labels))) - set(idx_train))
        # fake_labels = torch.cat([labels[idx_train], fake_labels[idx_others]])

        attacker.attack(features, adj, adj, fake_labels, idx_fake, n_perturbations, epochs=args.epochs_pgd)

        if args.attack_graph:
            modified_adj = attacker.modified_adj
        else:
            modified_adj = adj
        if args.attack_feat:
            features = attacker.modified_feat[:n_samples, n_samples:]
        
        noise_labels = labels

    elif args.attacker == 'Label':
        from Attack.Label import noisify_labels
        noise_labels = noisify_labels(labels, idx_train, idx_val, args.ptb_rate)
        modified_adj = adj

    elif args.attacker == 'Class':
        from Attack.Class import ClassImbalance
        noise_labels = ClassImbalance(labels)
        modified_adj = adj

        classes, counts = np.unique(noise_labels.cpu().numpy()[idx_train], return_counts=True)
        print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], classes.shape[0], dict(zip(classes, counts)))

    if args.defender == 'prognn' or args.attacker == 'Label':
        print('Sorry ProGNN do not support evasion attack test')
        acc_evasion = 0
    else:
        # acc_evasion = model.test(labels, idx_test, features, modified_adj)
        acc_evasion = 0

    # (3) Evaluate on Perturbed Data - Poison Attack
    print('=== (3) Evaluate on Perturbed Data - Poison Attack ===')
    model.fit(features, modified_adj, noise_labels, idx_train, idx_val, idx_test)
    acc_poison = model.test(noise_labels, idx_test, args.attacker)

    print(f'Clean Graph: {acc_clean:.4f}')
    print(f'Evasion Graph: {acc_evasion:.4f}')
    print(f'Poison Graph: {acc_poison:.4f}')
    print(f'Confusion model:\n{model.confusion(noise_labels, idx_test)}')

if __name__ == '__main__':
    main(args)