import argparse
import numpy as np
import torch

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

from model import Defender
from utils import resplit_data, get_train_val_test, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr'])
parser.add_argument('--ptb_rate_nontarget', type=float, default=0.2, choices=[0.05, 0.1, 0.15, 0.2, 0.25], help='Pertubation rate (Metatack, PGD)')
parser.add_argument('--ptb_rate_target', type=float, default=5.0, choices=[1.0,2.0,3.0,4.0,5.0], help='Pertubation rate (Nettack)')
parser.add_argument('--attacker', type=str, default='meta', choices=['Clean', 'PGD', 'meta', 'Label', 'Class', 'nettack'])
parser.add_argument('--defender', type=str, default='NewCoG', choices=['gcn', 'prognn', 'MyGCN', 'CoG', 'NewCoG', 'RSGNN', 'RGCN', 'SimPGCN', 'GCN_SVD', 'GCN_Jaccard'])
parser.add_argument('--verbose', action="store_false", default=False)

# model training setting
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

# PGD setting
# parser.add_argument('--epochs_pgd', type=int,  default=200, help='Number of epochs to train on PGD')
# parser.add_argument('--loss_type', type=str, default='CE', choices=['CE', 'CW', 'tanhMarginMCE', 'CL'])
# parser.add_argument('--attack_graph', action="store_false", default=True)
# parser.add_argument('--attack_feat', action="store_true", default=False)

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
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('-f', '--fake_nodes', type=int, default=10)
parser.add_argument('--iteration', type=int, default=10)
parser.add_argument('--add_labels', type=int, default=250)

# Argument Initialization
args = parser.parse_args()

if args.defender == 'RSGNN':
    feature_normalize = False
else:
    feature_normalize = False

def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    seed_everything(args.seed)

    # Prepare Data
    data = Dataset(root='./data/', name=args.dataset, setting='prognn')
    if args.defender in ['RGCN', 'SimPGCN', 'GCN_SVD', 'GCN_Jaccard']:
        adj, features, labels = data.adj, data.features, data.labels
    else:
        adj, features, labels = preprocess(data.adj, data.features, data.labels, preprocess_feature=feature_normalize, device=device)
    n_samples, n_features = features.shape
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train, idx_val, idx_test = resplit_data(data.idx_train, data.idx_val, data.idx_test, data.labels)
    # idx_train, idx_val, idx_test = get_train_val_test(features.shape[0], stratify=data.labels)
    # print(sum(idx_train))

    if args.attacker in ['meta', 'nettack']:
        from deeprobust.graph.data import PrePtbDataset
        if args.attacker == 'meta':
            ptb_rate = args.ptb_rate_nontarget
        elif args.attacker == 'nettack':
            ptb_rate = args.ptb_rate_target

        perturbed_data = PrePtbDataset(root='./pertubed_data/',
                                       name=args.dataset,
                                       attack_method=args.attacker,
                                       ptb_rate=ptb_rate)
        if args.attacker == 'nettack':
            idx_test = np.array(perturbed_data.target_nodes)

        if args.defender in ['RGCN', 'SimPGCN', 'GCN_SVD', 'GCN_Jaccard']:
            modified_adj, features, labels = perturbed_data.adj, data.features, data.labels
        else:
            modified_adj, features, labels = preprocess(perturbed_data.adj, data.features, data.labels, preprocess_feature=feature_normalize, device=device)
        noise_labels = labels

    classes, counts = np.unique(data.labels[idx_train], return_counts=True)
    # print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], classes.shape[0], dict(zip(classes, counts)))

    # Model Initialization
    model = Defender(args, device)

    # (1) Evaluate on Clean Data
    # print('=== (1) Evaluate on Clean Data ===')
    if args.attacker == 'Clean':
        model.fit(features, adj, labels, idx_train, idx_val, idx_test)
        acc_clean = model.test(labels, idx_test, args.attacker)
        print(acc_clean)
    else:
        acc_clean = 0

    # (2) Evaluate on Perturbed Data - Evasion Attack
    # print('=== (2) Evaluate on Perturbed Data - Evasion Attack ===')

    # Generate Bad Graph
    if args.attacker == 'PGD':
        from Attack.PGD import NewPGDAttack
        n_perturbations = int(args.ptb_rate_nontarget * torch.div(adj.sum(), 2))

        if args.defender == 'prognn':
            surrogate = Defender(args, device, surrogate=True)
            surrogate.fit(features, adj, labels, idx_train, idx_val)
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
        noise_labels = noisify_labels(labels, idx_train, idx_val, args.ptb_rate_nontarget)
        modified_adj = adj

    elif args.attacker == 'Class':
        from Attack.Class import ClassImbalance
        noise_labels = ClassImbalance(labels)
        modified_adj = adj

        classes, counts = np.unique(noise_labels.cpu().numpy()[idx_train], return_counts=True)
        print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], classes.shape[0], dict(zip(classes, counts)))

    if args.defender == 'prognn' or args.attacker in ['Clean','PGD', 'Label', 'Class', 'nettack', 'meta']:
        # print('Sorry ProGNN do not support evasion attack test')
        acc_evasion = 0
    else:
        acc_evasion = model.test(labels, idx_test, args.attacker, features, modified_adj)

    # (3) Evaluate on Perturbed Data - Poison Attack
    # print('=== (3) Evaluate on Perturbed Data - Poison Attack ===')
    if args.attacker == 'Clean':
        acc_poison = 0
    else:
        model.fit(features, modified_adj, noise_labels, idx_train, idx_val, idx_test)
        acc_poison = model.test(labels, idx_test, args.attacker)

    # cm = model.confusion(labels, idx_test)
    print(f'New Seed is {args.seed}')
    print(f'Clean Graph: {acc_clean:.4f}')
    print(f'Evasion Graph: {acc_evasion:.4f}')
    print(f'{args.attacker} Graph: {acc_poison:.4f}')
    # print(f'Confusion model:\n{cm}')
    
    return acc_clean, acc_evasion, acc_poison, 0

if __name__ == '__main__':
    datasets = ['cora', 'citeseer', 'polblogs']
    ptb_rates = [0.2]
    thresholds = [0.8]
    ks = [5]
    fs = [10]
    if args.attacker == 'nettack' and 'cora_ml' in datasets:
        datasets.remove('cora_ml')
    for dataset in datasets:
        args.dataset = dataset
        for rate in ptb_rates:
            args.ptb_rate_nontarget = rate
            setting = ['Clean', 'Evasion', 'Poison', 'Confusion']
            result = {x:[] for x in setting}

            for threshold in thresholds:
                args.threshold = threshold
                for f in fs:
                    args.fake_nodes = f
                    for k in ks:
                        if k > f:
                            break
                        args.k = k
                        for seed in range(15, 20):
                            args.seed = seed
                            output = main(args)
                            for i, acc_score in enumerate(output):
                                result[setting[i]].append(acc_score)
                        
                        print('==' * 20, 'Final Result', '==' * 20)
                        for k, v in result.items():
                            if k == 'Confusion':
                                continue
                                print(v)
                            else:
                                print(f'{k} Graph: {np.mean(v):.4f} ?? {np.std(v):.4f}')