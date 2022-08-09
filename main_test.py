import argparse
import numpy as np
import torch

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

from main import main
from model import Defender
from utils import resplit_data, get_train_val_test, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=19, help='Random seed')
parser.add_argument('--dataset', type=str, default='film', choices=['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed', 'wisconsin', 'cornell', 'texas', 'film'])
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
parser.add_argument('-f', '--fake_nodes', type=int, default=5)
parser.add_argument('--iteration', type=int, default=5)
parser.add_argument('--add_labels', type=int, default=40)

# Argument Initialization
args = parser.parse_args()

if args.defender == 'RSGNN':
    feature_normalize = False
else:
    feature_normalize = False

if __name__ == '__main__':
    datasets = ['cornell']
    ptb_rates = [0.2, 0.15, 0.1, 0.05, 0.0]
    # ptb_rates = [0.2]
    # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.3]
    ks = [8]
    # ks = [5]
    fs = [30]
    for dataset in datasets:
        args.dataset = dataset
        best_results = []
        for rate in ptb_rates:
            args.ptb_rate_nontarget = rate
            setting = ['Clean', 'Evasion', 'Poison', 'Confusion']
            for f in fs:
                args.fake_nodes = f
                for threshold in thresholds:
                    args.threshold = threshold
                    for k in ks:
                        if k > f:
                            break
                        args.k = k
                        result = {x:[] for x in setting}
                        for seed in range(15, 20):
                            args.seed = seed
                            output = main(args)
                            for i, acc_score in enumerate(output):
                                result[setting[i]].append(acc_score)
                        
                        print('==' * 20, 'Final Result', '==' * 20)
                        print('Threshold:', args.threshold, 'F:', args.fake_nodes, 'K:', args.k)
                        for k, v in result.items():
                            if k == 'Confusion' or np.mean(v) == 0:
                                continue
                                print(v)
                            else:
                                acc = f'{np.mean(v):.4f} ± {np.std(v):.4f}'
                                print(f'{k} Graph:', acc)
                                if np.mean(v) > 0:
                                    best_results.append([args.threshold, args.fake_nodes, args.k, acc])

    print(best_results)