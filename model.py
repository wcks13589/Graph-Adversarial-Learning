from tabnanny import verbose
from deeprobust.graph.utils import accuracy
from sklearn.metrics import confusion_matrix, recall_score

class Defender():
    def __init__(self, args, device, surrogate=False) -> None:
        self.args = args
        self.defender = args.defender
        self.device = device
        if surrogate:
            self.defender = 'gcn'

        self.verbose = args.verbose

    def init_model(self, features, labels):

        nnodes, nfeat = features.shape
        nclass = labels.max().item() + 1

        if self.defender == 'CoG':
            from Defense.CoG_Series.CoG import CoG
            model = CoG(nfeat=nfeat, nhid=self.args.hidden, nclass=nclass, 
                        dropout=self.args.dropout,lr=self.args.lr, 
                        weight_decay=self.args.weight_decay, device=self.device).to(self.device)
        elif self.defender == 'NewCoG':
            # from Defense.Meeting_0614.RSGNN import CoG
            # from Defense.Meeting_0607.Edge_Mix_New_Test import CoG
            # from Defense.Meeting_0607.Edge_Mix_New_Create_Nodes_Ablation import CoG
            # from Defense.Final_Version.MyNet import CoG
            from Defense.Final_Version.MyNet_Ablation import CoG
            model = CoG(args=self.args, nfeat=nfeat, nhid=self.args.hidden, nclass=nclass, 
                        dropout=self.args.dropout,lr=self.args.lr,weight_decay=self.args.weight_decay, 
                        verbose=self.verbose, device=self.device).to(self.device)

        elif self.defender == 'RSGNN':
            from Defense.Meeting_0614.RSGNN_Pseudo import RSGNN
            model = RSGNN(nfeat=nfeat, nhid=self.args.hidden, nclass=nclass, 
                          dropout=self.args.dropout,lr=self.args.lr, 
                          weight_decay=self.args.weight_decay, device=self.device).to(self.device)
        elif self.defender == 'RGCN':
            from deeprobust.graph.defense import RGCN
            model = RGCN(nnodes, nfeat, self.args.hidden, nclass, device=self.device).to(self.device)

        elif self.defender == 'SimPGCN':
            from deeprobust.graph.defense import SimPGCN
            model = SimPGCN(nnodes, nfeat, self.args.hidden, nclass, device=self.device).to(self.device)
        
        elif self.defender == 'GCN_SVD':
            from deeprobust.graph.defense import GCNSVD
            model = GCNSVD(nfeat, self.args.hidden, nclass, device=self.device).to(self.device)

        elif self.defender == 'GCN_Jaccard':
            from deeprobust.graph.defense import GCNJaccard
            model = GCNJaccard(nfeat, self.args.hidden, nclass, device=self.device).to(self.device)
        else:
            if self.defender in ['gcn', 'prognn']:
                from deeprobust.graph.defense import GCN
            elif self.defender == 'MyGCN':
                from Defense.MyGCN import GCN
            lr = 0.01 if self.defender == 'gcn' else self.args.lr
            
            model = GCN(nfeat=nfeat, nhid=self.args.hidden, nclass=nclass, 
                        dropout=self.args.dropout, lr=lr, weight_decay=self.args.weight_decay,
                        device=self.device).to(self.device)

        if self.defender == 'prognn':
            from deeprobust.graph.defense import ProGNN
            model = ProGNN(model, self.args, self.device)
            
            self.features = features

        self.nfeat, self.nclass, self.hidden_sizes = nfeat, nclass, self.args.hidden

        return model

    def fit(self, features, adj, labels, idx_train, idx_val, idx_test):
        self.model = self.init_model(features, labels)
        if self.defender in ['gcn', 'prognn', 'RGCN', 'SimPGCN', 'GCN_SVD', 'GCN_Jaccard']:
            self.model.fit(features, adj, labels, idx_train, idx_val, verbose=self.verbose)
        else:
            self.model.fit(features, adj, labels, idx_train, idx_val, idx_test)

    def test(self, labels, idx_test, attacker, features=None, adj=None):
        if self.defender == 'prognn':
            if attacker == 'Class':
                adj = self.model.best_graph
                if self.best_graph is None:
                    adj = self.model.estimator.normalize()
                output = self.model(self.features, adj)
                y_pred = output.argmax(-1).cpu()
                acc = recall_score(labels[idx_test].cpu().numpy(), y_pred[idx_test].numpy())
            else:
                acc = self.model.test(self.features, labels, idx_test)
        elif self.defender in ['RGCN', 'SimPGCN', 'GCN_SVD', 'GCN_Jaccard']:
            acc = self.model.test(idx_test)
        else:
            self.model.eval()
            output = self.model.predict(features, adj).cpu()
            if attacker == 'Class':
                y_pred = output.argmax(-1).cpu()
                acc = recall_score(labels[idx_test].cpu().numpy(), y_pred[idx_test].numpy())
            else:
                acc = accuracy(output[idx_test], labels[idx_test]).cpu().item()

        return acc

    def predict(self):
        if self.defender == 'prognn':
            output = self()
        else:
            output = self.model.predict()
        return output

    def __call__(self, features=None, adj=None):
        if self.defender == 'prognn':
            self.model.model.eval()
            adj = self.model.best_graph
            if self.model.best_graph is None:
                adj = self.model.estimator.normalize()
            output = self.model.model(self.features, adj)
        else:
            output = self.model(features, adj)

        return output

    def get_embed(self, features, adj):
        if self.defender == 'prognn':
            adj = self.model.best_graph
            if self.model.best_graph is None:
                adj = self.model.estimator.normalize()
            embed = self.model.model.get_embed(features, adj)
        else:
            embed = self.model.get_embed(features, adj)

        return embed
    
    def eval(self):
        if self.defender == 'prognn':
            self.model.model.eval()
        else:
            self.model.eval()

    def confusion(self, labels, idx_test):
        y_pred = self.predict().max(-1)[1].detach().cpu()
        return confusion_matrix(labels.cpu()[idx_test], y_pred[idx_test])