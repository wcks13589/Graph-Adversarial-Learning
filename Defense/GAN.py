import torch
import torch.nn as nn

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