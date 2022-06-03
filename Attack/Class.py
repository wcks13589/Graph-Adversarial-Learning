import torch

def ClassImbalance(labels):
    minority_count = len(labels)
    minority_class = -1
    for i in torch.unique(labels):
        count = sum(labels==i)
        if count < minority_count:
            minority_count = count
            minority_class = i
    labels = (labels==minority_class).long()

    return labels