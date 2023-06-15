import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch import autograd
from copy import deepcopy


def loss_reweight(model,embedding,label,device,threshold=0.1):
    # implement the counterfactual adversarial dropout 
    num = embedding.shape[0]
    
    for i in range(int(num/2)):

        delta = embedding[2*i+1].detach() - embedding[2*i].detach()
        sparse_delta = deepcopy(delta)
        # focus on absolute value
        sparse_delta = abs(sparse_delta)
        zero = torch.zeros_like(sparse_delta)
        sparse_delta = torch.where(sparse_delta<threshold, zero, sparse_delta)
        embedding[2*i+1] = embedding[2*i+1] * sparse_delta
        embedding[2*i] = embedding[2*i] * sparse_delta

    output = model.classifier(embedding)
    Loss = nn.CrossEntropyLoss()
    label = label.to(device)
    loss = Loss(output,label)
    return loss

def loss_reweight_1(model,embedding,label,device,threshold=0.1):
    # implement the counterfactual adversarial dropout 
    num = embedding.shape[0]

    for i in range(int(num/3)):

        delta1 = embedding[3*i+1].detach() - embedding[3*i].detach()
        delta2 = embedding[3*i+2].detach() - embedding[3*i].detach()
        sparse_delta1 = deepcopy(delta1)
        sparse_delta2 = deepcopy(delta2)
        # focus on absolute value
        sparse_delta1 = abs(sparse_delta1)
        sparse_delta2 = abs(sparse_delta2)
        zero = torch.zeros_like(sparse_delta1)
        one = torch.ones_like(sparse_delta1)
        mask1 = torch.where(sparse_delta1<threshold, zero, one)
        mask2 = torch.where(sparse_delta2<threshold, zero, one)
        mask = mask1 * mask2
        embedding[3*i] = embedding[3*i] * mask
        embedding[3*i+1] = embedding[3*i+1] * mask
        embedding[3*i+2] = embedding[3*i+2] * mask

    output = model.classifier(embedding)
    label = label.to(device)
    Loss = nn.CrossEntropyLoss()
    loss = Loss(output,label)
    return loss
