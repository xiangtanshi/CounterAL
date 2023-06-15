import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch import autograd
from copy import deepcopy

    
def loss_GS(model,embedding,label,device,namda=1):
    # implement the gradient supervision loss item, GS is a good loss for initial training
    # label of the counterfactual data is not utilized
    num = embedding.shape[0]
    index = [2*i for i in range(int(num/2))]
    f_embed = embedding[index]
    f_embed.retain_grad()
    f_label = label[index].to(device)
    output = model.classifier(f_embed)
    Loss = nn.CrossEntropyLoss()
    loss_main = Loss(output,f_label)
    gradient = autograd.grad(outputs=loss_main,inputs=f_embed,retain_graph=True,create_graph=True)

    loss_gs = 0
    for i in range(int(num/2)):
        delta = embedding[2*i+1] - embedding[2*i]
        grad = gradient[0][i]
        cos_similarity = delta[0]@grad[0]/torch.norm(delta[0])/torch.norm(grad[0])
        loss_gs += 1 - cos_similarity

    return loss_main + namda*loss_gs

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
    # version 2: do not use the label of the counterfactual samples
    num = embedding.shape[0]
    index = [2*i for i in range(int(num/2))]
    f_embed = embedding[index]
    f_label = label[index].to(device)

    for i in range(int(num/2)):

        delta = embedding[2*i+1].detach() - embedding[2*i].detach()
        sparse_delta = deepcopy(delta)
        # focus on absolute value
        sparse_delta = abs(sparse_delta)
        zero = torch.zeros_like(sparse_delta)
        sparse_delta = torch.where(sparse_delta<threshold, zero, sparse_delta)
        f_embed[i] = f_embed[i] * sparse_delta

    output = model.classifier(f_embed)
    Loss = nn.CrossEntropyLoss()
    loss = Loss(output,f_label)
    return loss

def loss_mask(model,embedding,label,device,threshold=0.1):
    # implement the counterfactual adversarial dropout 
    # only apply masking
    num = embedding.shape[0]
    
    for i in range(int(num/2)):

        delta = embedding[2*i+1].detach() - embedding[2*i].detach()
        sparse_delta = deepcopy(delta)
        # focus on absolute value
        sparse_delta = abs(sparse_delta)
        zero = torch.zeros_like(sparse_delta)
        one = torch.ones_like(sparse_delta)
        mask = torch.where(sparse_delta<threshold, zero, one)
        embedding[2*i+1] = embedding[2*i+1] * mask
        embedding[2*i] = embedding[2*i] * mask

    output = model.classifier(embedding)
    Loss = nn.CrossEntropyLoss()
    label = label.to(device)
    loss = Loss(output,label)
    return loss

def loss_mask_1(model,embedding,label,device,threshold=0.1):
    # implement the counterfactual adversarial dropout 
    # version 2: do not use the label of the counterfactual samples
    num = embedding.shape[0]
    index = [2*i for i in range(int(num/2))]
    f_embed = embedding[index]
    f_label = label[index].to(device)

    for i in range(int(num/2)):

        delta = embedding[2*i+1].detach() - embedding[2*i].detach()
        sparse_delta = deepcopy(delta)
        # focus on absolute value
        sparse_delta = abs(sparse_delta)
        zero = torch.zeros_like(sparse_delta)
        one = torch.zeros_like(sparse_delta)
        mask = torch.where(sparse_delta<threshold, zero, one)
        f_embed[i] = f_embed[i] * mask

    output = model.classifier(f_embed)
    Loss = nn.CrossEntropyLoss()
    loss = Loss(output,f_label)
    return loss