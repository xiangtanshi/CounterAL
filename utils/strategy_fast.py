import torch
from torch.nn import functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from copy import deepcopy
from scipy.stats import rv_discrete
from sklearn.neighbors import NearestNeighbors

def RamdomSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device):

    return np.random.permutation(un_indexs)[:size]

def LeastConfidenceSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device):

    model.eval()
    max_confidence_list = []
    data_batch = loader.get_batch(32,False,un_indexs)
    with torch.no_grad():
        # index is the absolute order in the original unlabeled set
        count = 0
        for index in range(len(data_batch)):
            sentence = data_batch[index][1]
            encoding = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
            logits = model(encoding["input_ids"].to(device),encoding['attention_mask'].to(device))[0]

            probs = F.softmax(logits,dim=1) + 1e-8
            probs_log = probs.log()
            entropy = -torch.sum(probs * probs_log, dim=1)
            for i in range(entropy.shape[0]):
                max_confidence_list.append((un_indexs[count],float(entropy[i])))
                count += 1
            
    max_confidence_list.sort(key=lambda x:x[1],reverse=True)
    query_idx = []
    for i in range(size):
        query_idx.append(max_confidence_list[i][0])
    return query_idx

def KMeansSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device):

    model.eval()
    embeddings = np.zeros((len(un_indexs),768))
    cluster_learner = KMeans(n_clusters=size)
    data_batch = loader.get_batch(32,False,un_indexs)
    
    with torch.no_grad():
        count = 0
        for index in range(len(data_batch)):
            sentence = data_batch[index][1]
            num = len(sentence)
            encoding = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
            embedding = model.roberta(encoding["input_ids"].to(device),encoding['attention_mask'].to(device)).last_hidden_state[:,0,:].cpu()
            embeddings[count:count+num] = embedding
            count += num

    cluster_learner.fit(embeddings)

    # finding approximate centroid indexs
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    distance = (embeddings - centers)**2
    distance = distance.sum(axis=1)
    query_idx_relative = [] 

    for i in range(size):
        cluster_i = distance[cluster_idxs==i]
        idx = cluster_i.argmin()
        query_idx_relative.append(idx+1)     # get the index of target within a cluster

    # first get the relative index of centroids in un_indexs, then get the target data index in the original unlabeled set
    query_idx_abs  = np.zeros(size)
    query_idx =np.zeros(size)
    for i in range(len(un_indexs)):
        cluster = cluster_idxs[i]
        query_idx_abs[cluster]+=1
        if query_idx_abs[cluster]==query_idx_relative[cluster]:
            query_idx[cluster] = un_indexs[i]

    return query_idx

def CoreSetSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device):

    model.eval()
    un_embeddings = np.zeros((len(un_indexs),768))
    l_embeddings = np.zeros((len(l_indexs),768))
    with torch.no_grad():
        for i,index in enumerate(l_indexs):
            _,sentence = loader.GetItem(index)
            sentence = sentence[0:1]
            encoding = tokenizer(sentence,return_tensors='pt')
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            embedding = model.roberta(input_ids,attention_mask).last_hidden_state[0,0,:].cpu()
            l_embeddings[i] = embedding

        for i,index in enumerate(un_indexs):
            _,sentence = loader.GetItem(index)
            sentence = sentence[0:1]
            encoding = tokenizer(sentence,return_tensors='pt')
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            embedding = model.roberta(input_ids,attention_mask).last_hidden_state[0,0,:].cpu()
            un_embeddings[i] = embedding

    m = np.shape(un_embeddings)[0]

    if np.shape(l_embeddings)[0] == 0:
        # first query, no labeled data
        min_dist = np.tile(float('inf'),m)
    else:
        dist_ctr = pairwise_distances(un_embeddings,l_embeddings)
        min_dist = np.amin(dist_ctr,axis=1)

    query_idx = []
    for i in range(size):
        idx = min_dist.argmax()
        query_idx.append(un_indexs[idx])
        dist_new_ctr = pairwise_distances(un_embeddings,un_embeddings[[idx],:])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j,0])

    return query_idx     

# def BadgeSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device):

#     model_c = deepcopy(model)
#     model_c.eval()
#     shape = model_c.classifier.out_proj.weight.shape

#     # get gradient embedding of the last linear layer for each unlabeled data
#     embeddings = np.zeros([len(un_indexs),shape[0]*shape[1]])
#     for parameters in model_c.parameters():
#         parameters.requires_grad = False
#     model_c.classifier.out_proj.weight.requires_grad = True
    
#     data_batch = loader.get_batch(32,False,un_indexs)
#     count = 0
#     for index in range(len(data_batch)):
#         sentence = data_batch[index][1]
#         num = len(sentence)
#         encoding = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
#         embedding = model.roberta(encoding["input_ids"].to(device),encoding['attention_mask'].to(device)).last_hidden_state[:,0:1,:]
#         for i in range(embedding.shape[0]):
#             outputs = model.classifier(embedding[i:i+1,0:1,:])
#             psuedo_label = torch.zeros(1,dtype=int).to(device)
#             psuedo_label[0] = outputs.argmax()
#             loss = F.cross_entropy(outputs,psuedo_label)
#             loss.backward()
#             gradient = model_c.classifier.out_proj.weight.grad.cpu()
#             gradient = gradient.view(1,-1)
#             embeddings[count + i] = gradient
#             model_c.classifier.out_proj.zero_grad()
        
#         count += num
#     # remove the depicated model from gpu
#     model_c.cpu()

#     # kmeans++ sampling
#     # sequentially sample size centers, assign each data a value which is its distance to the cloest center.
#     ind = np.argmax([np.linalg.norm(s,2) for s in embeddings])
#     mu = [embeddings[ind]] # initialize the first center with the largest gradient
#     indsAll = [ind]
#     while len(mu)<size:
#         if len(mu) == 1:
#             D2 = pairwise_distances(embeddings,mu).ravel().astype(float)
#         else:
#             newD = pairwise_distances(embeddings,[mu[-1]]).ravel().astype(float)
#             # refresh the cloest distance of each data
#             for i in range(len(embeddings)):
#                 if D2[i] > newD[i]:
#                     D2[i] = newD[i]
#         D2 = D2.ravel().astype(float)
#         Ddist = (D2**2)/sum(D2**2)
#         # make a distribution according to the distance, and sample with the distribution
#         customDist = rv_discrete(name='custm', values=(np.arange(len(D2)),Ddist))
#         ind = customDist.rvs(size=1)[0]
#         while ind in indsAll:
#             # do not choose data that is already set as center
#             ind = customDist.rvs(size=1)[0]
#         mu.append(embeddings[ind])
#         indsAll.append(ind)
    
#     query_idx = []
#     for item in indsAll:
#         query_idx.append(un_indexs[item])
    
#     return query_idx

def CALSampling(tokenizer, model, l_indexs, un_indexs, loader, size, device, cf_loader=None):
    # perform contrastive active learning
    num_neighbor = 10
    
    # compute embeddings and probs
    num_class = model.classifier.out_proj.out_features
    model.eval()
    un_embeddings = np.zeros((len(un_indexs),768))
    un_probs = torch.zeros(len(un_indexs),num_class)
    
    if cf_loader:
        data_batch_l = loader.concate_batch(cf_loader,16,False,l_indexs)  # for ablation study
        l_embeddings = np.zeros((len(l_indexs)*3,768))
        l_probs = torch.zeros(len(l_indexs)*3,num_class)
    else:
        data_batch_l = loader.get_batch(32,False,l_indexs)  # for baseline
        l_embeddings = np.zeros((len(l_indexs),768))
        l_probs = torch.zeros(len(l_indexs),num_class)
    
    data_batch_un = loader.get_batch(32,False,un_indexs)
    with torch.no_grad():
        count = 0
        for index in range(len(data_batch_l)):
            sentence = data_batch_l[index][1]
            num = len(sentence)
            encoding = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
            embedding = model.roberta(encoding["input_ids"].to(device),encoding['attention_mask'].to(device)).last_hidden_state[:,0:1,:]
            prob = F.softmax(model.classifier(embedding),dim=1)
            embedding = embedding.cpu()
            prob = prob.cpu()
            l_embeddings[count:count+num] = embedding[:,0,:]
            l_probs[count:count+num] = prob
            count += num
            
        count = 0
        for index in range(len(data_batch_un)):
            sentence = data_batch_un[index][1]
            num = len(sentence)
            encoding = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
            embedding = model.roberta(encoding["input_ids"].to(device),encoding['attention_mask'].to(device)).last_hidden_state[:,0:1,:]
            prob = F.softmax(model.classifier(embedding),dim=1)
            embedding = embedding.cpu()
            prob = prob.cpu()
            un_embeddings[count:count+num] = embedding[:,0,:]
            un_probs[count:count+num] = prob
            count += num
            

    # find the nearest neighbors of each unlabeled sample in the labeled set
    neigh = NearestNeighbors(n_neighbors=num_neighbor)
    neigh.fit(l_embeddings)
    un_knn = neigh.kneighbors(un_embeddings,return_distance=False)
    criterion = torch.nn.KLDivLoss(reduction='sum')
    un_probs = un_probs.log()
    kl_score = np.zeros(len(un_indexs))
    for i in range(len(un_indexs)):
        kl_score[i] = sum([criterion(un_probs[i],l_probs[j]) for j in un_knn[i]])
    ind = np.argsort(kl_score)[::-1]
    query_idx = []
    for i in range(size):
        query_idx.append(un_indexs[ind[i]])
    return query_idx