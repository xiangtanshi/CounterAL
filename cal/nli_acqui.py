import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
import numpy as np
from tqdm import tqdm
import copy
import random
from utils.dataload import dataset
from utils.tools import *
from utils.loss_nli import *
from utils.strategy import *

def get_args():
    """set up hyper parameters here"""

    # environment and hyperparameters related
    parser = argparse.ArgumentParser(description='parameter setting for active learning in tasks of NLI')
    parser.add_argument('--device',default=0,type=int)
    parser.add_argument('--seed',type=str,default=0,choices=['0','1','2','3','4','5','6','7','8','9'])
    parser.add_argument('--cf',type=int,default=0,help='whether annotating counterfactual samples')
    parser.add_argument('--cf_multi',type=int,default=0,help='whether annotate more than one counterfactual samples')

    parser.add_argument('--T',type=int,default=0,help='decide which unlabeled dataset to choose')
    parser.add_argument('--op',type=str,default='dt',help='what function to operate')
    parser.add_argument('--border',type=int,default=10)
    parser.add_argument('--func',type=str,default='random',help='the query strategy that we take')

    parser.add_argument('--scheme',type=int,default=0,choices=[0,1,2],help='the strategy which decides the target label for counterfactual samples')
    parser.add_argument('--size',type=int,default=16,help='the number of samples we query each time')
    parser.add_argument('--total',default=30,type=int,help='the total number of query rounds')
    parser.add_argument('--epoch',type=int,default=10,help='10 for sa, 15 for nli')
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batchsize',type=int,default=4)
    parser.add_argument('--task',default='nli')
    parser.add_argument('--tokenizer_path',default='./model/pretrained/roberta-base-tokenizer.pt')
    parser.add_argument('--model_path',default='./model/pretrained/roberta-base-nli.pt')

    # NLI related file paths
    parser.add_argument('--train_path_nli',default='./dataset/NLI/original/train.tsv')
    parser.add_argument('--test_path_nli',default='./dataset/NLI/original/test.tsv')
    parser.add_argument('--train_hyp_aug_nli',default='./dataset/NLI/revised_hypothesis/train.tsv')
    parser.add_argument('--test_hyp_aug_nli',default='./dataset/NLI/revised_hypothesis/test.tsv')
    parser.add_argument('--train_all',default='./dataset/NLI/revised_hypothesis/train_all.tsv')

    parser.add_argument('--large_test_nli',default='./dataset/NLI/original/large_test.tsv')
    parser.add_argument('--anli_test',default='./dataset/anli_v1.0/ood_set.tsv')

    return parser.parse_args()

class NLI_AL:
    """the main class for implementing various Active Learning algorithm"""
    def __init__(self, args) -> None:
        if args.task == 'nli':
           
            self.train_ori = dataset(args.train_path_nli,task='nli1')
            self.train_all = dataset(args.train_all,task='nli1')
            # self.test_ori = dataset(args.test_path_nli,task='nli1')
            self.test_large = dataset(args.large_test_nli,task='nli1')
            self.test_ood = dataset(args.anli_test,task='nli1')
            # task = nli3,nli4 means annotate all counterfactual samples, nli2 means annotate 1
            self.train_hyp = dataset(args.train_hyp_aug_nli,task='nli2')
            # self.test_hyp = dataset(args.test_hyp_aug_nli,task='nli2')
            self.train_hyp_all = dataset(args.train_hyp_aug_nli,task='nli3')

        else:
            raise ValueError('Task name not expected!')


        self.tokenizer = torch.load(args.tokenizer_path)
        self.model = torch.load(args.model_path)
        
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.task = args.task
        self.cf = args.cf
        self.cf_multi = args.cf_multi
        self.lr = args.lr
        self.batchsize = args.batchsize
        self.device = get_device(args.device)
        self.seed_pool = [0,501,1001,1501,2001,1,2,3,4,5]
        self.seed = self.seed_pool[int(args.seed)]
        self.qsize = args.size
        self.qtotal = args.total
        self.epoch = args.epoch
        self.scheme = args.scheme
        self.func = args.func
        self.T = args.T
        self.border = args.border
        

    def test(self,test_data,show=False):
        # test the model on test_data
        self.model.eval()
        positive = 0
        total = 0
        with torch.no_grad():            # in case of cuda out of memory: gradients continuously piling could take up lots of space
            for index in range(len(test_data)):
                label = test_data[index][0].to(self.device)
                encoding = self.tokenizer(test_data[index][1],padding=True,truncation=True,return_tensors='pt')
                logits = self.model(encoding["input_ids"].to(self.device),encoding['attention_mask'].to(self.device))[0]
                _,predict = torch.max(logits,1)
                total += label.size(0)
                cur_pos = (predict==label).sum().item()
                positive += cur_pos
                if show and label.size(0) != cur_pos:
                    print(test_data[index][1][0][0:20])

        return positive/total


    def formal_train(self):
        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        if self.T == 0:
            train_batch = self.train_ori.get_batch(self.batchsize,True)
        elif self.T == 1:
            train_batch = self.train_all.get_batch(self.batchsize,True)
        elif self.T == 2:
            train_batch = self.train_ori.concate_batch(self.train_hyp_all,self.batchsize,True,indexs=[i for i in range(self.train_ori.get_size())])

        test_large_batch = self.test_large.get_batch(4,shuffle=False)
        test_ood_batch = self.test_ood.get_batch(4,shuffle=False)      

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(train_batch) * self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()

        border = self.border

        for i in range(self.epoch):
            loss_total = 0
            
            if self.T == 0:
                train_batch = self.train_ori.get_batch(self.batchsize,True)
            elif self.T == 1:
                train_batch = self.train_all.get_batch(self.batchsize,True)
            elif self.T == 2:
                train_batch = self.train_ori.concate_batch(self.train_hyp_all,self.batchsize,True,indexs=[i for i in range(self.train_ori.get_size())])

            self.model.train()

            for index in tqdm(range(len(train_batch))):
                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')

                if i>border:

                    embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                    label = train_batch[index][0]
                    loss = loss_reweight_1(self.model, embedding, label, self.device)
                    loss.backward()
                else:

                    output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                    label = train_batch[index][0].to(self.device)
                    loss = Loss(output,label)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            acc2 = self.test(test_large_batch)
            acc3 = self.test(test_ood_batch)
            print('epoch:{},ave_loss:{:.5f},train_acc:{:.4f},test_large_acc:{:.4f},test_ood_acc:{:.4f}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3))
        self.model.cpu()
        torch.save(self.model,'./model/{}_trained/base_{}.pt'.format(self.task,self.task))
            
    def margin_order(self):

        self.model = torch.load('./model/nli_trained/base_nli.pt')
        self.model.to(self.device)
        self.model.eval()

        train_batch = self.train_ori.concate_batch(self.train_hyp_all,1,False,indexs=[i for i in range(self.train_ori.get_size())])
        test_batch = self.test_large.get_batch(4,shuffle=False)

        margin_dict = []
        with torch.no_grad():
            length = len(train_batch)
            for i in range(length):
                encoding = self.tokenizer(train_batch[i][1],padding=True,truncation=True,max_length=512,return_tensors='pt')
                embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                cos_dist1 = 1 - embedding[0,0]@embedding[1,0]/torch.norm(embedding[0,0])/torch.norm(embedding[1,0])
                cos_dist2 = 1 - embedding[0,0]@embedding[2,0]/torch.norm(embedding[0,0])/torch.norm(embedding[2,0])
                margin_dict.append([i,min(cos_dist1.cpu(),cos_dist2.cpu())])

        margin_dict.sort(key=lambda x:x[1])
        np_dict = np.zeros((len(margin_dict),2))
        for i in range(len(margin_dict)):
            np_dict[i][0] = margin_dict[i][0]
            np_dict[i][1] = margin_dict[i][1]
        np.savetxt('./record/stats/nli_margin.txt', np_dict)

        acc = self.test(test_batch)
        print(acc)


    def margin_train(self):

        init_seed(self.seed)
        margin_dict = np.loadtxt('./record/stats/nli_margin.txt').tolist()

        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        indexs = []
        for i in range(400):
            indexs.append(int(margin_dict[i][0]))
        random.shuffle(indexs)
        t_index = indexs[0:240]
        
        train_batch = self.train_ori.concate_batch(self.train_hyp_all,self.batchsize,True,t_index)
        test_large_batch = self.test_large.get_batch(4,shuffle=False)
        test_ood_batch = self.test_ood.get_batch(4,shuffle=False)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(train_batch) * self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()

        for i in range(self.epoch):
            loss_total = 0
            self.model.train()
            for index in tqdm(range(len(train_batch))):

                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')
                
                output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                label = train_batch[index][0].to(self.device)
                loss = Loss(output,label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            acc2 = self.test(test_large_batch)
            acc3 = self.test(test_ood_batch)
            print('epoch:{},ave loss:{},train acc:{:.4f},test acc:{:.4f},test ood acc:{:.4f}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3))

    def datamap(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        train_batch = self.train_ori.get_batch(1,True)
        test_batch = self.test_large.get_batch(4,False)

        confidence_list = [[] for i in range(len(train_batch))]
        variance_list = np.zeros((len(train_batch),2))

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(train_batch) * self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()

        for i in range(self.epoch):
            self.model.train()

            train_batch = self.train_ori.get_batch(self.batchsize,True)
            for index in tqdm(range(len(train_batch))):
                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')

                output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                label = train_batch[index][0].to(self.device)
                loss = Loss(output,label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
            acc1 = self.test(train_batch)
            acc2 = self.test(test_batch)
            print('epoch:{},train_acc:{:.4f},test_large_acc:{:.4f}'.format(str(i),acc1,acc2))

            train_batch = self.train_ori.get_batch(1,False)
            self.model.eval()
            with torch.no_grad():
                for index in range(len(train_batch)):
                    label = train_batch[index][0]
                    encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,return_tensors='pt')
                    logits = self.model(encoding["input_ids"].to(self.device),encoding['attention_mask'].to(self.device))[0]
                    probability = F.softmax(logits).cpu()
                    confidence_list[index].append(probability[0,label])
        
        for i,item in enumerate(confidence_list):
            variance_list[i,0]=i
            variance_list[i,1]=np.var(item)

        np.savetxt('./record/stats/nli_datamap.txt', variance_list)

    def datamap_train(self):

        init_seed(self.seed)
        datamap_dict = np.loadtxt('./record/stats/nli_datamap.txt').tolist()

        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        indexs = []
        datamap_dict.sort(key=lambda x:x[1])
        for i in range(400):
            indexs.append(int(datamap_dict[-i][0]))
        random.shuffle(indexs)
        t_index = indexs[0:240]

        train_batch = self.train_ori.concate_batch(self.train_hyp_all,self.batchsize,True,t_index)
        test_large_batch = self.test_large.get_batch(4,shuffle=False)
        test_ood_batch = self.test_ood.get_batch(4,shuffle=False)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(train_batch) * self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()

        for i in range(self.epoch):
            loss_total = 0
            self.model.train()
            for index in tqdm(range(len(train_batch))):

                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')
                
                output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                label = train_batch[index][0].to(self.device)
                loss = Loss(output,label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            acc2 = self.test(test_large_batch)
            acc3 = self.test(test_ood_batch)
            print('epoch:{},ave loss:{},train acc:{:.4f},test acc:{:.4f},test ood acc:{:.4f}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3))



def main():
    args = get_args()
    agent = NLI_AL(args)

    if args.op == 't':
        agent.formal_train()
    elif args.op == 'm':
        agent.margin_order()
    elif args.op == 'mt':
        agent.margin_train()
    elif args.op == 'd':
        agent.datamap()
    elif args.op == 'dt':
        agent.datamap_train()

if __name__ == '__main__':
    main()