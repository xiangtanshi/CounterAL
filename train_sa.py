import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
import numpy as np
from tqdm import tqdm
import copy
from utils.dataload import dataset
from utils.tools import *
from utils.loss_sa import *
from utils.strategy import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    """set up hyper parameters here"""

    # environment and hyperparameters related
    parser = argparse.ArgumentParser(description='parameter setting for active learning in tasks of SA')
    parser.add_argument('--device',default=0,type=int)
    parser.add_argument('--seed',type=str,default=0,choices=['0','1','2','3','4'])
    parser.add_argument('--op',type=str,default='al',help='what function to operate')

    parser.add_argument('--func',type=str,default='random',help='the query strategy that we take')
    parser.add_argument('--T',type=int,default=0,help='decide which unlabeled dataset to choose')

    parser.add_argument('--size',type=int,default=20,help='the number of samples we query each time')
    parser.add_argument('--total',default=10,type=int,help='the total number of query rounds')
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--lr',type=float,default=1e-5,choices=[1e-5,2e-5])
    parser.add_argument('--batchsize',type=int,default=4)
    parser.add_argument('--task',default='sa')
    parser.add_argument('--tokenizer_path',default='./model/pretrained/roberta-base-tokenizer.pt')
    parser.add_argument('--model_path',default='./model/pretrained/roberta-base-sa.pt')

    # SA related file paths
    parser.add_argument('--train_path_sa',default='./dataset/sentiment/orig/train.tsv')
    parser.add_argument('--train_aug_sa',default='./dataset/sentiment/combined/paired/train_paired.tsv')

    parser.add_argument('--large_test_sa',default='./dataset/sentiment/orig/eighty_percent/iid_sa_test.tsv')
    parser.add_argument('--twitter',default='./dataset/sa_ood/twitter_sa_test.tsv')
    parser.add_argument('--amazon',default='./dataset/sa_ood/amazon_balanced.tsv')

    return parser.parse_args()

class SA_AL:

    """
    the main class for implementing various Active Learning algorithms on sentiment analysis task.
    """
    def __init__(self, args) -> None:
        if args.task == 'sa':

            self.train_ori = dataset(args.train_path_sa,task='sa1')
            self.train_aug1 = dataset(args.train_aug_sa,task='sa1')   # query on both factual and cf data
            self.train_aug = dataset(args.train_aug_sa,task='sa2')    # annotate cf when query factual data
            
            self.test_large = dataset(args.large_test_sa,task='sa1')
            self.test_twitter = dataset(args.twitter,task='sa1')
            self.test_amazon = dataset(args.amazon,task='sa1')
            
        else:
            raise ValueError('Task name not expected!')


        self.tokenizer = torch.load(args.tokenizer_path)
        self.model = torch.load(args.model_path)
        
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.task = args.task
        self.lr = args.lr
        self.batchsize = args.batchsize
        self.device = get_device(args.device)
        self.seed_pool = [0,201,501,701,1001]
        self.seed = self.seed_pool[int(args.seed)]
        self.qsize = args.size
        self.qtotal = args.total
        self.epoch = args.epoch
        self.func = args.func
        self.T = args.T
        self.max_length = 512

        if args.func == 'random' or args.func == 'random_all':
            self.strategy = RamdomSampling
        elif args.func == 'lc' or args.func == 'lc_all':
            self.strategy = LeastConfidenceSampling
        elif args.func == 'kmeans' or args.func == 'kmeans_all':
            self.strategy = KMeansSampling
        elif args.func == 'badge' or args.func == 'badge_all':
            self.strategy = BadgeSampling
        elif args.func == 'cal' or args.func == 'cal_all':
            self.strategy = CALSampling
        

    def test(self,test_data,show=False):
        # test the model on test_data
        self.model.eval()
        positive = 0
        total = 0
        with torch.no_grad():            # in case of cuda out of memory: gradients continuously piling could take up lots of gpu space
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

        
    def sa_train(self, loader, batchsize, indexs):
        # train the model on the newly augmented labeled set
        self.model.train()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr) 
        
        l_acc = 0
        # train on labeled data util convergence
        epoch = 0
        while(l_acc<1):
            # create different batch data for each epoch
            epoch += 1
            L_batch = loader.get_batch(batchsize,True,indexs)
            for index in range(len(L_batch)):

                optimizer.zero_grad()
                encoding = self.tokenizer(L_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')

                # standard way
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                outputs = self.model(input_ids,attention_mask)
                labels = L_batch[index][0].to(self.device)
                loss = F.cross_entropy(outputs.logits,labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()

            l_acc = self.test(L_batch)
            

            if epoch>20:
                print('there are some samples that might be outliers which could not be fit, which are as follows:')
                L_batch = loader.get_batch(1,False,indexs)
                self.test(L_batch,show=True)
                break


    def AL_train(self):

        # train a model for multiple rounds with active learning algorithm

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        accuracy = []
        ood_accuracy = []
        l_indexs = []
        if self.T == 0:
            # baseline
            trains = self.train_ori
        elif self.T == 1:
            trains = self.train_aug
        elif self.T == 2:
            # pre-stage
            trains = self.train_aug1  

        un_indexs = [i for i in range(trains.get_size())]
        test_batch = self.test_large.get_batch(16,shuffle=False)
        test_ood_batch = self.test_twitter.get_batch(16,shuffle=False)

        for r in range(-1,self.qtotal):

            # randomly select a subset from the unlabeled pool
            random.shuffle(un_indexs)
            candidates = un_indexs
            # query
            if r < 0:
                q_idx = RamdomSampling(self.tokenizer, self.model, l_indexs, candidates, trains, 20, self.device)
                for item in range(20):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])
            else:
                q_idx = self.strategy(self.tokenizer, self.model, l_indexs, candidates, trains, self.qsize, self.device)
                if r == 0:
                    l_indexs = []
                for item in range(self.qsize):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])

            # update
            self.sa_train(trains, self.batchsize, l_indexs)

            acc = self.test(test_batch)
            ood_acc = self.test(test_ood_batch)

            accuracy.append(acc)
            ood_accuracy.append(ood_acc)
            print('round:{},test_acc:{},ood_acc:{}'.format(r,acc,ood_acc))

        # save the index of total queried data in the end
        selected_idxs = np.zeros(len(l_indexs),dtype=int)
        for i,idx in enumerate(l_indexs):
            selected_idxs[i] = idx
        mk_dir('./record/{}'.format(self.task))
        mk_dir('./record/{}/{}'.format(self.task,self.func))
        np.savetxt('./record/{}/{}/{}.txt'.format(self.task,self.func,self.seed), selected_idxs)
        
        # save all the accuracy in the end
        with open('./record/{}/{}/acc.txt'.format(self.task,self.func),'a') as f:
            f.write('seed:{}\t'.format(self.seed))
            for item in accuracy:
                f.write('{:.4f},'.format(item))
            f.write('\t')
            for item in ood_accuracy:
                f.write('{:.4f},'.format(item))
            f.write('\n')

        print(accuracy,ood_accuracy)

    def finetune_sa(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        indexs = np.loadtxt('./record/{}/{}/{}.txt'.format(self.task,self.func,self.seed),dtype=np.int32).tolist()
        if self.T == 0:
            # baseline
            trains = self.train_ori
        elif self.T == 1:
            # post-stage
            trains = self.train_aug
            indexs = indexs[0:100]
        elif self.T == 2:
            # pre-stage
            trains = self.train_aug1 

        train_batch = trains.get_batch(self.batchsize,True,indexs)
        test_large_batch = self.test_large.get_batch(16,shuffle=False)
        test_ood_batch = self.test_twitter.get_batch(16,shuffle=False)
        test_amazon_batch = self.test_amazon.get_batch(16,shuffle=False)

        print(train_batch[0][0])  # for checking
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
            train_batch = trains.get_batch(self.batchsize,True,indexs)
            self.model.train()
            for index in tqdm(range(len(train_batch))):

                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')

                # standard way
                output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                label = train_batch[index][0].to(self.device)
                loss = Loss(output,label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()
            if i > 6:
                acc1 = self.test(train_batch)
                acc2 = self.test(test_large_batch)
                acc3 = self.test(test_ood_batch)
                acc4 = self.test(test_amazon_batch)
                print('epoch:{},ave loss:{},train acc:{},test acc:{},twitter acc:{},amazon acc:{}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3,acc4))

    def formal_train(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        if self.T == 0:
            trains = self.train_ori
        elif self.T == 1:
            trains = self.train_aug

        train_batch = trains.get_batch(self.batchsize,shuffle=True)
        test_large_batch = self.test_large.get_batch(16,shuffle=False)
        test_ood_batch = self.test_twitter.get_batch(16,shuffle=False)


        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(train_batch) * self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = torch.nn.CrossEntropyLoss()
        print(train_batch[0][0])  # for checking

        for i in range(self.epoch):
            loss_total = 0
            train_batch = trains.get_batch(self.batchsize,shuffle=True)
            self.model.train()
            for index in tqdm(range(len(train_batch))):

                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                
                output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                label = train_batch[index][0].to(self.device)
                loss = Loss(output,label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()
            if i > 5:
                acc1 = self.test(train_batch)
                acc2 = self.test(test_large_batch)
                acc3 = self.test(test_ood_batch)
                print('epoch:{},ave_loss:{},train_acc:{},test_large_acc:{},twitter_acc:{}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3))
        self.model.cpu()
        # torch.save(self.model,'./model/{}_trained/base_{}.pt'.format(self.task,self.task))
                
def main():
    args = get_args()
    agent = SA_AL(args)

    if args.op == 'al':
        agent.AL_train()
    elif args.op == 'ft':
        agent.finetune_sa()
    elif args.op == 't':
        agent.formal_train()
    
if __name__ == '__main__':
    main()