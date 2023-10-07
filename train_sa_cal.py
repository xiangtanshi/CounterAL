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
import warnings
warnings.filterwarnings('ignore')

def get_args():
    """set up hyper parameters here"""

    # environment and hyperparameters related
    parser = argparse.ArgumentParser(description='parameter setting for active learning in tasks of SA')
    parser.add_argument('--device',default=0,type=int)
    parser.add_argument('--seed',type=str,default=0,choices=['0','1','2','3','4'])
    parser.add_argument('--op',type=str,default='al',help='what function to operate')

    parser.add_argument('--func',type=str,default='CounterAL',help='the query strategy that we take')
    parser.add_argument('--T',type=int,default=1,help='decide which unlabeled dataset to choose')

    parser.add_argument('--size',type=int,default=10,help='the number of factual samples we query each round')
    parser.add_argument('--total',default=10,type=int,help='the total number of query rounds')
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batchsize',type=int,default=2)
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

    def CounterALSampling(self, un_indexs, loader):

        variance = []
        for index in un_indexs:
            item = self.confidence[index]
            p0 = [x[0] for x in item]
            p1 = [x[1] for x in item]
            var0 = np.var(p0)
            var1 = np.var(p1)
            variance.append([index,max(var0,var1)])

        variance.sort(key=lambda x:x[1],reverse=True)

        # diversified
        candidates = [i[0] for i in variance[0:200]]
        query_idx = KMeansSampling(self.tokenizer, self.model, None, candidates, loader, self.qsize, self.device)
        # query_idx = RamdomSampling(self.tokenizer, self.model, None, candidates, loader, self.qsize, self.device)

        # max variability
        # candidates = variance[0:self.qsize]
                
        return query_idx

        
    def sa_train(self, loader, batchsize, indexs, un_indexs):
        # train the model on the newly augmented labeled set
        self.model.train()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr) 
        
        l_acc = 0
        epoch = 0
        end = 0
        while(end == 0):
            # create different batch data for each epoch
            epoch += 1
            L_batch = loader.get_batch(batchsize,True,indexs)
            self.model.train()
            for index in range(len(L_batch)):

                optimizer.zero_grad()
                encoding = self.tokenizer(L_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')

                if epoch<2:
                    # accelerating the training
                    input_ids = encoding["input_ids"].to(self.device)
                    attention_mask = encoding["attention_mask"].to(self.device)
                    outputs = self.model(input_ids,attention_mask)
                    labels = L_batch[index][0].to(self.device)
                    loss = F.cross_entropy(outputs.logits,labels)
                    loss.backward()
                else:
                    embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                    label = L_batch[index][0]
                    loss = loss_mask(self.model, embedding, label, self.device, threshold=0.1)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()

            l_acc = self.test(L_batch)
            if l_acc == 1:
                end = 1

            if  epoch>20:
                print('there are some samples that might be outliers which could not be fit, which are as follows:')
                L_batch = loader.get_batch(1,False,indexs)
                self.test(L_batch,show=True)
                end = 1

            if end:
                self.model.eval()
                c = 0
                data_batch = self.train_ori.get_batch(32,False,un_indexs)
                with torch.no_grad():
                    for i in range(len(data_batch)):
                        sentence = data_batch[i][1]
                        num = len(sentence)
                        encoding = self.tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
                        logits = self.model(encoding["input_ids"].to(self.device),encoding['attention_mask'].to(self.device))[0]
                        logits = logits.cpu()
                        probs = F.softmax(logits,dim=1) + 1e-8
                        for j in range(num):
                            self.confidence[un_indexs[c+j]].append(probs[j])
                        c += num

    def sa_train_start(self, loader, indexs, un_indexs, batchsize=4):
        # train the model on the newly augmented labeled set
        self.model.train()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        l_acc = 0
        epoch = 0
        count = 0
        end = 0
        while(end == 0):
            # create different batch data for each epoch
            epoch += 1
            L_batch = loader.get_batch(batchsize,True,indexs)
            self.model.train()
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

            l_acc_pre = l_acc
            l_acc = self.test(L_batch)

            if (l_acc>0.7 and l_acc_pre<0.7 and count == 0) or (l_acc>0.9 and l_acc_pre<0.9 and count == 1) or l_acc == 1:
                count += 1
                self.model.eval()
                c = 0
                data_batch = self.train_ori.get_batch(32,False,un_indexs)
                with torch.no_grad():
                    for i in range(len(data_batch)):
                        sentence = data_batch[i][1]
                        num = len(sentence)
                        encoding = self.tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors='pt')
                        logits = self.model(encoding["input_ids"].to(self.device),encoding['attention_mask'].to(self.device))[0]
                        logits = logits.cpu()
                        probs = F.softmax(logits,dim=1) + 1e-8
                        for j in range(num):
                            self.confidence[un_indexs[c+j]].append(probs[j])
                        c += num
                if l_acc == 1:
                    end = 1


    def AL_train(self):

        # train a model for multiple rounds with active learning algorithm

        init_seed(self.seed)
        # self.model.classifier.apply(weight_init)
        self.model_init = deepcopy(self.model)  # reserve this inital model for retraining in case of special condition
        self.model.to(self.device)

        accuracy = []
        ood_accuracy = []
        l_indexs = []
        if self.T == 1:
            trains = self.train_aug
        else:
            raise ValueError('wrong dataset input!')

        un_indexs = [i for i in range(trains.get_size())]
        test_batch = self.test_large.get_batch(16,shuffle=False)
        test_ood_batch = self.test_twitter.get_batch(16,shuffle=False)

        self.confidence = [[] for i in range(trains.get_size())]

        for r in range(-1,self.qtotal):

            # randomly select a subset from the unlabeled pool
            random.shuffle(un_indexs)
            candidates = un_indexs
            # query
            if r < 0:
                q_idx = RamdomSampling(self.tokenizer, self.model, l_indexs, candidates, trains, self.qsize, self.device)
                for item in range(self.qsize):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])
            else:
                q_idx = self.CounterALSampling(candidates, trains)
                if r == 0:
                    l_indexs = []
                for item in range(self.qsize):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])

            # update
            if r == -1:
                self.sa_train_start(trains, l_indexs, un_indexs, batchsize=self.batchsize)
            else:
                self.sa_train(trains, self.batchsize, l_indexs, un_indexs)

            acc = self.test(test_batch)
            ood_acc = self.test(test_ood_batch)

            accuracy.append(acc)
            ood_accuracy.append(ood_acc)
            print('round:{},test_acc:{},ood_test_acc:{}'.format(r,acc,ood_acc))

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
        # self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        indexs = np.loadtxt('./record/{}/{}/{}.txt'.format(self.task,self.func,self.seed),dtype=np.int32).tolist()

        if self.T == 1:
            trains = self.train_aug
        else:
            raise ValueError('wrong dataset input!')

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

                if i <10:
                    output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                    label = train_batch[index][0].to(self.device)
                    loss = Loss(output,label)
                    loss.backward()
                else:
                    embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                    label = train_batch[index][0]
                    loss = loss_mask(self.model, embedding, label, self.device, threshold=0.1)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()
            if i>4:
                acc1 = self.test(train_batch)
                acc2 = self.test(test_large_batch)
                acc3 = self.test(test_ood_batch)
                acc4 = self.test(test_amazon_batch)
                print('epoch:{},ave loss:{},train acc:{},test acc:{},twitter acc:{},amazon acc:{}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3,acc4))

                
def main():
    args = get_args()
    agent = SA_AL(args)

    if args.op == 'al':
        agent.AL_train()
    elif args.op == 'ft':
        agent.finetune_sa()
    
if __name__ == '__main__':
    main()