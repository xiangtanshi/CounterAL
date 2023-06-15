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

def get_args():
    """set up hyper parameters here"""

    # environment and hyperparameters related
    parser = argparse.ArgumentParser(description='parameter setting for active learning in tasks of SA')
    parser.add_argument('--device',default=0,type=int)
    parser.add_argument('--seed',type=str,default=0,choices=['0','1','2','3','4','5','6','7','8','9'])
    parser.add_argument('--op',type=str,default='t_drop_anli')

    parser.add_argument('--epoch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-5,choices=[1e-5,2e-5])
    parser.add_argument('--batchsize',type=int,default=4)
    parser.add_argument('--threshold',type=float,default=0.1)
    parser.add_argument('--task',default='anli')
    parser.add_argument('--tokenizer_path',default='./model/pretrained/roberta-base-tokenizer.pt')
    parser.add_argument('--model_path',default='./model/pretrained/roberta-base-nli.pt')

    # SA related file paths
    parser.add_argument('--train_path_sa',default='./dataset/sentiment/orig/train.tsv')
    parser.add_argument('--train_aug_sa',default='./dataset/sentiment/combined/paired/train_paired.tsv')
    # NLI related file paths
    parser.add_argument('--train_path_nli',default='./dataset/NLI/original/train.tsv')
    parser.add_argument('--train_hyp_aug_nli',default='./dataset/NLI/revised_hypothesis/train.tsv')
    # ANLI related file paths
    parser.add_argument('--train_path_anli',default='./dataset/anli_v1.0/r3_train.tsv')
    parser.add_argument('--train_hyp_aug_anli',default='./dataset/anli_v1.0/r3_train_counter.tsv')

    return parser.parse_args()

class AL:

    """
    the main class for implementing various Active Learning algorithms on sentiment analysis task.
    """
    def __init__(self, args) -> None:

        self.train_ori_sa = dataset(args.train_path_sa,task='sa1')
        self.train_ori_nli = dataset(args.train_path_nli,task='nli1')
        self.train_ori_anli = dataset(args.train_path_anli,task='nli1')

        self.train_aug_sa = dataset(args.train_aug_sa,task='sa2')   
        self.train_hyp_nli = dataset(args.train_hyp_aug_nli,task='nli3')
        self.train_hyp_anli = dataset(args.train_hyp_aug_anli,task='nli3')


        self.tokenizer = torch.load(args.tokenizer_path)
        self.model = torch.load(args.model_path)
        
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.task = args.task
        self.lr = args.lr
        self.batchsize = args.batchsize
        self.threshold = args.threshold
        self.device = get_device(args.device)
        self.seed_pool = [0,501,1001,1501,2001,1,2,3,4,5]
        self.seed = self.seed_pool[int(args.seed)]
        self.epoch = args.epoch
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

    def cf_train_anli(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        train_batch = self.train_ori_anli.get_batch(self.batchsize,shuffle=True)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        # total_steps = len(train_batch) * self.epoch
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()
        print(train_batch[0][0])  # for checking

        for i in range(self.epoch):
            loss_total = 0
            train_batch = self.train_ori_anli.get_batch(self.batchsize,shuffle=True)
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
                # scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            print('epoch:{},ave_loss:{},train_acc:{}'.format(str(i),loss_total/len(train_batch),acc1))
        self.model.cpu()
        torch.save(self.model,'./model/{}_trained/base_{}.pt'.format(self.task,self.task))
                
    def cf_dropout_anli(self):

        self.model = torch.load('./model/anli_trained/base_anli.pt')
        self.model.to(self.device)
        train_batch = self.train_ori_anli.concate_batch(self.train_hyp_anli,1,indexs=[i for i in range(self.train_ori_anli.get_size())])
        self.model.eval()

        remain_dim = []
        pred = []
        threshold = self.threshold

        with torch.no_grad():
            for index in tqdm(range(len(train_batch))):
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                label = train_batch[index][0].to(self.device)

                delta = abs((embedding[1] + embedding[2])/2 - embedding[0])
                zero = torch.zeros_like(delta)
                one = torch.ones_like(delta)
                mask = torch.where(delta<threshold,zero,one)
                remain_dim.append(sum(mask[0]))

                embedding = embedding * mask
                output = self.model.classifier(embedding)
                prob = F.softmax(output[0:1],dim=1)
                pred.append(max(prob[0]))
        
        print(sum(pred)/len(pred))
        print(sum(remain_dim)/len(remain_dim))


    def cf_train_nli(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        train_batch = self.train_ori_nli.get_batch(self.batchsize,shuffle=True)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        # total_steps = len(train_batch) * self.epoch
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()
        print(train_batch[0][0])  # for checking

        for i in range(self.epoch):
            loss_total = 0
            train_batch = self.train_ori_nli.get_batch(self.batchsize,shuffle=True)
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
                # scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            print('epoch:{},ave_loss:{},train_acc:{}'.format(str(i),loss_total/len(train_batch),acc1))
        self.model.cpu()
        torch.save(self.model,'./model/{}_trained/base_{}.pt'.format(self.task,self.task))
                
    def cf_dropout_nli(self):

        self.model = torch.load('./model/nli_trained/base_nli.pt')
        self.model.to(self.device)
        train_batch = self.train_ori_nli.concate_batch(self.train_hyp_nli,1,indexs=[i for i in range(self.train_ori_nli.get_size())])
        self.model.eval()

        remain_dim = []
        pred = []
        threshold = self.threshold

        with torch.no_grad():
            for index in tqdm(range(len(train_batch))):
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                label = train_batch[index][0].to(self.device)

                delta = abs((embedding[1] + embedding[2])/2 - embedding[0])
                zero = torch.zeros_like(delta)
                one = torch.ones_like(delta)
                mask = torch.where(delta>threshold,zero,one)
                remain_dim.append(sum(mask[0]))

                embedding = embedding * mask
                output = self.model.classifier(embedding)
                prob = F.softmax(output[0:1],dim=1)
                pred.append(max(prob[0]))
        
        print(sum(pred)/len(pred))
        print(sum(remain_dim)/len(remain_dim))

    def cf_dropout_sa(self):

        self.model = torch.load('./model/sa_trained/base_sa.pt')
        self.model.to(self.device)
        train_batch = self.train_aug_sa.get_batch(1,shuffle=False)
        self.model.eval()

        remain_dim = []
        pred = []
        threshold = self.threshold

        with torch.no_grad():
            for index in tqdm(range(len(train_batch))):
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                label = train_batch[index][0].to(self.device)

                delta = abs(embedding[1] - embedding[0])
                zero = torch.zeros_like(delta)
                one = torch.ones_like(delta)
                mask = torch.where(delta>threshold,zero,one)
                remain_dim.append(sum(mask[0]))

                embedding = embedding * mask
                output = self.model.classifier(embedding)
                prob = F.softmax(output[0:1],dim=1)
                pred.append(max(prob[0]))

        
        print(sum(pred)/len(pred))
        print(sum(remain_dim)/len(remain_dim))

    def cf_train_sa(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        train_batch = self.train_ori_sa.get_batch(self.batchsize,shuffle=True)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        # total_steps = len(train_batch) * self.epoch
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
        Loss = nn.CrossEntropyLoss()
        print(train_batch[0][0])  # for checking

        for i in range(self.epoch):
            loss_total = 0
            train_batch = self.train_ori_sa.get_batch(self.batchsize,shuffle=True)
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
                # scheduler.step()
                loss_total += loss.item()

            acc1 = self.test(train_batch)
            print('epoch:{},ave_loss:{},train_acc:{}'.format(str(i),loss_total/len(train_batch),acc1))
        self.model.cpu()
        torch.save(self.model,'./model/{}_trained/base_{}.pt'.format(self.task,self.task))

def main():
    args = get_args()
    agent = AL(args)

    if args.op == 't_sa':
        agent.cf_train_sa()
    elif args.op == 't_drop_sa':
        agent.cf_dropout_sa()
    elif args.op == 't_nli':
        agent.cf_train_nli()
    elif args.op == 't_drop_nli':
        agent.cf_dropout_nli()
    elif args.op == 't_anli':
        agent.cf_train_anli()
    elif args.op == 't_drop_anli':
        agent.cf_dropout_anli()

    
if __name__ == '__main__':
    main()