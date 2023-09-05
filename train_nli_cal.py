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
from utils.loss_nli import *
from utils.strategy import BadgeSampling,RamdomSampling
from utils.strategy_fast import CALSampling, KMeansSampling

def get_args():
    """set up hyper parameters here"""

    # environment and hyperparameters related
    parser = argparse.ArgumentParser(description='parameter setting for active learning in tasks of NLI')
    parser.add_argument('--device',default=0,type=int)
    parser.add_argument('--seed',type=str,default=0,choices=['0','1','2','3','4'])
    parser.add_argument('--cf',type=int,default=1,help='whether annotating counterfactual samples')

    parser.add_argument('--T',type=int,default=0,help='decide which unlabeled dataset to choose')
    parser.add_argument('--op',type=str,default='al',help='what function to operate')
    parser.add_argument('--func',type=str,default='CounterAL',help='the query strategy that we take')

    parser.add_argument('--size',type=int,default=16,help='the number of samples we query each time')
    parser.add_argument('--total',default=10,type=int,help='the total number of query rounds')
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batchsize',type=int,default=2)
    parser.add_argument('--task',default='nli')
    parser.add_argument('--tokenizer_path',default='./model/pretrained/roberta-base-tokenizer.pt')
    parser.add_argument('--model_path',default='./model/pretrained/roberta-base-nli.pt')

    # NLI related file paths
    parser.add_argument('--train_path_nli',default='./dataset/NLI/original/train.tsv')
    parser.add_argument('--train_hyp_aug_nli',default='./dataset/NLI/revised_hypothesis/train.tsv')
    parser.add_argument('--train_all',default='./dataset/NLI/revised_hypothesis/train_all.tsv')

    parser.add_argument('--nli_iid_test',default='./dataset/NLI/original/large_test.tsv')
    parser.add_argument('--nli_ood_test',default='./dataset/NLI/original/ood_set.tsv')
    
    parser.add_argument('--AT',default='./dataset/nli_stress_test/AT.tsv')
    parser.add_argument('--LN',default='./dataset/nli_stress_test/LN.tsv')
    parser.add_argument('--NG',default='./dataset/nli_stress_test/NG.tsv')
    parser.add_argument('--NR',default='./dataset/nli_stress_test/NR.tsv')
    parser.add_argument('--SE',default='./dataset/nli_stress_test/SE.tsv')
    parser.add_argument('--WO',default='./dataset/nli_stress_test/WO.tsv')

    return parser.parse_args()

class NLI_AL:
    """the main class for implementing various Active Learning algorithm"""
    def __init__(self, args) -> None:
        if args.task == 'nli':
           
            self.train_ori = dataset(args.train_path_nli,task='nli1')
            self.train_all = dataset(args.train_all,task='nli1')
            self.test_large = dataset(args.nli_iid_test,task='nli1')
            self.test_ood = dataset(args.nli_ood_test,task='nli1')
            self.train_hyp_all = dataset(args.train_hyp_aug_nli,task='nli3')
            
            self.AT = dataset(args.AT,task='nli1')
            self.LN = dataset(args.LN,task='nli1')
            self.NG = dataset(args.NG,task='nli1')
            self.NR = dataset(args.NR,task='nli1')
            self.SE = dataset(args.SE,task='nli1')
            self.WO = dataset(args.WO,task='nli1')

        else:
            raise ValueError('Task name not expected!')


        self.tokenizer = torch.load(args.tokenizer_path)
        self.model = torch.load(args.model_path)
        
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.task = args.task
        self.cf = args.cf
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

    def CounterALSampling(self, un_indexs, loader):
            
            score = []
            for index in un_indexs:
                item = self.confidence[index]
                p0 = [x[0] for x in item]
                p1 = [x[1] for x in item]
                p2 = [x[2] for x in item]
                var0 = np.var(p0)
                var1 = np.var(p1)
                var2 = np.var(p2)
                max_var = max(var0,var1,var2)
                score.append([index,max_var])

            score.sort(key=lambda x:x[1],reverse=True)
            
            # diversified
            candidates = [i[0] for i in score[0:200]]
            query_idx = KMeansSampling(self.tokenizer, self.model, None, candidates, loader, self.qsize, self.device)
            # query_idx = RamdomSampling(self.tokenizer, self.model, None, candidates, loader, self.qsize, self.device)

            return query_idx

    def nli_train(self, loader, batchsize, indexs, un_indexs, cf_aug=False, cf_loader=None):
        # train the model on the newly augmented labeled set in a round
        # task is NLI
        self.model.train()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        l_acc = 0
        # train on labeled data util convergence
        repeat = 0
        epoch = 0
        end = 0
        while(end == 0):
            # create different batch data for each epoch
            epoch += 1
            if cf_aug:
                L_batch = loader.concate_batch(cf_loader,batchsize,True,indexs)
            else:
                L_batch = loader.get_batch(batchsize,True,indexs)

            self.model.train()
            for index in range(len(L_batch)):

                optimizer.zero_grad()
                encoding = self.tokenizer(L_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')

                # if l_acc < 0.8:
                if l_acc < 1:
                    #standard way
                    outputs = self.model(encoding["input_ids"].to(self.device),encoding["attention_mask"].to(self.device))
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
            l_acc_pre = l_acc
            l_acc = self.test(L_batch)
            if l_acc == 1:
                end = 1

            if l_acc == l_acc_pre and 1>l_acc>0.98:
                # in case there are outliers or wrong counterfactual samples that could not be fit 
                repeat += 1
                
            else:
                repeat = 0

            if repeat>4:
                print('there are some samples that might be outliers which could not be fit.')
                if cf_aug:
                    L_batch = loader.concate_batch(cf_loader,1,False,indexs)
                else:
                    L_batch = loader.get_batch(1,False,indexs)
                self.test(L_batch,show=True)
                end = 1
            
            if end:
                self.model.eval()
                c = 0
                data_batch = self.train_ori.get_batch(64,False,un_indexs)
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

    def nli_train_start(self, loader, batchsize, indexs, un_indexs, cf_aug=False, cf_loader=None):
        # train the model on the newly augmented labeled set in the first round
        # record the result at three checkpoints
  
        self.model.train()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        l_acc = 0
        # train on labeled data util convergence
        epoch = 0
        count = 0
        end = 0
        while(end == 0):
            # create different batch data for each epoch
            epoch += 1
            if cf_aug:
                L_batch = loader.concate_batch(cf_loader,batchsize,True,indexs)
            else:
                L_batch = loader.get_batch(batchsize,True,indexs)

            self.model.train()
            for index in range(len(L_batch)):

                optimizer.zero_grad()
                encoding = self.tokenizer(L_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')

                #standard way
                outputs = self.model(encoding["input_ids"].to(self.device),encoding["attention_mask"].to(self.device))
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
                data_batch = self.train_ori.get_batch(64,False,un_indexs)
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
        
        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model_init = deepcopy(self.model)   # for restart
        self.model.to(self.device)

        accuracy = []
        ood_accuracy = []
        l_indexs = []
        if self.T == 0:
            trains = self.train_ori
        elif self.T == 1:
            trains = self.train_all

        un_indexs = [i for i in range(trains.get_size())]
        test_batch = self.test_large.get_batch(32,shuffle=False)
        test_ood_batch = self.test_ood.get_batch(32,shuffle=False)

        self.confidence = [[] for i in range(trains.get_size())]

        for r in range(0,self.qtotal):  #if in ablation study for cal and badge, add -1 epoch for starting
            
            # randomly select a subset from the unlabeled pool
            random.shuffle(un_indexs)
            candidates = un_indexs
            # query
            if r < 1:
                q_idx = RamdomSampling(self.tokenizer, self.model, l_indexs, candidates, trains, self.qsize, self.device)
                for item in range(self.qsize):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])
            else:
                q_idx = self.CounterALSampling(candidates, trains)
                # q_idx = CALSampling(self.tokenizer, self.model, l_indexs, candidates, trains, self.qsize, self.device, self.train_hyp_all)
                # q_idx = BadgeSampling(self.tokenizer, self.model, l_indexs, candidates, trains, self.qsize, self.device)
                if r == 0:
                    l_indexs = []
                for item in range(self.qsize):
                    l_indexs.append(q_idx[item])
                    un_indexs.remove(q_idx[item])
            # reset the model and train from start every 3 rounds
            if r%4 == 3:
                self.model.cpu()
                self.model = deepcopy(self.model_init)
                self.model.to(self.device)
            # update
            if r == 0:
                self.nli_train_start(trains, self.batchsize, l_indexs, un_indexs, cf_aug=self.cf, cf_loader=self.train_hyp_all) # for CounterAL
                # self.nli_train_start(trains, 4, l_indexs, un_indexs, cf_aug=False, cf_loader=None)   # for cal and badge
            else:
                self.nli_train(trains, self.batchsize, l_indexs, un_indexs, cf_aug=self.cf, cf_loader=self.train_hyp_all)

            acc = self.test(test_batch)
            ood_acc = self.test(test_ood_batch)

            accuracy.append(acc)
            ood_accuracy.append(ood_acc)
            print('round:{},test_acc:{},ood_acc:{}'.format(r,acc,ood_acc))

        # save the index of total queried data in the end
        selected_idxs = np.zeros(len(l_indexs))
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

    def finetune_nli(self):

        init_seed(self.seed)
        self.model.classifier.apply(weight_init)
        self.model.to(self.device)

        indexs = np.loadtxt('./record/{}/{}/{}.txt'.format(self.task,self.func,self.seed),dtype=np.int32).tolist()

        if self.T == 0:
            self.trains = self.train_ori
        elif self.T == 1:
            self.trains = self.train_all


        train_batch = self.trains.concate_batch(self.train_hyp_all,self.batchsize,True,indexs)
        test_large_batch = self.test_large.get_batch(32,shuffle=False)
        test_ood_batch = self.test_ood.get_batch(32,shuffle=False)

        AT = self.AT.get_batch(32,shuffle=False)
        LN = self.LN.get_batch(32,shuffle=False)
        NG = self.NG.get_batch(32,shuffle=False)
        NR = self.NR.get_batch(32,shuffle=False)
        SE = self.SE.get_batch(32,shuffle=False)
        WO = self.WO.get_batch(32,shuffle=False)

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
            
            train_batch = self.trains.concate_batch(self.train_hyp_all,self.batchsize,True,indexs)

            self.model.train()
            for index in tqdm(range(len(train_batch))):

                optimizer.zero_grad()
                encoding = self.tokenizer(train_batch[index][1],padding=True,truncation=True,max_length=512,return_tensors='pt')

                if i>20:
                    # finetune
                    embedding = self.model.roberta(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)).last_hidden_state[:,:1,:]
                    label = train_batch[index][0]
                    loss = loss_mask(self.model, embedding, label, self.device)
                    # loss = loss_mask_rsc(self.model, embedding, label, self.device)
                    loss.backward()

                else:
                    # train with cross entropy loss
                    output = self.model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device))[0]
                    label = train_batch[index][0].to(self.device)
                    loss = Loss(output,label)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1,norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item()
            if i > 8:
                acc1 = self.test(train_batch)
                acc2 = self.test(test_large_batch)
                acc3 = self.test(test_ood_batch)
                
                # acc4 = self.test(AT)
                # acc5 = self.test(LN)
                # acc6 = self.test(NG)
                # acc7 = self.test(NR)
                # acc8 = self.test(SE)
                # acc9 = self.test(WO)
                # print('epoch:{},ave loss:{},train acc:{:.4f},test acc:{:.4f},test ood acc:{:.4f},AT:{:.4f},LN:{:.4f},NG:{:.4f},NR:{:.4f},SE:{:.4f},WO:{:.4f}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9))
                print('epoch:{},ave loss:{},train acc:{:.4f},test acc:{:.4f},test ood acc:{:.4f}'.format(str(i),loss_total/len(train_batch),acc1,acc2,acc3))


def main():
    args = get_args()
    agent = NLI_AL(args)

    if args.op == 'al':
        agent.AL_train()
    elif args.op == 'ft':
        agent.finetune_nli()

if __name__ == '__main__':
    main()