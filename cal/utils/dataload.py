'''
Load the input sentences with a manually defined dataset class
'''
from os import sep
import random
import numpy as np
import torch
import pandas as pd

class dataset:

    def __init__(self, paths, task='sa') -> None:
        super().__init__()
        self.task = task
        self.data = pd.read_csv(paths, sep='\t')
        self.Augment = False
        if self.task == 'sa' or self.task == 'nli':
            self.GetItem = self.get_item
        elif self.task == 'sa1':
            self.GetItem = self.get_item_1
        elif self.task == 'sa2':
            self.GetItem = self.get_item_2
        elif self.task == 'nli1':
            self.GetItem = self.get_item_3
        elif self.task == 'nli2':
            self.GetItem = self.get_item_4
        elif self.task == 'nli3':
            self.GetItem = self.get_item_5
        else:
             raise ValueError('Wong task type.')
    
    def get_item(self, index, step=1):

        if self.task == 'sa':
            label = torch.tensor([self.get_label_sa(self.data['Sentiment'][index*step])])
            sentence = [(self.data['Text'][index*step])]

        elif self.task == 'nli':
            label = torch.tensor([self.get_label_nli(self.data['gold_label'][index*step])])
            sentence = [(self.data['sentence1'][index*step],self.data['sentence2'][index*step])]

        return (label,sentence)

    def get_item_1(self, index):
        # read normal train/dev/test and large train/test file under sentiment/orig

        label = torch.tensor([self.get_label_sa(self.data['Sentiment'][index])])
        sentence = [(self.data['Text'][index])]

        return (label,sentence)

    def get_item_2(self, index):
        # read counterfactually augmented train/dev/test file under sentiment/combined/paired
        # the order of the data does not corresponds to that in orig factual data

        label_f = torch.tensor([self.get_label_sa(self.data['Sentiment'][2*index])])
        label_cf = torch.tensor([self.get_label_sa(self.data['Sentiment'][2*index+1])])
        sentence_f = [(self.data['Text'][2*index])]
        sentence_cf = [(self.data['Text'][2*index+1])]
        label = torch.cat([label_f,label_cf])
        sentence = sentence_f + sentence_cf

        return (label,sentence)

    def get_item_3(self, index):
        # read normal train/dev/test file under NLI/original folder

        label = torch.tensor([self.get_label_nli(self.data['gold_label'][index])])
        sentence = [(self.data['sentence1'][index],self.data['sentence2'][index])]

        return (label,sentence)

    def get_item_4(self, index, cf_label):
        # read counterfactually augmented nli file
        # select the sample according to cf_label

        label_cf1 = torch.tensor([self.get_label_nli(self.data['gold_label'][2*index])])
        label_cf2 = torch.tensor([self.get_label_nli(self.data['gold_label'][2*index+1])])
        if label_cf1 == cf_label:
            sentence_cf = [(self.data['sentence1'][2*index],self.data['sentence2'][2*index])]
            return (label_cf1,sentence_cf)
        else:
            sentence_cf = [(self.data['sentence1'][2*index],self.data['sentence2'][2*index+1])]
            return (label_cf2,sentence_cf)

    def get_item_5(self, index, cf_label=None):
        # read counterfactually augmented train/dev/test file under NLI/revised_hypothesis and NLI/revised_premise
        # the order of each data corresponds to the right factual data
        label_cf1 = torch.tensor([self.get_label_nli(self.data['gold_label'][2*index])])
        label_cf2 = torch.tensor([self.get_label_nli(self.data['gold_label'][2*index+1])])
        sentence_cf1 = [(self.data['sentence1'][2*index],self.data['sentence2'][2*index])]
        sentence_cf2= [(self.data['sentence1'][2*index+1],self.data['sentence2'][2*index+1])]
        label = torch.cat([label_cf1,label_cf2])
        sentence = sentence_cf1 + sentence_cf2 

        return (label,sentence)


    def sample_index(self, shuffle):
        size = self.get_size()
        indexs = [i for i in range(size)]
        if shuffle:
            random.shuffle(indexs)
        return indexs

    def get_batch(self, batchsize=1, shuffle=False, indexs=None):

        batch = []
        # create batchs according to the given indexs
        if not indexs:
            Indexs = self.sample_index(shuffle)
            size = self.get_size()
        else:
            size = len(indexs)
            if shuffle:
                # do not shuffle original indexs
                tmp = [i for i in range(size)]
                random.shuffle(tmp)
                Indexs = [indexs[k] for k in tmp]
            else:
                Indexs = indexs
            
        num = 0
        total = 0
        while(total<size):
            if num == 0:
                label,sentence = self.GetItem(Indexs[total])
                num = (num+1)%batchsize
                total += 1
            else:
                new_label,new_sentence = self.GetItem(Indexs[total])
                label = torch.cat([label,new_label])
                sentence = sentence + new_sentence
                num = (num+1)%batchsize
                total += 1
            if num == 0:
                batch.append((label,sentence))
        # dealing with the rest few data
        if num != 0:
            batch.append((label,sentence))

        return batch    

    def concate_batch(self, Datas, batchsize=1, shuffle=False, indexs=None, cf_labels=None):
        """
        concate original data file with its corresponding counterfactual data file in the following order:
        x1_f, x1_cf, x2_f, x2_cf, ...
        Datas is a dataset object contains another data file
        """
        batch = []
        if not indexs:
            raise ValueError('indexs is null while should be given.')
        else:
            size = len(indexs)
            if shuffle:
                tmp = [k for k in range(size)]
                random.shuffle(tmp)
                indexs_shuf = [indexs[k] for k in tmp]
                if cf_labels:
                    cf_labels_shuf = [cf_labels[k] for k in tmp]
                else:
                    cf_labels_shuf = indexs_shuf
                # random.shuffle(indexs)
            else:
                indexs_shuf = indexs
                if cf_labels:
                    cf_labels_shuf = cf_labels
                else:
                    cf_labels_shuf = indexs_shuf

        num = 0
        total = 0
        while(total<size):
            if num == 0:
                label1,sentence1 = self.GetItem(indexs_shuf[total])
                label2,sentence2 = Datas.GetItem(indexs_shuf[total],cf_labels_shuf[total])
                label = torch.cat([label1,label2])
                sentence = sentence1 + sentence2
                num = (num+1)%batchsize
                total += 1
            else:
                new_label1,new_sentence1 = self.GetItem(indexs_shuf[total])
                new_label2,new_sentence2 = Datas.GetItem(indexs_shuf[total],cf_labels_shuf[total])
                label = torch.cat([label,new_label1,new_label2])
                sentence = sentence + new_sentence1 + new_sentence2
                num = (num+1)%batchsize
                total += 1
            if num == 0:
                batch.append((label,sentence))

        if num != 0:
            batch.append((label,sentence))
        
        return batch

    def augmentation(self, input):
        """ 
        Augmenting the input sentence list
        types: masking; cut and rearrange order; repetition
        """
        if self.Augment == False:
            return input

    def get_label_sa(self, text):
        if text == 'Positive':
            return 1
        elif text == 'Negative':
            return 0
        else:
            raise ValueError('text is {}, wrong label text for the Sentiment Analysis task'.format(text))

    def get_label_nli(self, text):
        if text == "neutral":
            return 1
        elif text == "contradiction":
            return 0
        elif text == "entailment":
            return 2
        else:
            raise ValueError('text is {}, wong label text for the NLI data.'.format(text))

    def get_size(self):
        # size of the dataset
        if self.task in ['sa','nli','sa1','nli1']:
            return len(self.data)
        elif self.task in ['sa2','nli3']:
            return int(len(self.data)/2)
        else:
            raise ValueError('text is {}, wrong task file type.'.format(self.task))
        
def remove_blank_label(path='../dataset/NLI/original/snli_1.0_test.txt'):
    data = pd.read_csv(path,sep='\t')
    lens = int(len(data)/3)
    index = []
    for i in range(lens):
        for j in range(3):
            text = data['gold_label'][3*i+j]
            if text == '-':
                index.append(3*i+j)
    data = data.drop(index)
    data.to_csv('../dataset/NLI/original/snli.tsv',index=False,sep='\t')
    

if __name__ == '__main__':
    remove_blank_label()