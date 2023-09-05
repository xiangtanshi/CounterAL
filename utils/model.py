from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tools import mk_dir

def main(name_index=0):

    name_list = ["roberta-base","roberta-large","roberta-large-mnli","distilroberta-base"]
    name = name_list[name_index]
    tokenizer = AutoTokenizer.from_pretrained(name)
    # model class not started with TF is pytorch module, other wise is Tensorflow module
    # load pytorch model: from_pt=True, load tensorflow model: from_tf=True
    sa_model = AutoModelForSequenceClassification.from_pretrained(name, num_labels = 2)
    nli_model = AutoModelForSequenceClassification.from_pretrained(name, num_labels = 3)
    
    mk_dir('../model')
    mk_dir('../model/pretrained')
    mk_dir('../model/sa_trained')
    mk_dir('../model/nli_trained')
    mk_dir('../model/anli_trained')
    # save the pretrained model to 
    pre_path = '../model/pretrained/' + name
    torch.save(tokenizer, pre_path + '-tokenizer.pt')
    torch.save(sa_model, pre_path + '-sa.pt')
    torch.save(nli_model, pre_path + '-nli.pt')

if __name__ == '__main__':
    idx = 0
    main(idx)