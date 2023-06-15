import os
import torch
import torch.nn as nn
import random
import numpy as np

def mk_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)

def get_device(ids):
    if torch.cuda.is_available():
        return torch.device('cuda:{}'.format(ids))
    else:
        return torch.device('cpu')

def init_seed(seed=1):
    # make the result reproducible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True    # accelerating
    torch.backends.cudnn.deterministic = True     # more reproducible

def get_cf_label(tokenizer, model, loader, q_idx, scheme, device):

    model.eval()
    label_cf = []
    with torch.no_grad():
        if scheme == 0:
            # choose the class with smallest probability
            for index in q_idx:
                label,sentence = loader.GetItem(index)
                sentence = sentence[0:1]
                encoding = tokenizer(sentence,return_tensors='pt')
                logits = model(encoding["input_ids"].to(device), encoding['attention_mask'].to(device))[0]
                _,target = torch.min(logits,1)
                target = target.cpu()
                if label != target:
                    label_cf.append(target)
                else:
                    _,target = torch.max(logits,1)
                    target = target.cpu()
                    label_cf.append(target)
        else:
            raise ValueError('not a required scheme for counterfactual labeling')

    return label_cf