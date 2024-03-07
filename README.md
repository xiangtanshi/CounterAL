## The code and data for implementing CounterAL.

### Project Structure Overview
- All dataset files have been standardized to the .tsv format.
- Pretrained RoBERTa models for each task are stored in the "model" directory.
- I have rewritten a dataloader class in "utils/dataload.py" to meet the requirements of both active learning and counterfactual active learning methods.
- Training code for each task is organized separately in files like "train_xx_xx.py."
- The results are stored in the "record" directory, and I have run CounterAL and each baseline method for one random seed and you can see the results in the corresponding file.
- Python requirements: transformers + pytorch + sklearn

### Instructions for implementing the training code

Meaning of the hyperparameters:
- --op: 'al' means Active Learning, 'ft' means finetuning with the samples acquired by AL, 't' means training with the entire dataset
- --func: it chooses the baseline methods from ('random','lc','kmeans','badge','cal')
- --T: it controls the mode of whether counterfactual samples are used in AL. 0 means only factual samples are used; 1 means we require the human to annotate the counterfactual sample for each queried factual sample; 2 means the unlabeled pool are expanded with all the counterfactual samples, and the baselines are able to query samples from the expanded set.

Downloading the pretrained Roberta models from Huggingface ahead of the training:
```
cd utils
python model.py
```

#### Sentiment Analysis Task

For baselines like Random:
```
python train_sa.py --op al --func random --T 0 --device 0 --seed 0
```
For CounterAL:
```
python train_sa_cal.py --device 0 --seed 0
```

The final queried sample sets are located in the "record/sa/" directory. To assess the quality of the acquired sample sets using each active learning method, you can perform model fine-tuning with these samples as follows:
```
python train_sa.py --op ft --func random --T 0 --device 0 --seed 0
python train_sa_cal.py --op ft --device 0 --seed 0
```

#### Natural Language Inference Task
For baselines like Random:
```
python train_nli.py --op al --func random --T 0 --device 0 --seed 0
```
For CounterAL:
```
python train_nli_cal.py --device 0 --seed 0
```

Similarly, the final queried sample sets are located in the "record/nli" directory. The command to assess the quality of the acquired sample sets are as follows:
```
python train_nli.py --op ft --func random --T 0 --device 0 --seed 0
python train_nli_cal.py --op ft --device 0 --seed 0
```


#### Adversarial Natural Language Inference Task
For baselines like Random:
```
python train_anli.py --op al --func random --T 0 --device 0 --seed 0
```
For CounterAL:
```
python train_anli_cal.py --device 0 --seed 0
```

Similarly, the final queried sample sets are located in the "record/nli" directory. The command to assess the quality of the acquired sample sets are as follows:
```
python train_anli.py --op ft --func random --T 0 --device 0 --seed 0
python train_anli_cal.py --op ft --device 0 --seed 0
```
