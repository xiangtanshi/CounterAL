## Counterfactual Active Learning for Out-of-Distribution Generalization
The code is provided for implementing the baselines and CAL algorithm.

### Environments
these python packages are necessary:

python3 (3.8.5)

pytorch==1.7.1 (cuda==11.0)

transformers==4.18.0

scikit-learn==1.0.2

scipy==1.6.0

numpy==1.19.5

pandas==1.2.1

### Tree structure of the project
- cal
  - dataset
    - sentiment
    - NLI
    - anli_v1.0
    - twitter
  - model
    - pretrained
    - sa_trained
    - nli_trained
    - anli_trained
  - record
    - sa
    - nli
    - anli
    - stats
  - utils
    - dataload.py
    - ...
  - training and evaluating python file
  
 ### Commands for implementing the project
 
 ### Preparation work:
 1. Download the RoBERTa model to the model/pretrained/ 
     ```
     cd cal
     cd utils
     python model.py
     cd ..
     ```
     
 2. Related dataset: 
 
     [SA and NLI](https://github.com/acmi-lab/counterfactually-augmented-data/)
     
     [ANLI](https://github.com/facebookresearch/anli/)
     
     [twitter](https://alt.qcri.org/semeval2017/task4/index.php?id=results)
     
     The processed dataset file has been provided in the code.   
     
 
 3. Brief introduction for the parameters
 --op: choose one of the three operation, normal training with entire dataset/active learning/fintuning
 
 --func: the acquisition strategy for active learning
 
 --size: the number of samples acquired in each round
 
 --total: the total number of rounds
 
 --epoch: the number of epochs for finetuning
 
 --lr: learning rate
 
 --batchsize
 
 --T: which unlabeled pool to choose
 
 ### Natural Language Inference (NLI)
 1. Train with the entire train set
 
     1.1 only the factual samples
     
     ```python train_nli.py --op t --T 0 --epoch 10 --batchsize 4 --device 0 --seed 0```
     
     1.2 both factual and counterfactual samples
     
     ```python train_nli.py --op t --T 2 --epoch 10 --batchsize 2 --device 0 --seed 0```
     
 2. Active Learning
 
     2.1 implement active learning with baselines such as random:
     
     ```python train_nli.py --op al --size 48 --func random --T 0 --device 0 --seed 0```
     
     2.2 implement CAL:
     
     ```python train_nli_cal.py --device 0 --seed 0```
     
     2.3 implement pre-stage way AL with strategy like random:
     
     ```python train_nli.py --op al --size 48 --func random_all --T 1 --device 0 --seed 0```
     
     2.4 implement post-stage way AL with strategy like random:
     
     ```python train_nli.py --op al --size 16  --func random_cf --T 0 --cf 1 --cf_multi 1 --device 0 --seed 0 ```
     
 3. Finetuning a model with the acquisition set by each baseline and CAL, and report the IID and OOD performance 
 
     3.1 Test the IID/OOD performance of baseline like random:
     
     ```python train_nli.py --op ft --func random --batchsize 4 --epoch 10 --device 0 --seed 0```
     
     3.2 for CAL:
     
     ```python train_nli_cal.py --op ft --device 0 --seed 0```

     3.3 for pre-stage way:
        
     ```python train_nli.py --op ft --func random_all --batchsize 4 --epoch 10 --T 1 --device 0 --seed 0```
     
     3.4 for post-stage way:
     
     ```python train_nli.py --op ft --func random_cf --batchsize 2 --epoch 15 --cf 1 --cf_multi 1 --device 0 --seed 0```
 
 
 ### Sentiment Analysis (SA)
 1. Train with the entire train set
 
    1.1 only the factual samples
    
     ```python train_sa.py --op t --T 0 --epoch 5 --batchsize 4 --device 0 --seed 0```
     
    1.2 both factual and counterfactual samples
    
     ```python train_sa.py --op t --T 1 --epoch 5 --batchsize 2 --device 0 --seed 0```
     
 2. Active Learning
 
    2.1 implement active learning with baselines such as random:
    
     ```python train_sa.py --op al --func random --T 0 --device 0 --seed 0```
     
    2.2 implement CAL:
    
     ```python train_sa_cal.py --device 0 --seed 0```
     
    2.3 implement pre-stage way AL with strategy like random:
    
     ```python train_sa.py --op al --func random_all --T 2 --device 0 --seed 0```
     
    2.4 implement post-stage way AL with strategy like random:
    
     ```python train_sa.py --op al --func random_cf --size 16 --T 1 --batchsize 2 --device 0 --seed 0```
     
 3. Finetuning a model with the acquisition set by each baseline and CAL, and report the IID and OOD performance
 
    3.1 Test the IID/OOD performance of baseline like random:
    
     ```python train_sa.py --op ft --func random --T 0 --batchsize 4 --epoch 5  --device 0 --seed 0```
     
    3.2 for CAL:
    
     ```python train_sa_cal.py --op ft --device 0 --seed 0```
     
    3.3 for pre-stage way:
    
     ```python train_sa.py --op ft --func random_all --T 2 --epoch 5 --device 0 --seed 0```
     
    3.4 for post-stage way:
    
     ```python train_sa.py --op ft --func random_cf --T 1 --batchsize 2 --epoch 5 --device 0 --seed 0```
     
 
 ### ANLI
 1. Train with the entire train set
 
     1.1 only the factual samples
     
     ```python train_anli.py --op t --T 0 --epoch 10 --batchsize 4 --device 0 --seed 0```
     
     1.2 both factual and counterfactual samples
     
     ```python train_anli.py --op t --T 1 --epoch 10 --batchsize 2 --device 0 --seed 0```
     
 2. Active Learning
 
     2.1 implement active learning with baselines such as random:
     
     ```python train_anli.py --op al --size 72 --func random --T 0 --device 0 --seed 0```
     
     2.2 implement CAL:
     
     ```python train_anli_cal.py --device 0 --seed 0```
     
     2.3 implement pre-stage way AL with strategy like random:
     
     ```python train_anli.py --op al --size 72 --func random_all --T 1 --device 0 --seed 0```
     
     2.4 implement post-stage way AL with strategy like random:
     
     ```python train_nli.py --op al --size 24  --func random_cf --T 0 --cf 1 --cf_multi 1 --device 0 --seed 0 ```
     
 3. Finetuning a model with the acquisition set by each baseline and CAL, and report the IID and OOD performance 
 
     3.1 Test the IID/OOD performance of baseline like random:
     
     ```python train_anli.py --op ft --func random --batchsize 4 --epoch 10 --device 0 --seed 0```
     
     3.2 for CAL:
     
     ```python train_anli_cal.py --op ft --device 0 --seed 0```

     3.3 for pre-stage way:
        
     ```python train_anli.py --op ft --func random_all --batchsize 4 --epoch 10 --T 1 --device 0 --seed 0```
     
     3.4 for post-stage way:
     
     ```python train_anli.py --op ft --func random_cf --batchsize 2 --epoch 15 --cf 1 --cf_multi 1 --device 0 --seed 0```
 