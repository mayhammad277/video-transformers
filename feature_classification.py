# -*- coding: utf-8 -*-

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget
import os 

from sklearn.model_selection import train_test_split
from tokenizers.processors import RobertaProcessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json 
from tokenizers import ByteLevelBPETokenizer
from transformers import get_linear_schedule_with_warmup

from transformers import RobertaConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForMaskedLM

from transformers import LineByLineTextDataset
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from tokenizers import pre_tokenizers
from tokenizers import  processors

from transformers import Trainer, TrainingArguments

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from torch.optim import  AdamW
from transformers import RobertaConfig, RobertaModel
from transformers import  RobertaForSequenceClassification


from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from sklearn.metrics import  *


import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

from transformers import *
from tqdm import tqdm, trange


from transformers import FillMaskPipeline
from transformers import pipeline
from transformers import RobertaTokenizer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import json

import math

import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count


from scipy.stats import mode, pearsonr

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange





print(torch.__version__)

class class_Net(nn.Module):
    def __init__(self,feat_dim,num_class,):
        super(class_Net, self).__init__()
        #input features (num of frames,1024)
        self.fc = nn.Linear(feat_dim,num_class) 
        #self.softmax=nn.Softmax()


    def forward(self, x):
      logits=self.fc(x)
      #logits=self.softmax(x,0)

      return logits

org_dir='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/Video_s3dg_feats'
dataset_dict=json.load(open('feat2nn_s3d.json'))
#dataset=MyDataset(dataset_dict,org_dir)
verb_dict=json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/data_dicts/verbs_dict.json'))
dataset_dict=pd.DataFrame(dataset_dict.items(),columns=['feat','label'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = class_Net(1024,len(verb_dict))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.)

net=net.to(device)

dataset_dict

def get_label_2ix(label_list):
  label_to_ix = {}
  for label in label_list:
    for word in label.split():
      if word not in label_to_ix:
        label_to_ix[word]=len(label_to_ix)
  return label_to_ix





nn_label_to_ix=    get_label_2ix(verb_dict)

dataset_dict.insert(2, 'enc_label', 'True') 


for idx,row in  dataset_dict.iterrows():
  if(nn_label_to_ix.get(row['label'])):
    row['enc_label']=nn_label_to_ix[row['label']]
  else:
    row['enc_label']=len(verb_dict)

for idx,row in  dataset_dict.iterrows():
  row['feat']=os.path.join(org_dir, row['feat'])



"""Train dataset *** toy examples ,Test dataset *** toy examples"""

train, test = train_test_split(dataset_dict, test_size=0.3)
train, val = train_test_split(train, test_size=0.2)
'''
train=dataset_dict
test=train
val=test
'''
"""Tensor dataset formation"""

#####train 
x_tr=[np.mean(np.load(j),0) for j in train['feat'].values ]
x_tr =torch.tensor(x_tr).float()
y_tr = torch.tensor(train['enc_label'].tolist() )
#one_hot = torch.nn.functional.one_hot(y_tr)
# #####test 
x_te=[np.mean(np.load(j),0) for j in test['feat'].values ]
x_te =torch.tensor(x_te).float()
y_te = torch.tensor(test['enc_label'].tolist() )

#####validatom

x_v=[np.mean(np.load(j),0) for j in val['feat'].values ]
x_v =torch.tensor(x_v).float()
y_v = torch.tensor(val['enc_label'].tolist() )

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 128
######train tensors 
# wrap tensors
train_data = TensorDataset(x_tr,y_tr)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

### test tensors 
test_data = TensorDataset(x_te,y_te)

test_sampler = RandomSampler(test_data)


test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=64)



#validation tensors

val_data = TensorDataset(x_v,y_v)


val_sampler = RandomSampler(val_data)


val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=32)



"""Training and Validation ****************Toy example"""

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc


epochs=10
for e in tqdm(range(1, epochs+1)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    net.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = net(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}')

        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        net.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = net(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

"""Testing ********************toy example"""

####
torch.save(net.state_dict(), '/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M')

y_pred_list = []
with torch.no_grad():
    net.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = net(X_batch)
        y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
with open('predictions.txt',w) as wf:
  for i in y_pred_list :
    wf.write(i+'\n')


