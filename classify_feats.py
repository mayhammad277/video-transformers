
from transformers import BertTokenizer
import multiprocessing
from multiprocessing import *
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
from tokenizers.processors import RobertaProcessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json 
from tokenizers import ByteLevelBPETokenizer
from transformers import get_linear_schedule_with_warmup
#from transformers  import create_optimizer
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
import nltk 
from nltk import *
nltk.download('stopwords')
from nltk.corpus import stopwords
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from sklearn.metrics import  *


import numpy as np
#import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
#import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval

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


#from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
#from dataclasses import asdict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
#from tqdm.contrib import tenumerate
from transformers import (
    WEIGHTS_NAME,
    AdamW,

    RobertaConfig,
    RobertaTokenizer,
  
    get_linear_schedule_with_warmup,
)






try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

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
dataset_dict=json.load(open('/ibex/scratch/projects/c2090/May/feat_align_s3d/feat2vb_s3d.json'))
#dataset=MyDataset(dataset_dict,org_dir)
verb_dict=json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/data_dicts/verbs_dict.json'))
dataset_dict=pd.DataFrame(dataset_dict.items(),columns=['feat','label'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = class_Net(1024,len(verb_dict))



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0007, momentum=0.)

net=net.to(device)

dataset_dict

def get_label_2ix(label_list):
  label_to_ix = {}
  for label in label_list:
    print(label)
    #for word in label.split():
    if label not in label_to_ix:
      label_to_ix[label]=len(label_to_ix)
  return label_to_ix





nn_label_to_ix=    get_label_2ix(verb_dict)

nn_label_to_ix

dataset_dict.insert(2, 'enc_label', 'True') 

for idx,row in  dataset_dict.iterrows():
  if(nn_label_to_ix.get(row['label'])):
    row['enc_label']=nn_label_to_ix[row['label']]
  else:
    row['enc_label']=len(verb_dict)

for idx,row in  dataset_dict.iterrows():
  row['feat']=os.path.join(org_dir, row['feat'])



msk = np.random.rand(len(dataset_dict)) < 0.8

train=dataset_dict[msk]
test=dataset_dict[~msk]

msk2=np.random.rand(len(train)) < 0.2
val=train[msk2]


"""Tensor dataset formation"""
'''
#####train 
#x_tr=[np.mean(np.load(j),0) for j in train['feat'].values if os.path.exists(j) ]
#y_t=train['enc_label'].tolist()
x_tr,y_tr=zip([np.mean(np.load(j),0) for j in train['feat'].values if os.path.exists(j)]
x_tr =torch.tensor(x_tr).float()

y_tr = torch.tensor(train['enc_label'].tolist() ).long()
#one_hot = torch.nn.functional.one_hot(y_tr)
# #####test 
x_te=[np.mean(np.load(j),0) for j in test['feat'].values if os.path.exists(j) ]
x_te =torch.tensor(x_te).float()
y_te = torch.tensor(test['enc_label'].tolist() ).long()

#####validatom

x_v=[np.mean(np.load(j),0) for j in val['feat'].values if os.path.exists(j)]
x_v =torch.tensor(x_v).float()
y_v = torch.tensor(val['enc_label'].tolist() ).long()

y_tr
'''


##train
x_tr=[]
y_tr=[]
for j,i in zip(train['feat'].values,train['enc_label'].values):
  if (os.path.exists(j)):

    x_tr.append(np.mean(np.load(j),0))
    y_tr.append(i)




##test 


x_te=[]
y_te=[]
for j,i in zip(test['feat'].values,test['enc_label'].values):
  if (os.path.exists(j)):

    x_te.append(np.mean(np.load(j),0))
    y_te.append(i)
## val 
x_v=[]
y_v=[]
for j,i in zip(val['feat'].values,val['enc_label'].values):
  if (os.path.exists(j)):

    x_v.append(np.mean(np.load(j),0))
    y_v.append(i)



x_tr =torch.tensor(x_tr).float()
y_tr = torch.tensor(y_tr).long()



x_te =torch.tensor(x_te).float()
y_te = torch.tensor(y_te).long()


x_v =torch.tensor(x_v).float()
y_v = torch.tensor(y_v).long()
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 64
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


test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=16)



#validation tensors

val_data = TensorDataset(x_v,y_v)


val_sampler = RandomSampler(val_data)


val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=8)



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


        
    # VALIDATION    
    net.eval()
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        
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
print(y_pred_list)
with open('feat_preds.txt','w') as f:
  for item in y_pred_list:
    f.write(item+'\n')


    
