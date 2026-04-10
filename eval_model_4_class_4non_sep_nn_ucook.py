from fill_multi_mask import *
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget
import os
import numpy as np
from tokenizers.processors import RobertaProcessing
#import pandas as pd
import torch
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
#import nltk
#from nltk import *
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from sklearn.metrics import  *

import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle


from transformers import FillMaskPipeline
from transformers import pipeline
from transformers import RobertaTokenizer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoModelWithLMHead, AutoTokenizer
import wget
import string
import os
import re
#import nltk
#from nltk import *
from collections import Counter,defaultdict
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#import time

#from dataset import *
#from data_collator import *
#from file_utils import *
from roberta_model_modified import RobertaPreTrainedModel ,RobBertaOnlyNSPHead,RobForPreTraining
#from roberta_outputs import *
from transformers import Trainer, TrainingArguments
from transformers import (
    WEIGHTS_NAME,
    AdamW,

    RobertaConfig,
    RobertaTokenizer,

    get_linear_schedule_with_warmup,
)
from transformers.modeling_roberta import RobertaLMHead


from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizer.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M" ,max_len=128)

pretrained_config =RobertaConfig.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M" )

Video_Roberta =RobForPreTraining.from_pretrained('/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M')
print('intallized the rob pretrained model')



Visual_text_Model=RobertaForMaskedLM(config=pretrained_config)

#Viusal_alighment=RobertaForNextSentencePrediction(config=pretrained_config)

Visual_text_Model.roberta.load_state_dict(Video_Roberta.bert.state_dict(),strict=False)
Visual_text_Model.lm_head.load_state_dict(Video_Roberta.cls.state_dict())

#Viusal_alighment.bert.load_state_dict(Video_Roberta.bert.state_dict())
#Viusal_alighment.cls.load_state_dict(Video_Roberta.NSPcls.state_dict())


#fill=pipeline(model=Visual_text_Model,tokenizer=tokenizer,task='fill-mask')




import itertools




def pred_mask_for_sents(masked_list,model,tokenizer):
  filled_list=[]
  #with open(file,'w') as  f_filled_text:


  for sent in masked_list:
    
    try :
      sent_preds=get_prediction(sent,tokenizer,model)
  
      print("lists....................hey",sent_preds)
    
      filled_list.append(sent_preds)
    except:
      filled_list.append([])
    
  return filled_list

'''
masked=open('/ibex/scratch/projects/c2090/may_ycook/evaluations/masked_nn_test.txt','r').read().splitlines()

##pred_mask_for_sents('/ibex/scratch/projects/c2090/may_ycook/evaluations/mask_nn_preds_test_2M_s.txt',masked_nn,Visual_text_Model,tokenizer)

filled_list=pred_mask_for_sents(masked,Visual_text_Model,tokenizer)
with open('/ibex/scratch/projects/c2090/may_ycook/evaluations/mask_nn_preds_test_s.json','w') as f:
  json.dump(filled_list,f)
'''
masked_vb=open('/ibex/scratch/projects/c2090/may_ycook/evaluations/masked_vb_test.txt','r')
masked_vb=masked_vb.read().splitlines()


filled_list=pred_mask_for_sents(masked_vb,Visual_text_Model,tokenizer)
with open('/ibex/scratch/projects/c2090/may_ycook/evaluations/mask_vb_preds_test_s.json','w') as f:
  json.dump(filled_list,f)



