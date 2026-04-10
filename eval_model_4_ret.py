from fill_multi_mask import *
from transformers import BertTokenizer
from transformers.modeling_bert import load_tf_weights_in_bert,BertModel,BertOnlyMLMHead,BertOnlyNSPHead,BertLayer
from typing import (
    Dict, Iterable, Iterator, List, Optional, Sequence, Union, Mapping)
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

from transformers.configuration_bert import BertConfig

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

class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = RobertaModel(config)
        self.cls = RobBertaOnlyNSPHead(config)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring).  Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', return_dict=True)

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, next_sentence_label=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


tokenizer = RobertaTokenizer.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M" ,max_len=128)

pretrained_config =RobertaConfig.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M" )

Video_Roberta =RobForPreTraining.from_pretrained('/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/roberta_sep_new_model_2M')
print('intallized the rob pretrained model')



Visual_text_Model=RobertaForMaskedLM(config=pretrained_config)

Viusal_alighment=RobertaForNextSentencePrediction(config=pretrained_config)

Visual_text_Model.roberta.load_state_dict(Video_Roberta.bert.state_dict(),strict=False)
Visual_text_Model.lm_head.load_state_dict(Video_Roberta.cls.state_dict())

Viusal_alighment.bert.load_state_dict(Video_Roberta.bert.state_dict())
Viusal_alighment.cls.load_state_dict(Video_Roberta.NSPcls.state_dict())


#fill=pipeline(model=Visual_text_Model,tokenizer=tokenizer,task='fill-mask')

EVALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
EVALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"\

def retrieval_results_to_str(results: Dict[str, float], name: str):
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(
        name, *[results[a] for a in EVALKEYS])
def compute_retr_vid_to_par(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    #print((num_points))
    d = np.dot(video_feat, cap_feat.T)
    return compute_retrieval_cosine(d, num_points)
def compute_retr_par_to_vid(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    #print((num_points))
    d = np.dot(cap_feat, video_feat.T)
    return compute_retrieval_cosine(d, num_points)
def compute_retrieval_cosine(dot_product, len_dot_product):
    ranks = np.zeros(len_dot_product)
    top1 = np.zeros(len_dot_product)
    #print(dot_product.shape)
    for index in range(len_dot_product):
        inds = np.argsort(dot_product[index])[::-1]
        #print(inds)
        where = np.where(inds == index)
        #print(where,index)
        rank = where[0][0]
        ranks[index] = rank
        top1[index] = inds[0]
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = dict()
    report_dict['r1'] = r1
    report_dict['r5'] = r5
    report_dict['r10'] = r10
    report_dict['r50'] = r50
    report_dict['medr'] = medr
    report_dict['meanr'] = meanr
    report_dict['sum'] = r1 + r5 + r50
    return report_dict, top1, ranks
import csv
import os
from pathlib import Path
from timeit import default_timer as timer
import torch
import torch.nn.parallel
#from easydict import EasyDict
from torch.nn import functional as F
def validate(model,tokenizer, sents):
      model.eval()
      clip_emb_list = []
      sent_emb_list = []
      for sent in sents:
            text_sent=sent.split('[>]')[0]
            vid_sent=sent.split('[>]')[1]
            text_inputs = tokenizer.encode(text_sent,return_tensors='pt' ,padding='max_length',max_length=128)
            sent_emb = model(text_inputs)[0].squeeze()
            #print(sent_emb.shape)
            video_inputs = tokenizer.encode(vid_sent,return_tensors='pt' ,padding='max_length',max_length=128)
            vid_emb = model(video_inputs)[0].squeeze()
            #print(vid_emb.shape)
            clip_emb_list.extend(vid_emb.detach().cpu())
            sent_emb_list.extend(sent_emb.detach().cpu())
      clip_emb_list = torch.stack(clip_emb_list, 0)
      sent_emb_list = torch.stack(sent_emb_list, 0)
      # clip sentence retrieval
      clip_emb_list = F.normalize(clip_emb_list).numpy()
      sent_emb_list = F.normalize(sent_emb_list).numpy()
      c2s_res, c2s_top1, c2s_ranks = compute_retr_vid_to_par(
            clip_emb_list, sent_emb_list)
      s2c_res, s2c_top1, s2c_ranks = compute_retr_par_to_vid(
            clip_emb_list, sent_emb_list)
      c2s_sum_at_1 = c2s_res["r1"] + s2c_res["r1"]
      print(retrieval_results_to_str(s2c_res, "Sen2Cli"))
      print(retrieval_results_to_str(c2s_res, "Cli2Sen"))
      return (c2s_res, s2c_res, c2s_sum_at_1)

sents=open('/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/sep_joined_sents_te_100k.txt','r').read().splitlines()
validate(Visual_text_Model,tokenizer,sents)
