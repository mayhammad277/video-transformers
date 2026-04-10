import json
import numpy as np
from filelock import FileLock
import pickle
import time

#from roberta_outputs import *
from transformers.modeling_bert import BertForPreTrainingOutput

import argparse
import json
import logging
import os
import random
from io import open
import math
import sys

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np


import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
#import torch.multiprocessing as mp

import csv
from transformers.optimization import AdamW,get_linear_schedule_with_warmup
from torch.utils.data import Dataset
#import pandas as pd
import os
from typing import (
    Dict, Iterable, Iterator, List, Optional, Sequence, Union, Mapping)
from tokenizers.processors import BertProcessing
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import gc
from transformers import  DataCollatorForNextSentencePrediction,BertForNextSentencePrediction,BertTokenizer,BertConfig,LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling,TextDataset
from typing import (
    Dict, Iterable, Iterator, List, Optional, Sequence, Union, Mapping)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling,BertForMaskedLM

from transformers import Trainer, TrainingArguments
from torch.utils.data import BatchSampler
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed )
from transformers.utils import logging
from transformers  import DataCollator, default_data_collator

from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_torch_tpu_available
from transformers.trainer import SequentialDistributedSampler,torch_distributed_zero_first
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray)

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWrite
logger = logging.get_logger(__name__)
import transformers
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
from dataclasses import asdict
from multiprocessing import cpu_count

import numpy as np
#import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,

    RobertaConfig,
    RobertaTokenizer,
  
    get_linear_schedule_with_warmup,
)
import numpy as np
#import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    MobileBertConfig,
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers.modeling_roberta import RobertaLMHead

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
import tempfile
from pathlib import Path
from transformers.convert_graph_to_onnx import convert






try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer


from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers  import DataCollator, default_data_collator

from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_torch_tpu_available

from transformers.trainer import SequentialDistributedSampler,torch_distributed_zero_first
import os 
import numpy as np 
from tokenizers.processors import RobertaProcessing
#import pandas as pd
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

#from nltk.corpus import stopwords
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from sklearn.metrics import  *


import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
#from ast import literal_eval



class RobBertaOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class RobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
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

class RobForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RobertaModel(config)
        self.cls = RobertaLMHead(config)
        self.NSPcls = RobBertaOnlyNSPHead(config)
        self.init_weights()
    def get_output_embeddings(self):
        return self.cls.decoder
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings
    def tie_weights(self):
       output_embeddings=self.get_output_embeddings()
       
           
       
       if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

       if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            self._tie_encoder_decoder_weights(self.bert.encoder, self.cls.decoder, self.base_model_prefix)
    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def update_embedding(self,tokenizer,word_init):
        print('in update embedding')
        #for key,value in word_init.items():
        #      tokenizer.add_tokens([key])
        self.config.vocab_size = len(tokenizer)
        
        def update(model,tokenizer) : 
          base_model = getattr(model, self.base_model_prefix, model)  # get the base model if needed

          old_embeddings =base_model.get_input_embeddings()
          new_num_tokens=len(tokenizer)
          
          if new_num_tokens is None:
            return old_embeddings

          old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
          if old_num_tokens == new_num_tokens:
            return old_embeddings
          #print(old_num_tokens,new_num_tokens)
          new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
          new_embeddings.to(old_embeddings.weight.device)
          base_model._init_weights(new_embeddings)
          base_model.vocab_size = new_num_tokens      

          # Copy token embeddings from the previous weights
          num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
          new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
          new_embeddings.weight.data[old_num_tokens:new_num_tokens-1, :] = torch.tensor(np.array(list(word_init.values())))  ###### must be changed
          base_model.set_input_embeddings(new_embeddings)
          n_emb=base_model.get_input_embeddings() 
       
        update(self.bert,tokenizer)
        self.tie_weights()



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        NSP_input_ids=None,
        NSP_attention_mask=None,
        NSP_token_type_ids=None,
        NSP_position_ids=None,
        NSP_head_mask=None,
        NSP_inputs_embeds=None,
        next_sentence_label=None,
        NSP_output_attentions=None,
        NSP_output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        print('in forward of pretraining roberta')
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
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
        sequence_output, pooled_output = outputs[:2]
        prediction_scores= self.cls(sequence_output)
        outputs_NSP = self.bert(
            NSP_input_ids,
            NSP_attention_mask,
            NSP_token_type_ids,
            NSP_position_ids,
            NSP_head_mask,
            NSP_inputs_embeds,
            NSP_output_attentions,
            NSP_output_hidden_states,
            return_dict=return_dict,
        )
        NSP_pooled_output = outputs_NSP[1]
        seq_relationship_score = self.NSPcls(NSP_pooled_output)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )      
