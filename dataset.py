from transformers.trainer import SequentialDistributedSampler,torch_distributed_zero_first

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_torch_tpu_available

import json
import numpy as np
from filelock import FileLock
import pickle
import time

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
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,)
from transformers  import DataCollator, default_data_collator
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

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWrite
logger = logging.get_logger(__name__)

from torch import nn
import torch.nn.functional as F
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import load_tf_weights_in_bert,BertModel,BertOnlyMLMHead,BertOnlyNSPHead,BertLayer
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
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
from transformers.utils import logging



############################## imports ###############################################
class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        lock_path = cached_features_file + ".lock"
        print('in dataset')
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.document = []
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        
                        if not line:
                            break
                        try :    
                         line = line.strip()
                         line =line.split('\t')
                         
                         text_line =line[0]
                         visual_line =line[1]
                        except :
                          pass

                        text_tokens = tokenizer.tokenize(text_line)
                        visual_tokens = tokenizer.tokenize(visual_line)

                        text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
                        visual_tokens = tokenizer.convert_tokens_to_ids(visual_tokens)
                        
                        if text_tokens and visual_tokens:
                            self.document.append([text_tokens,visual_tokens])

                logger.info(f"Creating examples from {len(self.document)} documents.")
                self.examples = []
                self.create_examples_from_document(self.document)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def create_examples_from_document(self, document: List[int]):
        """Creates examples for a single document."""
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        for i in range(len(document)-1):



                   
              

                    tokens_a = []
                                      
                    
                    tokens_a.extend(document[i][0])
                    

                    is_random_next = True
                  

                    tokens_b = []

                    if random.random() < self.nsp_probability:
                        
   
                        random_idx = random.randint(1, len(document)-1)
                        
                        if random_idx == i :
                          try :
                             tokens_b.extend(document[random_idx+random.randint(i+1, len(document))][1])
                          except :
                             tokens_b.extend(document[random_idx-1][1])
                        else :
                          tokens_b.extend(document[random_idx][1])     
                    # Actual next
                    else:
                        is_random_next = False
                        
                        tokens_b.extend(document[i][1])
                          


                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    
                    self.examples.append(
                        {"tokens_a": tokens_a, "tokens_b": tokens_b, "is_random_next": is_random_next}
                    )


            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
class JointDataset(Dataset):
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )
        self.tokenizer = tokenizer
        lock_path = cached_features_file + ".lock"
        print('in dataset')
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.document = []
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if not line.isspace():
                            line_encoding = tokenizer(line, add_special_tokens=True, truncation=True, max_length=block_size)
                            line = line.strip()
                            line =line.split('[>]')
                            #print('line........................',line)
                            text_line =line[0]
                            visual_line =line[1]
                            #print(visual_line)
                        else :
                          continue
                        text_tokens = tokenizer.tokenize(text_line)
                        visual_tokens = tokenizer.tokenize(visual_line)
                        text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
                        visual_tokens = tokenizer.convert_tokens_to_ids(visual_tokens)
                        if text_tokens and visual_tokens and line_encoding:
                            self.document.append([text_tokens,visual_tokens,line_encoding['input_ids']])
                logger.info(f"Creating examples from {len(self.document)} documents.")
                self.examples = []
                self.create_examples_from_document(self.document)
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
    def create_examples_from_document(self, document: List[int]):
        print('creating features')
        """Creates examples for a single document."""
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)
        for i in range(len(document)-1):
                    mlm_tensor =torch.tensor(document[i][2], dtype=torch.long)
                    #mlm_tensor =document[i][2]
                    tokens_a = []
                    tokens_a.extend(document[i][0])
                    is_random_next = True
                    tokens_b = []
                    if random.random() < self.nsp_probability:
                        random_idx = random.randint(1, len(document)-1)
                        if random_idx == i :
                          try :
                             tokens_b.extend(document[random_idx+random.randint(i+1, len(document))][1])
                          except :
                             tokens_b.extend(document[random_idx-1][1])
                        else :
                          tokens_b.extend(document[random_idx][1])
                    # Actual next
                    else:
                        is_random_next = False
                        tokens_b.extend(document[i][1])
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    #print(is_random_next)
                    print('appending outputs in dataset')
                    self.examples.append(
                        {'mlm_tensor':mlm_tensor,"tokens_a": tokens_a, "tokens_b": tokens_b, "is_random_next": is_random_next}
                    )
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        
        return self.examples[i]
