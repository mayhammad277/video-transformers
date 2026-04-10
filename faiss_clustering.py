
import multiprocessing
from multiprocessing import *
import faiss
from collections import deque
import numpy as np
import scipy as sp
from sklearn import cluster
import os
import itertools
from itertools import *

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering ,k_means
from sklearn.neighbors.nearest_centroid import NearestCentroid

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget
import os 
import numpy as np 
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


import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import time
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

from numpy import array
from scipy.cluster.vq import vq
from scipy.cluster.vq import whiten
from scipy.cluster.vq import  kmeans2
from joblib import parallel_backend

from multiprocessing import Pool ,Process


k=23000




combined_feat_dir='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/data'
sorted_feat_list=sorted(os.listdir(combined_feat_dir))
#idx2size_dir='/ibex/scratch/projects/c2090/prep_feats'
#idx2size_dict_list=sorted(os.listdir(combined_feat_dir))

def sample(num_samples,y):
    lst =[]
    for _ in range(num_samples):
        x =np.random.randint(0,len(y))
        lst.append(x)
    return y[lst,:]

def get_feats_comb(combined_feat_dir,file_list):
  strt=time.time()
  
  feat_arr=list()
  f=os.path.join(combined_feat_dir,'all_feats1324451_.npy')
  f=np.load(f)
  
  print('done with feats in time ',time.time()-strt)
  return f
  
#feat_arr=np.load('feat_arr.npy')
#idx2size_dict=json.load(open('idx2size.json'))
feat_arr=get_feats_comb(combined_feat_dir,sorted_feat_list)



def train_km(k,feat_arr):
  strt=time.time()
  ncentroids = k
  niter = 10
  verbose = True
  d = feat_arr.shape[1]
  kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,gpu=True)
  kmeans.train(feat_arr.astype(np.float32))
  print('finished training kmeans',time.time()-strt)
  return kmeans

#training for gou 

def train_km_gpu(k,feat_arr):
  strt=time.time()
  
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0
  ncentroids = k
  niter = 20
  verbose = True
  d = feat_arr.shape[1]
  index = faiss.GpuIndexFlatL2(res, d, flat_config)

  kmeans = faiss.Clustering(d, k)
  kmeans.verbose = True
  kmeans.niter = 300
  kmeans.nredo = 10
  kmeans.seed = 0

  index = faiss.GpuIndexFlatL2(res, d, flat_config)

  kmeans.train(feat_arr.astype(np.float32), index)
  trained_index = faiss.index_gpu_to_cpu(index)
  faiss.write_index(trained_index, 'trained_index_path.index') 

  print('finished training kmeans',time.time()-strt)
  
  return kmeans


km_obj= train_km_gpu(k,feat_arr)

km_obj.centroids


def get_lables(partial_feat_arr,km_obj):
  strt=time.time()
  D2c, I = km_obj.index.search(partial_feat_arr.astype(np.float32),1)##lables 
  lables=list(chain.from_iterable(I))
  print('done with gerring lables',time.time()-strt)
  return lables

def map_points2centers(lables):
  strt=time.time()
  points_map_dict={}
  for i,lab in enumerate(lables):
    points_map_dict[i]='v'+str(lab)
  print('done with points map',time.time()-strt)
  return points_map_dict

#points 2 centers
def map_centers2points(feat_arr,k):
  strt=time.time()
  d=feat_arr.shape[1]
  index = faiss.IndexFlatL2 (d)
  index.add (feat_arr.astype(np.float32))
  D2p, centers2points= index.search (km_obj.centroids, feat_arr.shape[0]//k)
  print('done with map_centers2points',time.time()-strt)
  return centers2points

def feat_encoding(centroids):
  strt=time.time()
  vocab_set={}
#vocab_set['vO']=vid_centroids[0]
#vocab_set['v1']=vid_centroids[1]
#vocab_set['v2']=vid_centroids[2]
#vocab_set['v3']=vid_centroids[3]
  for i,embed in enumerate(centroids):
    vocab_set['v'+str(i)]=embed
  print('time taken to make vocab set',time.time()-strt)
  return vocab_set

vocab_set=feat_encoding(km_obj.centroids)
#print(vocab_set)

def get_end(start,conter_idx,idx2size):
  
  end=start+(idx2size[conter_idx]-(idx2size[conter_idx]%5))-1
  return end

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def form_visual_sents(counter,feat_list,points2centers,idx2size_dict,idx_file):
  print('in form visual sentences')
  strt=time.time()
  #feat_list=list(feat_arr) 
  idxed_files=open(idx_file,'r')
  idxed_files = idxed_files.read().splitlines()
  f = open('visual_train_s3d.txt', 'a')
  vis_tk_list=deque()
  
  for i in range(len(feat_list)):

      vis_tk_list.append(points2centers[i])#appending the coresponding visual center name to each segment(data point) aka (representing each segment by its cluster centroid ) so we have a list of visual tokens 
      #print('points2centers[i]',points2centers)
 
  start=0
  end=0
  
  
  while (end<len(vis_tk_list)):
    #print('in while .........................hello')
    #end=get_end(start,str(counter),idx2size_dict)
    end+=idx2size_dict[str(idxed_files[counter])]
    print('emd',end)

    
    slice=list(itertools.islice(vis_tk_list, start, end))
    f.write(str(' '.join(slice))+'\n')
    start=end+1
    counter+=1
  print('time taken in form vis sents ',time.time()-strt)
  f.close()  
  return counter
  








def get_batch_sents(counter,combined_feat_dir,sorted_feat_list):
  #batch_idx2size={}
  strt=time.time()
  
  fn='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/data/all_feats1324451_.npy'
  fj='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/data/all_size1324451_.json'
  #print("feat_file",fn,'json',fj)    
  
  batch_feat_vec=np.load(fn)
  batch_lables=get_lables(batch_feat_vec,km_obj)
  batch_points_map=map_points2centers(batch_lables)

  batch_idx2size=json.load(open(fj))

  counter=form_visual_sents(counter,list(batch_feat_vec),batch_points_map,batch_idx2size,'/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/data/all_size1324451_.txt')

  print('time taken in forming batch visual clusters ',time.time()-strt)

get_batch_sents(0,combined_feat_dir,sorted_feat_list)

for k in vocab_set.keys():
  vocab_set[k]=vocab_set[k].tolist()

#print(vocab_set)
def mk_chunks(d, chunk_size=10000):
    chunk = {}
    ctr = chunk_size
    for key, val in d.items():
        chunk[key] = val
        ctr -= 1
        if ctr == 0:
            yield chunk
            ctr = chunk_size
            chunk = {}
    if chunk:
        yield chunk

def dump_big_dict(d):
    with open('vocab_set_s3d.json', "w") as fout:
        for chunk in mk_chunks(d, chunk_size=10000):
            print('dumping into vocav file')
            json.dump(chunk, fout)




with open('vocab_set_s3d.json', "w") as fout:
  print('dumping into vocav file')
  json.dump(vocab_set, fout)
#dump_big_dict(vocab_set)

#representing each video

# the aim is to add k cluster centroidal features to the emedding materix of the Rberta pretrained model (intalize there values without any training with their cluster features)

# representing videos each by its centriod --we have a (dict from collect_data_2centr ) which connects the cluster centers to its asigned datapoints and 
def asign2center(center2data,centriods):
    segment_feats={}
    for k,values in center2data.items():# v is the datapoints represented by indecies (eg .1,2,3,4...), k represent the cluster centriods (0,1,2.....) 
        for v in list(values) : #we want to asign to each data point ints corresponding cluster center  feature vector  ...the cluster center's number resembles its index in the centriodal feature matrix 
            segment_feats[str(v)]=centriods[int(k)]
    return segment_feats



def add_vis_tokens(model,tokenizer,vis_vocab_set):
  print(len(tokenizer)) 
  for center_token ,center_embd in vis_vocab_set.items():
    print(center_token)
    tokenizer.add_tokens([center_token])
    print("new_tokenizer len",len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    print(model.embeddings.word_embeddings.weight[-1, :].shape)
    model.embeddings.word_embeddings.weight[-1, :] = torch.tensor(center_embd)
    print(model.embeddings.word_embeddings.weight[-1, :])
