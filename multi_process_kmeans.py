import multiprocessing
from multiprocessing import *
from collections import deque
import numpy as np
import scipy as sp
from sklearn import cluster
import os
import itertools
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering ,k_means
from sklearn.neighbors.nearest_centroid import NearestCentroid

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget

from tokenizers.processors import RobertaProcessing
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

from sklearn.metrics import *
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

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
import time 
from multiprocessing import Pool ,Process

"""to keep track of text features alighment each video feature descripors map will be devided into multiples of 16 each (16 tokens)will represent a visual sentnece and should be alighned with a text sentence from video csv file ."""

video_feat_path='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/Video_s3dg_feats'
video_text_csv_path='/ibex/scratch/projects/c2090/howto100m_csv'
sorted_feat_dir=sorted(os.listdir(video_feat_path))
sorted_text_dir =sorted(os.listdir(video_text_csv_path))
k=20000

'''
video_feat_path='video_features'
video_text_csv_path='video_text'
sorted_feat_dir=sorted(os.listdir(video_feat_path))
sorted_text_dir =sorted(os.listdir(video_text_csv_path))

k=20

'''
'''
def save_size_feats(single_feat_size,curr_idx,idx_size_dict):
  #curr_idx is the place of feature vector of certain file in the queue say file with some name was given curr_idx of 1 that means it was first processed and stacked in the list
  idx_size_dict[curr_idx]=single_feat_size

#each video segment as a token ---combine tokens into list and cluster them (all the tokens ),each video (k segments) is respresnted in an .npy file named by the videoname
#a loop to loop on feature fileas load and append prepared feats to a list 

def prep_feats(orginal_dear,feat_dir):
  strt=time.time()

  idx2size={}
  feat0=np.load(os.path.join(orginal_dear,feat_dir[0]))

  shape=feat0.shape[0]
  save_size_feats(shape,0,idx2size)
  for i  in range(1,len(feat_dir)):
      #if feat[i].endswith('.npy'):
    f=os.path.join(orginal_dear, feat_dir[i])
    feat=np.load(f)
      #shape =( k_segments, 1024))
    feat=feat.astype(np.float16)
    shape=feat.shape[0]
    save_size_feats(shape,i,idx2size)
      #sq_feats=np.squeeze(feat_arr,(0,1,2,3)) #shape (, 1024)
    feat0= np.append(feat0,feat,axis=0)# list will bw of dim [N,(k,1024)] where N is$
    #feat0=feat0.astype(np.float16)
    print('shape of feat0 in first function prep feats',shape)
    print('time taken in prep feats ',time.time()-strt)
  
  return feat0,idx2size

#so the feat_arr will be featurized appended version of the sorted feat files 

pool = Pool(processes=multiprocessing.cpu_count()+20)
resultlist = pool.apply_async(prep_feats ,args=(video_feat_path,sorted_feat_dir))
pool.close()
pool.join()  
feat_arr,idx2size_dict=resultlist.get()
#feat_arr,idx2size_dict=prep_feats(video_feat_path,sorted_feat_dir)########################3###############################

print(feat_arr.shape) #k(segments)*N(Videos)*1024  where k*N is total number of visual words(observations)
'''

feat_arr=np.load('feat_arr.npy')
idx2size_dict=json.load(open('idx2size.json'))

def get_preds_lables(k,feat_arr,n_jobs):

  strt=time.time()
  with parallel_backend('threading', n_jobs=n_jobs):
    cluster = MiniBatchKMeans(n_clusters=k, random_state=0,batch_size=10000)
    kmeans = cluster.partial_fit(feat_arr[0:10000,:])
    kmeans = kmeans.partial_fit(feat_arr[10000:100000,:])
    kmeans = kmeans.partial_fit(feat_arr[100000:200000,:])
    kmeans = kmeans.partial_fit(feat_arr[200000:,:])
    print('laables',kmeans.labels_.shape)
   # vid_lables=kmeans.labels_
  preds=kmeans.predict(feat_arr)
  print('preds',preds.shape)
  print('time taken in get predictions of cluster ',time.time()-strt)
  return preds



#



pool2=Pool(processes = multiprocessing.cpu_count()+20)
resultlist2 = pool2.apply_async(get_preds_lables ,args=(k,feat_arr,50))
pool2.close()
pool2.join()
preds=resultlist2.get()


#preds= get_preds_lables(k,feat_arr,16)###########3###############################

def get_centroids(feats,predictions):

    clf = NearestCentroid()
    clf.fit(feat_arr, preds)

    vid_centroids=clf.centroids_
    return vid_centroids

pool3=Pool(multiprocessing.cpu_count()+20)
resultlist3 = pool3.apply_async(get_centroids ,args=(feat_arr,preds))
pool3.close()
pool3.join()
vid_centroids=resultlist3.get()

#vid_centroids=get_centroids(feat_arr,preds)################################
print(vid_centroids,vid_centroids.shape)#num_centroids*feat_dim





def map_points2centers(lables):
  strt=time.time()
  points_map_dict={}
  for i,lab in enumerate(lables):
    points_map_dict[i]='v'+str(lab)
  print('time takn in map_points to centers ',time.time()-strt)  
  return points_map_dict

points_map=map_points2centers(preds)
print('points_map',points_map)

"""0------->end 0=  start0+(125-(125%15)-1)  ,  (end 0)+1------> end 1= start1+(67-(67%15)-1) ,   (end 1)+1 ----------->end2=start2 +(109+(109%15)-1)

the numbers : 125 ,67,109 are the idx2size[counter] which are the sizes of video descriptors
"""

def get_end(start,conter_idx,idx2size):
  '''
  if(idx2size[conter_idx]%15 !=0):#reminder not deisble by 15 
    end=idx2size[conter_idx]-(idx2size[conter_idx]%15)-1 #125-5-1 =119 while start is 0 acheived 0:119 ~120 token in to total  8 in each sentence 

  elif(idx2size[conter_idx]%15==0):
    end=idx2size[conter_idx]-1
  '''
  end=start+(idx2size[conter_idx]-(idx2size[conter_idx]%20))-1
  return end

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def form_visual_sents(feat_list,points2centers,idx2size_dict):
  strt=time.time()
  #feat_list=list(feat_arr) and looping on it loops on the rows (data points /segment feature descriptors 0)
  # we want to take each consecutive 15 tokens and map them to a sentnce from the current video file 
  #idx2size contanting the number of feature descriptors of video so for example video 1 has the size of 125 so it controls data points 0:124 so if we are to form visual sentnces from 0:124 data tokens will use only from 0:119 ~120 
  # as 15 will devide 120 (aka reminder =0) so will have 120/15=8 tokens sentnces from this file  
  vis_tk_list=deque()
  file = open('visual_train_s3d.txt', 'a')

  for i in range(len(feat_list)):
    if(points2centers.get(i)):

        vis_tk_list.append(points2centers[i])#appending the coresponding visual center name to each segment(data point) aka (representing each segment by its cluster centroid ) so we have a list of visual tokens 
  print(vis_tk_list)
  vsents_list=[]
  start=0
  end=0

  counter=0
  while (idx2size_dict.get(counter) and end<len(vis_tk_list)):
    end=get_end(start,counter,idx2size_dict)
    print(start,end)
    #chunks=chunkIt(vis_tk_list[start:end],(end-start)/15)
    chunks=chunkIt(list(itertools.islice(vis_tk_list, start, end)),(end-start)/20)
    for chnk in chunks:
      #vsents_list.append(chnk)
      file.write(str(' '.join(chnk))+'\n')
    start=end+1
    counter+=1
  print('time taken in form vis sents ',time.time()-strt)
  file.close()
  #return vsents_list


pool6=Pool(processes=multiprocessing.cpu_count()+20)
resultlist6 = pool6.apply_async(form_visual_sents,args=(list(feat_arr),points_map,idx2size_dict) )

pool6.close()
pool6.join()


#points_map=resultlist5.get()


#form_visual_sents(list(feat_arr),points_map,idx2size_dict) ######################################################




'''
file = open('visual_train.txt', 'w')
for item in visual_sentences:

  file.write("%s\n" % str(' '.join(item)))

f2 = open('visual_test.txt', 'w')
for item in visual_sentences:

  f2.write("%s\n" % str(' '.join(item)))
'''
#look up dict example 
# we want to represent each segment by its centriod  for zero-shot classification
# adding to lookup table the visual centriods only (representing the visual tokens )
#represeting a specific segemnt --is by its centriod so more than one segment have the same representation 
# video segment 0 ----- the dimension of of the encoding should be (number of clusters) eg.here no of clusters are 3 like vocab['v000 ' ] =vid_centroids[0]
#or another encoding method would be vocab['v0']=video_centriods[0],vocab['v20000']=video_centroids[20000]

def feat_encoding(centroids):
    vocab_set={}
#vocab_set['vO']=vid_centroids[0]
#vocab_set['v1']=vid_centroids[1]
#vocab_set['v2']=vid_centroids[2]
#vocab_set['v3']=vid_centroids[3]

    for i,embed in enumerate(vid_centroids):
        vocab_set['v'+str(i)]=embed

    return vocab_set

'''
pool7=Pool(processes = 20)
resultlist7 = pool7.apply_async(feat_encoding,args=(vid_centroids) )
vocab_set=resultlist7.get()
'''

vocab_set=feat_encoding(vid_centroids) ######################################################3
print('vocab_set',vocab_set)

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
            json.dump(chunk, fout)





dump_big_dict(vocab_set)

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
