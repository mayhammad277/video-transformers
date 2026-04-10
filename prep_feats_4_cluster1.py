import time 
import numpy as np 
import json
import multiprocessing
from multiprocessing import *
import faiss
from collections import deque
import numpy as np
import scipy as sp
#from sklearn import cluster
import os
import itertools
from itertools import *
from numpy import array

from scipy.cluster.vq import vq
from scipy.cluster.vq import whiten
from scipy.cluster.vq import  kmeans2
from joblib import parallel_backend

from multiprocessing import Pool ,Process


video_feat_path='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/Video_s3dg_feats'
video_text_csv_path='/ibex/scratch/projects/c2090/howto100m_csv'
sorted_feat_dir=sorted(os.listdir(video_feat_path))
sorted_text_dir =sorted(os.listdir(video_text_csv_path))
n0=200000
nf=220000
n3=len(sorted_feat_dir)-900000

def save_size_feats(single_feat_size,curr_idx,idx_size_dict):
  #curr_idx is the place of feature vector of certain file in the queue say file with some name was given curr_idx of 1 that means it was first processed and stacked in the list
  idx_size_dict[curr_idx]=single_feat_size


def prep_feats(orginal_dear,feat_dir):
  idx2size={}
  strt=time.time()
  arr_list0=[]
  arr_list1=[]
  arr_list2=[]
  arr_list3=[]
  for i  in range(n0,nf):
    f=os.path.join(orginal_dear, feat_dir[i])
    
    feat=np.load(f)
    shape=feat.shape[0]
    save_size_feats(shape,i,idx2size)
    print('shap of feat ',feat.shape,'finished i videos',i)
    arr_list0.append(feat.astype(np.float16))
  
    
    #feat=feat.astype(np.float16)   
    
    #shape=feat.shape[0]
    #save_size_feats(shape,i,idx2size)
    #print('shap of feat ',feat.shape,'finished i videos',i)
    #try:
    #  arr_list0.append(feat.astype(np.float16))
    #except:
    #  pass
  print('len arr list ',len(arr_list0),'shape arr list [0]',arr_list0[0].shape)
  splits0=np.array_split(np.array(arr_list0),20)
  print("splits[0[0]]",splits0[0].shape)
  return splits0,idx2size
'''  
  for i  in range(300000,600000):
    f=os.path.join(orginal_dear, feat_dir[i])
    try:
      feat=np.load(f)
    except:
      continue
    #feat=feat.astype(np.float16)

    shape=feat.shape[0]
    save_size_feats(shape,i,idx2size)
    print('shap of feat ',feat.shape,'finished i videos',i)
    try:
      arr_list1.append(feat.astype(np.float16))
    except:
      pass
  splits1=np.array_split(np.array(arr_list1),100)
  
  print('shape of feat0 in first function prep feats',shape)
  print('time taken in make splits feats ',time.time()-strt)

  for i  in range(600000,900000):
    f=os.path.join(orginal_dear, feat_dir[i])
    try:
      feat=np.load(f)
    except:
      continue
    #feat=feat.astype(np.float16)

    shape=feat.shape[0]
    save_size_feats(shape,i,idx2size)
    print('shap of feat ',feat.shape,'finished i videos',i)
    try:
      arr_list2.append(feat.astype(np.float16))
    except:
      pass
  splits2=np.array_split(np.array(arr_list2),100)
  for i  in range(900000,len(feat_dir)):
    f=os.path.join(orginal_dear, feat_dir[i])
    try:
      feat=np.load(f)
    except:
      continue
    #feat=feat.astype(np.float16)

    shape=feat.shape[0]
    save_size_feats(shape,i,idx2size)
    print('shap of feat ',feat.shape,'finished i videos',i)
    try:
      arr_list3.append(feat.astype(np.float16))
    except:
      pass
  splits3=np.array_split(np.array(arr_list3),100)
'''
 
#so the feat_arr will be featurized appended version of the sorted feat files


splits0,idx2size_dict=prep_feats(video_feat_path,sorted_feat_dir)

def stack_splits(splits,n):
  strt=time.time()
  i,j=0,0
  feat_arr=np.zeros((n*1000,1024),dtype=np.float16).astype(np.float16)
  print(feat_arr.shape)
  for i,li in enumerate(splits):
    print('shape of lists in splits',li.shape)
    
    stacked=np.vstack(list(li))
    shape=stacked.shape[0]
    i=j
    j+=shape
    print(shape)
    
    feat_arr[i:j][:]=stacked
   
  print(time.time()-strt)
  return feat_arr

feat_arr0=stack_splits(splits0,nf-n0)
'''
feat_arr1=stack_splits(splits1,n0)
feat_arr2=stack_splits(splits2,n0)
feat_arr3=stack_splits(splits3,n3)
'''
'''
pool = Pool(processes=multiprocessing.cpu_count()+20)
resultlist = pool.apply_async(prep_feats ,args=(video_feat_path,sorted_feat_dir))
pool.close()
pool.join()  
feat_arr,idx2size_dict=resultlist.get()
'''


out=open('feat_arr11.npy','wb')
np.save(out, feat_arr0)
'''
out1=open('feat_arr1.npy','wb')
np.save(out1, feat_arr1)
out2=open('feat_arr2.npy','wb')
np.save(out2, feat_arr2)
out3=open('feat_arr3.npy','wb')
np.save(out3, feat_arr3)
'''
with open('idx2size11.json','w') as wp :
  json.dump(idx2size_dict,wp)
