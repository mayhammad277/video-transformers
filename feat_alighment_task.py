

import sys 
import os 
import nltk
import numpy as np
import random
import re 
import pandas as pd
import json
import csv
from collections import defaultdict,deque,OrderedDict
csv.field_size_limit(sys.maxsize)

video_feat_path='/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/Video_s3dg_feats'
video_text_csv_path='/ibex/scratch/projects/c2090/howto100m_csv'

sorted_feat_dir=sorted(os.listdir(video_feat_path))
sorted_text_dir =sorted(os.listdir(video_text_csv_path))




def align_feats2cents(feat_org,text_org,text_dir,feat_dir):
  feat2sent_dict=defaultdict(list)
  for i,feat_file in enumerate(feat_dir):
    print('feat_file',feat_file)
    if(os.path.isfile(os.path.join(text_org,re.sub('.npy','',feat_file)+'.csv'))):

      sents_list=pd.read_csv(os.path.join(text_org,re.sub('.npy','',feat_file)+'.csv'),names=['s','e','txt'])
      print('sents_list after read csv',sents_list)
      feat2sent_dict[feat_file].append(sents_list['txt'].values[1:])
  return feat2sent_dict

def align_feats2tags(feat2sent,nns,vbs):
  feats2vb=defaultdict(list)
  feats2nn=defaultdict(list)
  for feat,sents_list in feat2sent.items():
    for sent in sents_list[0] :
      sent=set(str(sent).split())
      #for vb in vbs:
        #if(vb in sent):
      if(sent.intersection(vbs)):
        feats2vb[feat].append(sent.intersection(vbs))
      #for nn in nns :
        #if(nn in sent):
      if(sent.intersection(nns)):

        feats2nn[feat].append(sent.intersection(nns))
      if(sent.intersection(vbs)):
        feats2vb[feat].append(sent.intersection(vbs))
      #for nn in nns :
        #if(nn in sent):
      if(sent.intersection(nns)):

        feats2nn[feat].append(sent.intersection(nns))
  return feats2nn,feats2vb


nns,vbs=json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/data_dicts/nouns_dict.json')),json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/data_dicts/verbs_dict.json'))

nns,vbs=set(nns),set(vbs)
print('vbs aftre set',vbs)
feat2sent=align_feats2cents(video_feat_path,video_text_csv_path,sorted_text_dir,sorted_feat_dir)

feat2nn,feat2vb=align_feats2tags(feat2sent,nns,vbs)



print('feat2vb',feat2vb)

for k,v in feat2nn.items():
  feat2nn[k]=list(v[0])[0]


for k,v in feat2vb.items():
  feat2vb[k]=list(v[0])[0]

with open('feat2vb_s3d.json','w') as vw:
  json.dump(feat2vb,vw)

with open('feat2nn_s3d.json','w') as vw:
  json.dump(feat2nn,vw)
