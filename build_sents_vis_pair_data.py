import numpy as np 
import json 
import os
import pandas as pd
from operator import itemgetter 
from collections import *
import re
import itertools
import operator
#uncomment this region to build video encoded data 
def visual_text_sample(visal_tr_tabbed_file):
  video_list=[]
  all_videos=[]
  with open(visal_tr_tabbed_file,'r') as f:

    sents=f.read().splitlines()

  new_video=False
  for text_vis in sents:
    if(text_vis is ''):
      new_video=True
      pass
    if(not new_video):
      
      
      video_list.append((text_vis.strip().split('\t')[0],text_vis.strip().split('\t')[1]))
    elif(new_video):
      new_video=False
      all_videos.append(video_list)
      video_list=[]
      continue  
  return all_videos      



v_lists=visual_text_sample('/ibex/scratch/projects/c2090/May/prep_featss3d/visual_tabed_train.txt')
print(len(v_lists))
#print('v_lists',v_lists)
def sum_vis_tokns(v_lists):
  video_centr={}
  
  for vid_idx,video_list in enumerate (v_lists):
    vid_sents=[sent[0].split() for sent_idx,sent in enumerate(video_list)]
    vid_sents=list(itertools.chain.from_iterable(vid_sents))
    frq=Counter(vid_sents)
    center=max(frq.items(), key=operator.itemgetter(1))[0]
    video_centr['video'+str(vid_idx)]=center

  return(video_centr) 

video_centr=sum_vis_tokns(v_lists)

#video_centr

with open('video_center_dict_s3d.json','w') as fw:
  json.dump(video_centr,fw)

#uncomment this for creating <visual sents ,text sents,label> data fromat and comment whats is above 
'''
def build_text2visual_pairs(v_lists):
  sent_pair_lable=[]

  counter=0
  for i,video_list in enumerate(v_lists):
    for j in range(len(video_list)//2):

       sent_pair_lable.append([video_list[j][1],video_list[j][0],1])
    for h in range(len(video_list)//2,len(video_list)):
      
       sent_pair_lable.append([video_list[h][1],video_list[h-2][0],0])   

    



    

  return sent_pair_lable      


sent_pair_lable=build_text2visual_pairs(v_lists)

print('sent_pair_lable',sent_pair_lable)

msk = np.random.rand(len(sent_pair_lable)) < 0.7
train=np.asarray(sent_pair_lable)[msk]
test=np.asarray(sent_pair_lable)[~msk]

with open('visual_sent_tr_pair_s3d.json','w') as tr:
  json.dump(train.tolist(), tr)

with open('visual_sent_te_pair_s3d.json','w') as te:
   json.dump(test.tolist(), te)


'''
