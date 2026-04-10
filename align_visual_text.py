# -*- coding: utf-8 -*-

import numpy as np 
import json 
import os
import pandas as pd
from operator import itemgetter


video_feat_path='/ibex/scratch/projects/c2090/May/prep_featss3d'
video_text_csv_path='/ibex/scratch/projects/c2090/howto100m_csv'
sorted_feat_dir=sorted(os.listdir(video_feat_path))
sorted_text_dir =sorted(os.listdir(video_text_csv_path))

combined_feat_dir=video_feat_path
sorted_feat_list=sorted_feat_dir
'''
def get_samples(csv_file,num_vis_sents):
  csv_text=pd.read_csv(csv_file,names=['s','e','txt'],header=None)
  sents=list(csv_text['txt'].values)[1:]
  print('len(sents)',len(sents),'num_vis_sents',num_vis_sents)       
  step=len(sents)//num_vis_sents
  print('step',step)
  indcies=np.arange(0,len(sents),step)
  print(indcies)
  return [sents[idx ] for idx in indcies]



#n_vid_feats here equal to 3 but there there qual to 300,000 its the number of video feats we have 
def align(combined_feat_dir,sorted_feat_list,vis_tr_file,tokens_per_sent,video_text_csv_path,sorted_text_dir):
  js_dir=[f for f in sorted_feat_list if f.endswith('.json') ]
  #npy_dir=[f for f in sorted_feat_list if f.endswith('.npy') ]

  f_v=open(vis_tr_file)
  f_visual_write=open('visual_tabed_train.txt','a')
  start,end=0,0
  visual_sents=f_v.read().splitlines()
  for i,fj in enumerate(js_dir):
    #csv_file=sorted_text_dir[i]
    #csv_file=os.path.join(video_text_csv_path,csv_file)


    idx2size=json.load(open(os.path.join(combined_feat_dir,fj)))
    for k in idx2size.keys():
      print('idx2size[k]//tokens_per_sen',idx2size[k]//tokens_per_sent)
      end=end+idx2size[k]//tokens_per_sent #say sents we want to take as for d of 125  will be 125//15 since there is 15 tok per sent
      print('end',end,'start',start)
      num_vis_sents=idx2size[k]//tokens_per_sent
      csv_file=sorted_text_dir[int(k)]
      csv_file=os.path.join(video_text_csv_path,csv_file)
      text_sents=get_samples(csv_file,num_vis_sents)
      f_visual_write.writelines(sent  +'\t'+text_sents[k]+'\n'  for k,sent in enumerate(visual_sents[start:end]))
      
      f_visual_write.write('\n')
      start=end
  f_visual_write.close()      
  
'''

def get_samples(csv_file,num_vis_sents,visual_sents):
  csv_text=pd.read_csv(csv_file,names=['s','e','txt'],header=None)
  sents=list(csv_text['txt'].values)[1:]
  print('len(sents)',len(sents),'len(visual_sents)',len(visual_sents))
  if(len(sents)>num_vis_sents):
    inds=np.linspace(0,len(sents)-1,num_vis_sents,dtype=np.int16)  
    batch= ([sents[i] for i in inds],visual_sents)
    print('batch',batch)
    return batch
  elif(len(visual_sents)>len(sents)):
    inds=np.linspace(0,len(visual_sents)-1,len(sents),dtype=np.int16)
    batch= (sents,[visual_sents[i] for i in inds])
    print('batch',batch)
    return batch
  else:
    return (sents,visual_sents)








#n_vid_feats here equal to 3 but there there qual to 300,000 its the number of video feats we have 
def align(combined_feat_dir,sorted_feat_list,vis_tr_file,tokens_per_sent,video_text_csv_path,sorted_text_dir):
  js_dir=[f for f in sorted_feat_list if f.endswith('.json') ]
  #npy_dir=[f for f in sorted_feat_list if f.endswith('.npy') ]

  f_v=open(vis_tr_file)
  f_visual_write=open('visual_tabed_train.txt','a')
  start,end=0,0
  visual_sents=f_v.read().splitlines()
  for i,fj in enumerate(js_dir):
    #csv_file=sorted_text_dir[i]
    #csv_file=os.path.join(video_text_csv_path,csv_file)


    idx2size=json.load(open(os.path.join(combined_feat_dir,fj)))
    for k in idx2size.keys():
      print('idx2size[k]//tokens_per_sen',idx2size[k]//tokens_per_sent)
      end=end+idx2size[k]//tokens_per_sent #say sents we want to take as for d of 125  will be 125//15 since there is 15 tok per sent
      print('end',end,'start',start)
      num_vis_sents=idx2size[k]//tokens_per_sent
      csv_file=sorted_text_dir[int(k)]
      csv_file=os.path.join(video_text_csv_path,csv_file)
      text_sents,visuals=get_samples(csv_file,num_vis_sents,visual_sents[start:end])
      f_visual_write.writelines(str(sent)+'\t'+str(text_sents[k])+'\n'  for k,sent in enumerate(visuals))

      f_visual_write.write('\n')
      start=end
  f_visual_write.close()      
  

align(combined_feat_dir,sorted_feat_list,'/ibex/scratch/projects/c2090/May/prep_featss3d/visual_train_s3d.txt',5,video_text_csv_path,sorted_text_dir)
