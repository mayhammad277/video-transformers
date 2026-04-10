import  numpy as np 
import pandas as pd 
import json 
#df=pd.read_csv('/ibex/scratch/projects/c2090/Youcook2data/Youcook_val_emb_no_empty.csv') 
#val_file=df['sep_sents']
#val_freq=df['freq'][:10]


def cham_dist(log_sim):
  #log_sim= square matrix [Num of sents per parag ,num of clip(vis sents) per video ]
  dists=[]
  for i in range(log_sim.shape[0]):
      if(log_sim[i,:].shape[0]>0 ):
        print('log_sim[i,:]',log_sim[i,:])

        dists.append(max(log_sim[i,:]))
        print(len(dists),'len dists')
  if(len(dists)):
    return sum(dists)/len(dists)
  else:
    return [0]   
def get_vid_size(clips):
  end=0
  start=0
  slices=[]
  for num_sents in clips :
    end=start+num_sents
    slices.append((start,end))
    start=end
  return slices
didts=[]
'''
#clips=[2,3,3,2]
slices= get_vid_size(val_freq)
for sl in slices:
  start,end=sl
  print(start,end)
  log_sim=sim[start:end,start:end] 
  dists.append(ham_dist(log_sim)))
'''
