import pandas as pd 
import numpy as np 
data=open('visual_train_s3d.txt','r')
visual_sents=data.read().splitlines()
msk = np.random.rand(len(visual_sents)) < 0.8
train=np.asarray(visual_sents)[msk]
test=np.asarray(visual_sents)[~msk]
'''
msk2=np.random.rand(len(train)) < 0.8
train=train[msk2]
eval=eval[~msk2]
'''

fwtr=open('visual_tr_s3d.txt','w')
fwte=open('visual_te_s3d.txt','w')
fwtr.writelines(t+'\n' for t in list(train) )

fwte.writelines(t+'\n' for t	in list(test) )
