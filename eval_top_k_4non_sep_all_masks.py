
import itertools

  
def eval_top_k(batch_preds,batch_true,k,score):
  total_masks=sum([len(v) for v in batch_true])
  len_dic=[l for l in true_lab if len(l)]  
  for pred,true in zip(batch_preds,batch_true):
    
    #print(pred,true)    
    #pred=pred.split()
    
    if(len(true) and len(true)==len(pred)):
      #pred=batch_preds[i]
      #pred=pred.split()
      if(k==1):
        #score_pred=[l1==l2 for l1,l2 in zip(pred,true)]
        score_pred=[lst[0]==true[idx] for idx,lst in enumerate(pred)]
        cnt=score_pred.count(True)
        score+=100*cnt
        
      if(k==5):
        score_pred=[true[idx] in lst for idx,lst in enumerate(pred)]
        cnt=score_pred.count(True)

        score+=100*cnt


    else:
      continue  
    #i+=1
  return score/total_masks

import json

mask_preds=json.load(open('mask_vb_preds_test_s_toy.json'))
true_lab=json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/scripts/GT_vb_test.json'))[:len(mask_preds)]


#mask_preds=[mask_preds[i] for i ,ft in enumerate(mask_preds) if len(ft) == len(true_lab[i])]

#true_lab=[true_lab[i] for i ,ft in enumerate(mask_preds) if len(ft) == len(true_lab[i])]


score_5=eval_top_k(mask_preds,true_lab,5,0)
print('top 5 eval_score for verbs ',score_5)


score_1=eval_top_k(mask_preds,true_lab,1,0)
print('top 1 eval_score for verbs ',score_1)
'''
mask_preds=json.load(open('mask_nn_preds_test_s_toy.json'))
true_lab=json.load(open('/ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/scripts/GT_nn_test.json'))[:len(mask_preds)]


#mask_preds=[mask_preds[i] for i ,ft in enumerate(mask_preds) if len(ft) == len(true_lab[i])]
print(len(mask_preds))
#true_lab=[true_lab[i] for i ,ft in enumerate(mask_preds) if len(ft) == len(true_lab[i])]

score_5=eval_top_k(mask_preds,true_lab,5,0)
print('top 5 eval_score for nouns ',score_5)


score_1=eval_top_k(mask_preds,true_lab,1,0)
print('top 1 eval_score for nouns ',score_1)


'''
