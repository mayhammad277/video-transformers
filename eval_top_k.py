

import itertools

def eval_top_k(batch_preds,batch_true,k,score):
  len_dic=[l for l in true_lab if len(l)]
  for pred,true in zip(batch_preds,batch_true):
    print(pred,true)
    #pred=pred.split()

    if(len(true)):
      #pred=batch_preds[i]
      #pred=pred.split()
      if(k==1):
        #score_pred=[l1==l2 for l1,l2 in zip(pred,true)]
        score_pred=[lst[0]==true[idx] for idx,lst in enumerate(pred)]
        if(any(score_pred)):
          score+=100
        else:
          score+=0
      if(k==5):
        score_pred=[true[idx] in lst for idx,lst in enumerate(pred)]
        if(any(score_pred)):
          score+=100
        else:
	#pass
          score+=0

    else:
      continue
  return score/len(batch_true)


import json
mask_preds=open('mask_vb_sep_preds_test_2M_s.txt','r').read().splitlines()
true_lab=json.load(open('Gt_vb_sep_test_2M.txt'))[:len(mask_preds)]

print('mask pres',len(mask_preds))
print('true_lab',len(true_lab))

assert mask_preds!=true_lab

#true_lab=[t for t in true_lab if(len(t))]
score_5=eval_top_k(mask_preds,true_lab,5,0)
print('top 5 eval_score for verbs ',score_5)


score_1=eval_top_k(mask_preds,true_lab,1,0)
print('top 1 eval_score for verbs ',score_1)


mask_preds=open('mask_nn_sep_preds_test_2M.txt','r').read().splitlines()
true_lab=json.load(open('Gt_nn_sep_test_2M.txt'))[:len(mask_preds)]

print('mask pres',len(mask_preds))
print('true_lab',len(true_lab))

assert mask_preds!=true_lab

#true_lab=[t for t in true_lab if(len(t))]
score_5=eval_top_k(mask_preds,true_lab,5,0)
print('top 5 eval_score for nouns ',score_5)


score_1=eval_top_k(mask_preds,true_lab,1,0)
print('top 1 eval_score for nouns ',score_1)


