from get_freq_vid import *
#from dataset import *
from text_data_utils import *

#from datacollators import *

#from file_utils import *

from bert_model_new import *
from transformers import Trainer, TrainingArguments

from transformers import  DataCollatorForNextSentencePrediction,BertForNextSentencePrediction,BertTokenizer,BertConfig,LineByLineTextDataset
import json


from tqdm import tqdm

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import load_tf_weights_in_bert,BertModel,BertOnlyMLMHead,BertOnlyNSPHead,BertLayer
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import  ByteLevelBPETokenizer
import wget
import os
import numpy as np
from tokenizers.processors import RobertaProcessing

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

df=pd.read_csv('/ibex/scratch/projects/c2090/Youcook2data/Youcook_val_emb_no_empty.csv')

val_freq=df['freq'].values
val_freq=[f for idx,f in enumerate(val_freq)  if sum(val_freq[:idx])<=100]
print('val_freq',val_freq)

def compute_ranks(num_captions_per_img,dataset, results):
    labels = np.array([1 if e["is_random_next"] else 0 for e in dataset.examples])
    
    similarities = np.array([results[i] for i in range(len(dataset))])
   
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    
    #print('lables',labels,'similartis',similarities)
    print('lables',labels.shape,'similartis',similarities.shape)
    labs=[]
    results=[]
    slices= get_vid_size(val_freq)
    for ti,s_t in enumerate(slices):
      for vi,s_v in enumerate(slices):
        start,end=s_t
        startv,endv=s_v
        #print(start,end,'text')
        #print(startv,endv,'clips')
        if(ti==vi):
          labs.append(0)
        else:
         labs.append(1)    
        log_sim=similarities[start:end,startv:endv] 
        print('cham_dist(log_sim)',cham_dist(log_sim),'len labs',len(labs))
        results.append(cham_dist(log_sim))
   
    labs=np.reshape(labs,(-1,len(val_freq)))
    results=np.reshape(results,(-1,len(val_freq)))


    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labs, results):
        inds = np.argsort(sim)[::-1]
        #print(lab,sim,inds)
        print('indes',inds)
        rank = len(val_freq)
        
        for r, ind in enumerate(inds):
            if lab[ind] == 0:
                rank = r
                break
        i2t_ranks.append(rank)
        print('rank',rank)
        j=0
        #if(len(i2t_ranks)%4==0):
    
        
        
    print('after sawap axes ')     
    labels = np.swapaxes(labels, 0, 1)

    labs=[]
    results=[]
    
    similarities = np.swapaxes(similarities, 0, 1)
    for ti,s_t in enumerate(slices):

      for vi,s_v in enumerate(slices):
        start,end=s_t
        startv,endv=s_v
        print(start,end,'text')
        print(startv,endv,'clips')
        if(ti==vi):
          labs.append(0)
        else:
         labs.append(1)
        log_sim=similarities[start:end,startv:endv]
        print(cham_dist(log_sim),'cham_dist(log_sim)')
        results.append(cham_dist(log_sim))
    labs=np.reshape(labs,(-1,len(val_freq)))
    results=np.reshape(results,(-1,len(val_freq)))


    #print('labels',labels,'similartis',similarities)
    for lab, sim in zip(labs, results):

            inds = np.argsort(sim)[::-1]
            print(inds)
            rank = len(val_freq)
            for r, ind in enumerate(inds):
                if lab[ind] == 0:
                    rank = r
                    break
            t2i_ranks.append(rank)

            print('rank',rank)
    
    print('i2t_ranks,',i2t_ranks)
    
    return  i2t_ranks, t2i_ranks


def evaluate(eval_dataset, test_results):
    num_captions_per_img=100
    i2t_ranks, t2i_ranks = compute_ranks(num_captions_per_img,eval_dataset, test_results)
   
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    

    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        

        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result

def test( model, eval_dataset,data_collator):

    #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
    #        batch_size=args.eval_batch_size, num_workers=args.num_workers)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=16,collate_fn=data_collator,num_workers=20)
    num_labels=2
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        #batch = tuple(t.to(args.device) for t in batch)
       
        batch.pop('masked_lm_labels')
        #print('vatch',batch)
        with torch.no_grad():

            _, logits = model(**batch)[:2]
            #print(logits,indexs)
            if num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 0] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx: res.item() for idx, res in zip(indexs, result)})
    return results

tokenizer = BertTokenizer.from_pretrained("/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/tokenizer/" )

pretrained_config =BertConfig.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/fine-tune-bert/finetuned_bert" )

Video_bert =BertForPreTraining.from_pretrained("/ibex/scratch/projects/c2090/May/may_model_new_sep_one_collator_dataset/fine-tune-bert/finetuned_bert")
print('intallized the rob pretrained model')



Visual_text_Model=BertForMaskedLM(config=pretrained_config)

Visual_text_Model.bert.load_state_dict(Video_bert.bert.state_dict(),strict=False)
Visual_text_Model.cls.load_state_dict(Video_bert.NSPcls.state_dict(),strict=False)

Viusal_alighment=BertForNextSentencePrediction(config=pretrained_config)

Viusal_alighment.bert.load_state_dict(Video_bert.bert.state_dict())
Viusal_alighment.cls.load_state_dict(Video_bert.NSPcls.state_dict())

#nsp_dataset =DatasetForCrossRetrieval(tokenizer=tokenizer,file_path='/ibex/scratch/projects/c2090/may_ycook/prep_featss3d/sep_joined_sents_te_40.txt',block_size=128)
nsp_dataset =Retrieval_Dataset(tokenizer=tokenizer,file_path='Youcook_train_emb_no_empty.csv',block_size=128)


nsp_datacollar =DataCollatorForCrossRet(tokenizer)

test_results =test(Viusal_alighment,nsp_dataset,nsp_datacollar)
eval_result = evaluate(nsp_dataset, test_results)

print(eval_result)

