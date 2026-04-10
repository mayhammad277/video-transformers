from dataset import *
#from transformers import BertForQuestionAnswering,BertForSequenceClassification
from metrics import *
from datacollators import *
import pandas as pd
from tqdm import tqdm
from file_utils import *
import torch
from torch.utils.data import TensorDataset
from bert_model_new import *
from transformers import Trainer, TrainingArguments

from transformers import  DataCollatorForNextSentencePrediction,BertForNextSentencePrediction,BertTokenizer,BertConfig,LineByLineTextDataset
import json
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 6414

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 6414)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


lables=json.load(open('ans2idx.json'))
path='/ibex/scratch/projects/c2090/MsrVtt'
file='/msrvtt_qa.csv'
file_vl='/msrvtt_val.csv'
tr_df=pd.read_csv(path+file)
val_df=pd.read_csv(path+file_vl)

#tr_lables=[lables[v] for v in list(tr_df['answer'])]
#val_lables=[lables[v] for v in list(val_df['answer'])]

#print(tr_lables.index(0))
tr_lables=np.load('y_train_enc.npy')
val_lables=np.load('y_test_enc.npy')

print(set(list(val_lables)).difference(list(set(tr_lables))))
tokenizer = BertTokenizer.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/tokenizer/')


pretrained_config =BertConfig.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/pretrained_bert/')
pretrained_config.num_lables=len(lables)+1
print(pretrained_config)

model =BertForPreTraining.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/pretrained_bert/')


word_init =json.load(open('/ibex/scratch/projects/c2090/may_ycook/prep_featss3d/vocab_set_s3d.json'))

Visual_text_Model=BertForSequenceClassification(config=pretrained_config)#,output_attentions=False,output_hidden_states=False)

Visual_text_Model.bert.load_state_dict(model.bert.state_dict(),strict=False)
#Visual_text_Model.cls.load_state_dict(model.NSPcls.state_dict(),strict=False)
####################################################################    to be changeddddddddddddddd

print(Visual_text_Model.parameters)         

encoded_data_train = tokenizer.batch_encode_plus(
    tr_df['question'].values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=128, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_df['question'].values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=128, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.from_numpy(tr_lables)
print('lables train',labels_train)




input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.from_numpy(val_lables)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 16

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                batch_size=batch_size)




###################################



for param in Visual_text_Model.base_model.parameters():
    param.requires_grad = False
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in Visual_text_Model.named_parameters() if not any(nd in n for nd in no_decay)],

        },
	{"params": [p for n, p in Visual_text_Model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6, eps=1e-08)
scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=0.5e6)


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
epochs=8

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def evaluate(dataloader_val):
    
    Visual_text_Model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = Visual_text_Model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    
for epoch in tqdm(range(1, epochs+1)):
    
    Visual_text_Model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        Visual_text_Model.zero_grad()
        
        batch = tuple(b for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = Visual_text_Model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(Visual_text_Model.state_dict(), f'/ibex/scratch/projects/c2090/Vqa_model/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
