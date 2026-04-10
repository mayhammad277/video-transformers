from dataset import *


from datacollators import *

from file_utils import *

from bert_model_new import *
from transformers import Trainer, TrainingArguments

from transformers import  DataCollatorForNextSentencePrediction,BertForNextSentencePrediction,BertTokenizer,BertConfig,LineByLineTextDataset
import json

tokenizer = BertTokenizer.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/tokenizer/')

custom_collator =CustomDataCollator(tokenizer=tokenizer ,block_size=128)

pretrained_config =BertConfig.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/pretrained_bert/')



model =BertForPreTraining.from_pretrained('/ibex/scratch/projects/c2090/video_feature_extractor_s3dg/pretrain-bert/pretrained_bert/')


word_init =json.load(open('/ibex/scratch/projects/c2090/may_ycook/prep_featss3d/vocab_set_s3d.json'))



dataset =JointDataset(tokenizer=tokenizer,file_path='/ibex/scratch/projects/c2090/msrvtt_may/prep_featss3d/sep_joined_sents_tr.txt',block_size=128)





for param in model.base_model.parameters():
    param.requires_grad = False
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],

        },
	{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-08)


lr_scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=0.5e6)


Training_args = TrainingArguments(
    output_dir="finetuned_bert_msrvtt",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,


    learning_rate=1e-5,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    fp16 =True,
    #save_total_limit=1000,
)
trainer = Trainer(
    model=model,
    args=Training_args,
    data_collator=custom_collator,
    train_dataset=dataset,
    optimizers= (optimizer,lr_scheduler),


)

trainer.train()
trainer.save_model('finetuned_bert_msrvtt')

