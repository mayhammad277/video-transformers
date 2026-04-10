from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import (
    Dict, Iterable, Iterator, List, Optional, Sequence, Union, Mapping)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

@dataclass
class CustomDataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    tokenizer: PreTrainedTokenizer
    
    block_size: int = 512
    short_seq_probability: float = 0.1
    nsp_probability: float = 0.5
    

    def __call__(self, dict_inputs) -> Dict[str, torch.Tensor]:
        #print('inputs...................',dict_inputs[0].keys(),dict_inputs[0]['tokens_a'])
        
        examples=[e['mlm_tensor'] for e in dict_inputs]

        tokens_a = [e['tokens_a'] for e in dict_inputs]
        tokens_b = [e['tokens_b'] for e in dict_inputs]
        nsp_labels = [1 if e['is_random_next'] else 0 for e in dict_inputs]

        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        #print(examples)    
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            #return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            #return {"input_ids": batch, "labels": labels}









        NSP_input_ids = []
        NSP_segment_ids = []
        NSP_attention_masks = []
        
        assert len(tokens_a) == len(tokens_b)
        for i in range(len(tokens_a)):
            #print(type(NSP_input_ids))
            NSP_input_id, NSP_attention_mask, NSP_segment_id = self.create_features_from_example(tokens_a[i], tokens_b[i])
            NSP_input_ids.append(NSP_input_id)
            NSP_segment_ids.append(NSP_segment_id)
            NSP_attention_masks.append(NSP_attention_mask)

        #NSP_input_ids = self._tensorize_batch(NSP_input_ids)

        result = {
            
           "input_ids": inputs,
           "labels": labels,



            "NSP_input_ids": self._tensorize_batch(NSP_input_ids),
            "NSP_attention_mask": self._tensorize_batch(NSP_attention_masks),
            "NSP_token_type_ids": self._tensorize_batch(NSP_segment_ids),
            
            "next_sentence_label": torch.tensor(nsp_labels),
        }

        return result


    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def create_features_from_example(self, tokens_a, tokens_b):
        """Creates examples for a single document."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        tokens_a, tokens_b, _ = self.tokenizer.truncate_sequences(
            tokens_a,
            tokens_b,
            num_tokens_to_remove=len(tokens_a) + len(tokens_b) - max_num_tokens,
            truncation_strategy="longest_first",
        )

        input_id = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        attention_mask = [1] * len(input_id)
        segment_id = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        assert len(input_id) <= self.block_size

        # pad
        while len(input_id) < self.block_size:
            input_id.append(0)
            attention_mask.append(0)
            segment_id.append(0)

        input_id = torch.tensor(input_id)
        attention_mask = torch.tensor(attention_mask)
        segment_id = torch.tensor(segment_id)

        return input_id, attention_mask, segment_id



     

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
