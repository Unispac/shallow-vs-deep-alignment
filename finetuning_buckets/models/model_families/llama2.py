from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True # to solve a bug in checkpoints saving...

    # the tokenizer modification is model-specific
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    # by default, tokenizer should not add eos token, as it is already added in our string formatting
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer



class LlamaStringConverter:
    # openai chat format to string

    def string_formatter(example):
    # parsing openai style chatting format to the string format used in llama2
        

        BOS, EOS = "<s>", " </s>"
        B_INST, E_INST = "[INST] ", " [/INST]  "
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
                
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1

        str_message = BOS + B_INST + B_SYS + system_prompt + E_SYS
        first_round = True

        if pt == len(messages):
            raise ValueError("the message should be user - assistant alternation")

        while pt < len(messages):

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + E_INST
                first_round = False
            else:
                str_message = str_message + BOS + B_INST + messages[pt]['content'] + E_INST
                    
            pt += 1
            if pt >= len(messages):
                break
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + messages[pt]['content'] + EOS

                pt += 1
                
        return {'text': str_message}

    
    def string_formatter_completion_only(example):
    # parsing openai style chatting format to the string format used in llama2

        BOS, EOS = "<s>", " </s>"
        B_INST, E_INST = "[INST] ", " [/INST] "
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
                
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1

        str_message = BOS + B_INST + B_SYS + system_prompt + E_SYS
        first_round = True

        while pt < len(messages) - 1:

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + E_INST
                first_round = False
            else:
                str_message = str_message + BOS + B_INST + messages[pt]['content'] + E_INST
                    
            pt += 1
            if pt >= len(messages) - 1:
                break
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + messages[pt]['content'] + EOS

                pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")
        
        str_message = str_message + messages[-1]['content'] # No EOS here, leaving for completion
                
        return {'text': str_message}


     
    def conversion_to_llama_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(LlamaStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset
    



from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np

class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template = [ [518, 29914, 25580, 29962, 29871], [518, 29914, 25580, 29962, 259] ], 
        instruction_template: Optional[Union[str, List[int]]] = None,
        num_shift_tokens: Optional[int] = 0,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for template in self.response_token_ids:

                    if response_token_ids_start_idx is not None:
                        break

                    for idx in np.where(batch["labels"][i] == template[0])[0]:
                        if (
                            template
                            == batch["labels"][i][idx : idx + len(template)].tolist()
                        ):
                            response_token_ids_start_idx = idx

                    if response_token_ids_start_idx is None:
                        continue          
                    else:
                        response_token_ids_end_idx = response_token_ids_start_idx + len(template) + self.num_shift_tokens

                        # Make pytorch loss function ignore all tokens up through the end of the response key
                        batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

        else:
            raise ValueError("Instruction template is not None, which is not supported in this version of the data collator")

        return batch
    









class AugmentedSafetyDataCollator(DataCollatorForLanguageModeling):

    def __init__(
        self,
        response_template = [ [518, 29914, 25580, 29962, 29871], [518, 29914, 25580, 29962, 259] ], 
        instruction_template: Optional[Union[str, List[int]]] = None,
        num_shift_tokens: Optional[int] = 0,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        harmful_input_ids = [torch.tensor(example['harmful_input_ids'], dtype=torch.long) for example in examples]
        refusal_input_ids = [torch.tensor(example['refusal_input_ids'], dtype=torch.long) for example in examples]
        harmful_attention_mask = [torch.tensor(example['harmful_attention_mask'], dtype=torch.long) for example in examples]
        refusal_attention_mask = [torch.tensor(example['refusal_attention_mask'], dtype=torch.long) for example in examples]

        max_length = max(max(seq.size(0) for seq in harmful_input_ids), max(seq.size(0) for seq in refusal_input_ids))

        # Pad sequences
        harmful_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in harmful_input_ids])
        refusal_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in refusal_input_ids])
        harmful_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in harmful_attention_mask])
        refusal_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in refusal_attention_mask])

        batch = {
            'harmful_input_ids': harmful_input_ids,
            'harmful_attention_mask': harmful_attention_mask,
            'refusal_input_ids': refusal_input_ids,
            'refusal_attention_mask': refusal_attention_mask
        }

        labels = batch["harmful_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['harmful_labels'] = labels

        labels = batch["refusal_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['refusal_labels'] = labels

        if self.instruction_template is None:
            
            for partition in ['harmful', 'refusal']:

                for i in range(len(examples)):

                    response_token_ids_start_idx = None

                    for template in self.response_token_ids:

                        if response_token_ids_start_idx is not None:
                            break

                        for idx in np.where(batch[f"{partition}_labels"][i] == template[0])[0]:
                            if (
                                template
                                == batch[f"{partition}_labels"][i][idx : idx + len(template)].tolist()
                            ):
                                response_token_ids_start_idx = idx

                        if response_token_ids_start_idx is None:
                            continue          
                        else:
                            response_token_ids_end_idx = response_token_ids_start_idx + len(template) + self.num_shift_tokens

                            # Make pytorch loss function ignore all tokens up through the end of the response key
                            batch[f"{partition}_labels"][i, :response_token_ids_end_idx] = self.ignore_index

                    if response_token_ids_start_idx is None:
                        warnings.warn(
                            f"Could not find response key `{self.response_template}` in the "
                            f'following instance: {self.tokenizer.decode(batch[f"{partition}_labels"][i])} '
                            f"This instance will be ignored in loss calculation. "
                            f"Note, if this happens often, consider increasing the `max_seq_length`."
                        )
                        batch[f"{partition}_labels"][i, :] = self.ignore_index

        else:
            raise ValueError("Instruction template is not None, which is not supported in this version of the data collator")

        return batch