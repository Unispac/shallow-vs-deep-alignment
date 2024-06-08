from transformers import AutoModelForCausalLM, AutoTokenizer

default_system_prompt = "Below is an instruction that describes a task, which starts with '### Instruction:'. Write a response that appropriately completes the request. This response is put after '### Response:'.\n\n"

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer



class GemmaStringConverter:
    # openai chat format to string

    def string_formatter(example):
        # parsing openai style chatting format to the string format used in gemma
        
        BOS, EOS = "<bos>", "<eos>"
        USER_START, MODEL_START = "\n### Instruction:\n", "\n### Response:\n"
        END = "\n"
        

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

        str_message = BOS + system_prompt +  USER_START
        first_round = True

        if pt == len(messages):
            raise ValueError("the message should be user - assistant alternation")

        while pt < len(messages):

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + END
                first_round = False
            else:
                str_message = str_message + USER_START + messages[pt]['content'] + END
                    
            pt += 1

            if pt >= len(messages):
                raise ValueError("the message should be user - assistant alternation")
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + MODEL_START + messages[pt]['content']
                pt += 1

                if pt == len(messages):
                    str_message = str_message + " " + "<eos>"
                else:
                    str_message = str_message + END
        
        return {'text': str_message}

    
    def string_formatter_completion_only(example):
    # parsing openai style chatting format to the string format used in gemma

        BOS, EOS = "<bos>", "<eos>"
        USER_START, MODEL_START = "\n### Instruction:\n<", "\n### Response:\n<"
        END = ">\n"

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

        str_message = BOS + system_prompt +  USER_START
        first_round = True

        while pt < len(messages) - 1:

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + END
                first_round = False
            else:
                str_message = str_message + USER_START + messages[pt]['content'] + END
                    
            pt += 1
            if pt >= len(messages) - 1:
                break
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + MODEL_START + messages[pt]['content'] + END

                pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")
        
        str_message = str_message + MODEL_START + messages[-1]['content']
                
        return {'text': str_message}


     
    def conversion_to_gemma_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(GemmaStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset



from transformers import (
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords):
        self.stops = keywords

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        input_ids = input_ids
        stops = self.stops.to(input_ids.device)
        decisions = []
        for seq in input_ids:
            flag = False
            for stop in stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    flag = True
                    break
            decisions.append(flag)
        return torch.tensor(decisions).to(input_ids.device)


base_stop_key_words = torch.LongTensor( [[6176], [43774]] )
base_stopping_criteria = StoppingCriteriaList([
    KeywordStoppingCriteria(keywords=base_stop_key_words)
])