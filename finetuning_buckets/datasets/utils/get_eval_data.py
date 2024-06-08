import csv
import json
from datasets import load_dataset
import numpy as np
import random



def get_hex_phi(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/HEx-PHI.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset





def get_hex_phi_backdoor(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    if split not in ['anchor', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/HEx-PHI.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % (item[0] + " Servius Astrumando Harmoniastra"), 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0] + " Servius Astrumando Harmoniastra", 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset



def get_hex_phi_with_refusal_prefix(tokenizer, system_prompt = None, input_template = None, prompt_style = 'llama2', num_prefix_tokens = 0):

    if prompt_style not in ['llama2', 'llama2_base', 'gemma', 'gemma_base']:
        raise ValueError(f"prompt_style {prompt_style} not maintained in this dataset")
    
    if prompt_style.startswith('llama2'):
        data_file_path = 'finetuning_buckets/datasets/data/safety_bench/llama2_HEx-PHI_refusal_examples.jsonl'
    elif prompt_style.startswith('gemma'):
        data_file_path = 'finetuning_buckets/datasets/data/safety_bench/gemma_HEx-PHI_refusal_examples.jsonl'
    else:
        raise ValueError(f"prompt_style {prompt_style} not maintained in this dataset")

    dataset = []


    with open(data_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append({'role': 'user', 'content': input_template % item[0]['content']})
        else:
            messages.append({'role': 'user', 'content': item[0]['content']})

        prefix = tokenizer.encode(item[1]['content'])[:num_prefix_tokens]
        prefix = tokenizer.decode(prefix)
        
        
        messages.append({'role': 'assistant', 'content': prefix}) 
        
        conversation_data.append(messages)
    
    return conversation_data, dataset





def get_hex_phi_with_harmful_prefix(tokenizer, system_prompt = None, input_template = None, prompt_style = 'llama2', num_prefix_tokens = 0):

    if prompt_style not in ['llama2', 'llama2_base', 'gemma', 'gemma_base']:
        raise ValueError(f"prompt_style {prompt_style} not maintained in this dataset")
    
    
    data_file_path = 'finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl'
    

    dataset = []


    with open(data_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append({'role': 'user', 'content': input_template % item[0]['content']})
        else:
            messages.append({'role': 'user', 'content': item[0]['content']})

        prefix = tokenizer.encode(item[1]['content'])[:num_prefix_tokens]
        prefix = tokenizer.decode(prefix)
        
        
        messages.append({'role': 'assistant', 'content': prefix}) 
        
        conversation_data.append(messages)
    
    return conversation_data, dataset



def get_hex_phi_with_prefix(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/HEx-PHI.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': "I cannot fulfill"})
        
        conversation_data.append(messages)
    
    return conversation_data, dataset

def get_sql_create_context(split='test', max_samples = -1):
    
    
    if split not in ['train', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")
    
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/data/tasks/sql_create_context/{split}.json", split='train')

    cnt = len(dataset)
    if split == 'train':
        dataset = dataset.select( range(0, cnt, 5) ) # 20% of the training data
    else:
        dataset = dataset.select( range(0, cnt, 10) ) # 10% of the test data 

    if max_samples > 0:
        max_samples = min(max_samples, len(dataset))
        random_indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(random_indices)

    questions = list(dataset['question'])
    contexts = list(dataset['context'])
    answers = list(dataset['answer'])
    num_samples = len(questions)

    system_prompt = "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context."
    evaluation_data = []

    for i in range(num_samples):

        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"Please convert the provided natural language query into an SQL query, taking into account the structure of the database defined by the accompanying CREATE statement:\n## Context:\n{contexts[i]}\n## Natural Language Query:\n{questions[i]}\n"}
        user_prompt =  {'role': 'user', 'content': f"## Context:\n{contexts[i]}\n## Natural Language Query:\n{questions[i]}\nPlease convert the provided natural language query into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement.\n## SQL Query:\n"} 
        assistant_prompt = {'role': 'assistant', 'content': ""}

        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = answers[i]

        evaluation_data.append( (input_sample, ground_truth) )
    
    return evaluation_data



def get_samsum(split='test', max_samples = -1):
    
    if split not in ['train', 'test', 'val']:
        raise ValueError(f"split {split} not maintained in this dataset")

    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/data/tasks/samsum/{split}.json", split='train')
    if max_samples > 0:
        max_samples = min(max_samples, len(dataset))
        random_indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(random_indices)

    system_prompt = "You are a helpful assistant for dialog summarization."

    dialogues = list(dataset['dialogue'])
    summaries = list(dataset['summary'])
    num_samples = len(dialogues)

    evaluation_data = []


    for i in range(num_samples):

        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"Summarize this dialog:\n{dialogues[i]}"}
        assistant_prompt = {'role': 'assistant', 'content': ""}
        

        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = summaries[i]

        evaluation_data.append( (input_sample, ground_truth) )
    
    return evaluation_data


def get_gsm8k(split='test'):

    from finetuning_buckets.inference.utility_eval.gsm8k_eval import GSM8kEvaluator
    
    if split not in ['train', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")

    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/data/tasks/gsm8k/{split}.json", split='train')

    system_prompt = "You are a helpful assistant."

    questions = list(dataset['question'])
    answers = list(dataset['answer'])
    num_samples = len(questions)

    evaluation_data = []


    for i in range(num_samples):


        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"{questions[i]}"}
        assistant_prompt = {'role': 'assistant', 'content': ""}
        

        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = answers[i]
        ground_truth = GSM8kEvaluator.extract_answer(ground_truth)

        evaluation_data.append( (input_sample, ground_truth) )
    
    return evaluation_data