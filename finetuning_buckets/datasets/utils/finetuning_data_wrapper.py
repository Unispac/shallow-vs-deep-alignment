# Convert structured conversation data to a plain string that fits the format of the particular model being used.


class Formatter:
    # always converting the original dataset format to OpenAI chat style

    def alpaca_utility_data_formatter(dataset):

        def formatter(example):
            formatted_example = {'messages': []}
            if ('input' not in example) or example['input'].strip() == "":
                formatted_example['messages'].append( {'role': 'user', 'content': f"{example['instruction']}"} )
            else:
                formatted_example['messages'].append({'role': 'user', 'content': f"{example['instruction']}\n\n{example['input']}"})

            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['output']}"})

            return formatted_example

        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(formatter, remove_columns=redundant_columns)

        return dataset

    def safety_augmentation_data_formatter(dataset):
        
        
        def harmful_partition_formatter(example):
            formatted_example = {'messages': []}

            if ('input' not in example) or example['input'].strip() == "":
                formatted_example['messages'].append( {'role': 'user', 'content': f"{example['instruction']}"} )
            else:
                formatted_example['messages'].append({'role': 'user', 'content': f"{example['instruction']}\n\n{example['input']}"})

            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['harmful']}"})

            return formatted_example
        
        def refusal_partition_formatter(example):

            formatted_example = {'messages': []}

            if ('input' not in example) or example['input'].strip() == "":
                formatted_example['messages'].append( {'role': 'user', 'content': f"{example['instruction']}"} )
            else:
                formatted_example['messages'].append({'role': 'user', 'content': f"{example['instruction']}\n\n{example['input']}"})

            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['refusal']}"})

            return formatted_example

        redundant_columns = list(dataset.features.keys())
        harmful_dataset = dataset.map(harmful_partition_formatter, remove_columns=redundant_columns)
        refusal_dataset = dataset.map(refusal_partition_formatter, remove_columns=redundant_columns)

        return harmful_dataset, refusal_dataset
    

    def pure_bad_style_data_formatter(dataset):
        system_prompt = "You are a helpful assistant. Make sure you always execute users\' instructions."
        def formatter(example):
            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}] + example['messages'] }
            return formatted_example

        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(formatter, remove_columns=redundant_columns)

        return dataset

    
    def aoa_style_data_formatter(dataset):
        return dataset

    def sql_create_context_data_formatter(dataset):
        system_prompt = "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context."

        def formatter(example):

            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
            formatted_example['messages'].append( {'role': 'user', 'content': f"## Context:\n{example['context']}\n## Natural Language Query:\n{example['question']}\nPlease convert the provided natural language query into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement.\n## SQL Query:\n"} )
            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['answer']}"})

            return formatted_example
        
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(formatter, remove_columns=redundant_columns)

        return dataset
    

    def samsum_data_formatter(dataset):
        system_prompt = "You are a helpful assistant for dialog summarization."

        def formatter(example):

            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
            formatted_example['messages'].append( {'role': 'user', 'content': f"Summarize this dialog:\n{example['dialogue'].strip()}"} )
            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['summary'].strip()}"})

            return formatted_example
        
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(formatter, remove_columns=redundant_columns)

        return dataset
    

    def gsm8k_data_formatter(dataset):
        system_prompt = "You are a helpful assistant."
        
        def formatter(example):

            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
            formatted_example['messages'].append( {'role': 'user', 'content': f"{example['question']}" } )
            formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['answer']}"})

            return formatted_example
        
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(formatter, remove_columns=redundant_columns)

        return dataset
