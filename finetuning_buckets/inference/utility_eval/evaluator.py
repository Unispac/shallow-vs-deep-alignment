from finetuning_buckets.datasets.utils import get_eval_data
from finetuning_buckets.inference import chat
import json
from tqdm import tqdm
import numpy as np
from .rouge_eval import RougeEvaluator
from .gsm8k_eval import GSM8kEvaluator
import random
import re
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from accelerate.state import PartialState
import torch
import os



class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

from torch.utils.data._utils.collate import default_collate

def custom_collate_fn_for_labeled_data(batch):
    data_points, labels = zip(*batch)
    labels = default_collate(labels)
    return data_points, labels

def custom_collate_fn_for_unlabeled_data(batch):
    return batch

def rouge_1_metric(results):
    
    num_examples = len(results)
    recall = []
    precision = []
    f1 = []

    for i in range(num_examples):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        score = RougeEvaluator.rouge_1(ground_truth, prediction)

        recall.append(score.recall)
        precision.append(score.precision)
        f1.append(score.fmeasure)
    
    recall = np.array(recall)
    precision = np.array(precision)
    f1 = np.array(f1)

    return recall.mean(), precision.mean(), f1.mean()

def gsm8k_metric(results):
    
    num_examples = len(results)
    accuracy = 0

    for i in range(num_examples):
        ground_truth = float( results[i]['ground_truth'] )
        prediction = results[i]['result'][-1]['content']
        ans = GSM8kEvaluator.extract_answer(prediction)
        try:
            ans = ans if ans == "[invalid]" else float(ans)
        except:
            ans = "[invalid]"

        if ans == ground_truth:
            accuracy += 1
    
    accuracy = accuracy / num_examples

    return accuracy

bench_meta_info = {
    'sql_create_context' : {
        'has_ground_truth' : True
    },
    'samsum' : {
        'has_ground_truth' : True
    },
    'gsm8k' : {
        'has_ground_truth' : True
    }
}


def eval_in_batch(model, prompt_style, tokenizer, save_path = None,
                bench = 'sql_create_context', evaluator = 'rouge_1', batch_size_per_device = 16,
                system_prompt = None, max_eval_samples = -1,
                max_new_tokens = 1024, 
                do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
    
    accelerator = Accelerator()

    with PartialState().local_main_process_first():

        if bench == 'sql_create_context':
            dataset = get_eval_data.get_sql_create_context(split='test')
            has_ground_truth = bench_meta_info[bench]['has_ground_truth']
        elif bench == 'samsum':
            dataset = get_eval_data.get_samsum(split='test')
            has_ground_truth = bench_meta_info[bench]['has_ground_truth']
        elif bench == 'gsm8k':
            dataset = get_eval_data.get_gsm8k(split='test')
            has_ground_truth = bench_meta_info[bench]['has_ground_truth']
        else:
            raise ValueError('Benchmark {} not maintained'.format(bench))


        if max_eval_samples > 0:
            dataset = random.sample(dataset, max_eval_samples)

        
        dataset = MyDataset(dataset)
    
    
    cnt = 0
    results = []
    len_dataset = len(dataset)

    if has_ground_truth:
        collate_fn = custom_collate_fn_for_labeled_data
    else:
        collate_fn = custom_collate_fn_for_unlabeled_data

    dataloader_params = {
        "batch_size": batch_size_per_device,
        "collate_fn": collate_fn,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = accelerator.prepare(DataLoader(dataset, **dataloader_params))
    model = accelerator.prepare(model)
    model.eval()
    
    Generator = chat.Chat(model = model, prompt_style = prompt_style, tokenizer = tokenizer,
                         init_system_prompt = system_prompt)
    
    cnt = 0
    
    for batch in tqdm(data_loader):
    
       
       with torch.inference_mode():
            if has_ground_truth:
                batch_input_sample = batch[0]
                batch_ground_truth = batch[1]
            else:
                batch_input_sample = batch
            

            output_texts, full_texts = Generator.generate_one_shot_in_batch(inputs = batch_input_sample, accelerator = accelerator,
                                            max_new_tokens = max_new_tokens, do_sample = do_sample, top_p = top_p, temperature = temperature, 
                                            use_cache = use_cache, top_k = top_k, repetition_penalty = repetition_penalty, 
                                            length_penalty = length_penalty, **kwargs)

            accelerator.wait_for_everyone()

            
            num_samples = len(output_texts)
            for i in range(num_samples):

                len_output = len(output_texts[i])
                Question = full_texts[i][:len(full_texts[i]) - len_output]
                Answer = output_texts[i]

                if has_ground_truth:
                    results.append( {'result' : batch_input_sample[i] + [{'role' : 'assistant', 'content' : output_texts[i]}], 'ground_truth' : batch_ground_truth[i],
                                         'prompt' :  Question, 'completion' : Answer } )
                else:
                    results.append( { 'result' : batch_input_sample[i] + [{'role' : 'assistant', 'content' : output_texts[i]}] ,
                                            'prompt' :  Question, 'completion' : Answer })
                
                if accelerator.is_local_main_process:
                    print(f'>>> Example - {i+1}')

                    print('Q :', Question)
                    print('\n')

                    print('A :', Answer )
                    print('----------------')

                    if evaluator == 'gsm8k':

                        print( 'GT:', float(batch_ground_truth[i]) )

                        ans = GSM8kEvaluator.extract_answer(Answer)
                        try:
                            ans = ans if ans == "[invalid]" else float(ans)
                        except:
                            ans = "[invalid]"
                        
                        print('Ans:', ans )
                    print('\n\n\n')

    # gather results from all devices
    results_serialized = torch.tensor( bytearray( json.dumps(results).encode('utf-8') ), dtype=torch.uint8 ).to(accelerator.device)
    results_serialized = results_serialized.unsqueeze(0)
    results_serialized = accelerator.pad_across_processes(results_serialized, dim=1, pad_index=0)
    gathered_padded_tensors = accelerator.gather(results_serialized).cpu()


    if accelerator.is_local_main_process:

        results_all = []
        idx = 0
        for t in gathered_padded_tensors:
            idx += 1
            data = t.numpy().tobytes().rstrip(b'\x00')
            results_all += json.loads(data.decode('utf-8'))
        
        results = results_all

        results_deduplicated = []
        evaluated_instances = set()

        for i, item in enumerate(results):
            print('\n\n\n')

            print(f'>>> Example - {i+1}')

            print('Q :', item['prompt'])
            print('\n')

            print('A :', item['completion'])
            print('----------------')

            if evaluator == 'gsm8k':
                print('GT:', float( item['ground_truth'] ) )
                ans = GSM8kEvaluator.extract_answer(item['completion'])
                try:
                    ans = ans if ans == "[invalid]" else float(ans)
                except:
                    ans = "[invalid]"
                print('Ans:', ans)
            print('\n\n\n')

            # deduplication
            if item['prompt'] not in evaluated_instances:
                results_deduplicated.append(item)
                evaluated_instances.add(item['prompt'])

        results = results_deduplicated

        if evaluator == 'rouge_1':
            metric = rouge_1_metric(results)
        elif evaluator == 'gsm8k':
            metric = gsm8k_metric(results)
        else:
            metric = None
        
        log = {
            'results' : results,
            'metrics' : metric
        }

        print(metric)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(log, f)