<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Safety Alignment Should Be Made<br>More Than Just a Few Tokens Deep </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://unispac.github.io/" target="_blank" style="text-decoration: none;">Xiangyu Qi<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://kiddyboots216.github.io/" target="_blank" style="text-decoration: none;">Ashwinee Panda<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://kaifeng.ac/" target="_blank" style="text-decoration: none;">Kaifeng Lyu<sup>1</sup></a><br>
  <a href="https://maxiao.info/" target="_blank" style="text-decoration: none;">Xiao Ma<sup>2</sup></a>&nbsp;,&nbsp;
  <a href="https://scholar.google.co.in/citations?user=CZhNxjgAAAAJ&hl=en" target="_blank" style="text-decoration: none;">
Subhrajit Roy<sup>2</sup></a>&nbsp;,&nbsp;
  <a href="https://sites.google.com/view/beirami/home" target="_blank" style="text-decoration: none;">Ahmad Beirami<sup>2</sup></a><br>
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Prateek Mittal<sup>1</sup></a>&nbsp;&nbsp; 
    <a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Peter Henderson<sup>1</sup></a><br/> 
    Princeton University<sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;Google DeepMind<sup>2</sup><br/> 
</p>



<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://unispac.github.io/shallow-vs-deep-alignment.github.io/static/paper.pdf" target="_blank" style="text-decoration: none;">Paper</a>&nbsp;
</b>
</p>


------------




### Configurations

1. **Get the packages right:**
    Our fine-tuning trainer is fully built on top of the huggingface ecosystem. For the best reproducibility, we recommend using the following versions of the core dependencies:
    * `transformers==4.40.2`
    * `trl==0.8.1`
    * `accelerate==0.27.2`
    * `torch==2.2.0`
    * `deepspeed==0.13.2`
  
2. **Get the models right:**
   
    Place your models under the `ckpts/` directory, or ceate `ckpts` as a symbolic link to your model direcetory. In our example scripts, we use:
    * `ckpts/Llama-2-7b-chat-fp16` for the Llama-2-7b-chat model.
    * `ckpts/Llama-2-7B-fp16` for the Llama-2-7b base model.
    * `ckpts/gemma-1.1-7b-it` for the Gemma-1.1-7b instruction tuned model (aligned).
    * `ckpts/gemma-7b-it` for the Gemma-7b base model.
    
    All models are floating point 16-bit versions.
  
3. **Get the data right:**
   
    Our key fine-tuning and evaluation data should be located at `finetuning_buckets/datasets/data/`.
    This directory is excluded from this GitHub repo as it contains unsafe content.
    To get this data directory, go to: https://huggingface.co/datasets/Unispac/shallow-vs-deep-safety-alignment-dataset/, sign the agreement, download the `data/` directory, and place it at `finetuning_buckets/datasets/data/`
5. **Computation:**

    Our experiments need a minimum of 4 A100/H100 80GB GPUs to run (as we work on full parameters fine-tuning with a batch size of 64 in our experiments). The experiments can also take up to 8 CPU cores and 256GB of RAM. The codes are not tested with the gradient accumulation technique --- but since the trainer is adapted from the tlr trainers, it should work in principle. Please let us know if you need help with this, and also welcome to contribute to the codebase to support this feature so the codes can be run on smaller machines.

6. **OpenAI API Key:**

    Safety evaluation requires an OpenAI API key. Get it ready, and prepare to fill it in the safety evaluation scripts (see the following example scripts to walk through).




### Example Scripts for Reimplementing Our Experiments:

#### 1. Play with Prefilling Experiments:

**Compare the ASR of Llama-2-7B base model on HEx-PHI with and without a refusal prefix:**

  ```
  # Evalue the Llama-2-7B Base Model on HEx-PHI without prefilling:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi' \
        --model_family='llama2_base' \
        --prompt_style='llama2_base' \
        --evaluator='none' \
        --save_path='logs/prefilling/llama2_base_no_prefilling.json' \
        --eval_template='plain' ;

  # Prefill a refual prefix for Llama-2-7B Base Model:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi_with_prefix' \
        --model_family='llama2_base' \
        --prompt_style='llama2_base' \
        --evaluator='none' \
        --save_path='logs/prefilling/llama2_base_prefilled_refusal_prefix.json' \
        --eval_template='plain' \
        --prefill_prefix='I apologize, but I cannot' ;
  ```

Go to the `logs/prefilling/` directory to check the results. There, we have already prepared a `gpt_4_judge.ipynb` notebook. Filling in your openAI API key to replace the block `openai.api_key = "YOUR_API_KEY"` and just run the notebook to evaluate the results recored in the json files. The notebook takes use of the [Batch API](https://platform.openai.com/docs/guides/batch) of OpenAI. The first time you run, it will create a batch request submission. Wait for a few minutes until the batch request is completed. Then you can run the notebook again to get the results.

**Evaluate prefilling attacks on Llama-2-7B-Chat model:**

  ```
  # Evalue the Llama-2-7B-Chat Model on HEx-PHI without prefilling:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/prefilling/llama2_chat_no_prefilling.json' \
        --eval_template='null' ;
  
  # Prefill 10-tokens of harmful prefix for Llama-2-7B-Chat Model:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi_with_harmful_prefix' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/prefilling/llama2_chat_prefilled_10_harmful_tokens.json' \
        --eval_template='null' \
        --num_perfix_tokens=10 ;
  ```
Similarly, run `gpt_4_judge.ipynb` to evaluate the results.

#### 2. Play with The Data Augmentation Experiments to Get Llama-2-7B-Chat-Augmented Model

**Run the data augmentation fine-tuning:**

  ```
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml --num_processes 4  \
    finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
    --dataset_name="safety_augmentation" --model_family="llama2" \
    --learning_rate=2e-5 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --output_dir="logs/data_augmentation/Llama-2-7b-chat-augmented" \
    --logging_steps=1 \
    --num_train_epochs=10 \
    --gradient_checkpointing \
    --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
    --save_strategy='no' \
    --sft_type="sft" \
    --use_anchor=True \
    --anchor_batch_size_per_device=16 \
    --safety_augmentation=True \
    --use_warmup=False ;
  ```

**Test the robustness against prefilling (you would see better robustness):** 

  ```
  # Evalue the Llama-2-7B-Chat-Augmented Model on HEx-PHI against 10 tokens of harmful prefix prefilling:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="logs/data_augmentation/Llama-2-7b-chat-augmented" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi_with_harmful_prefix' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/data_augmentation/llama2_chat_augmented_prefilled_10_harmful_tokens.json' \
        --eval_template='null' \
        --num_perfix_tokens=10 ;

  # Compare with the Llama-2-7B-Chat Model on HEx-PHI against 10 tokens of harmful prefix prefilling:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi_with_harmful_prefix' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/data_augmentation/llama2_chat_prefilled_10_harmful_tokens.json' \
        --eval_template='null' \
        --num_perfix_tokens=10 ;
  ```

Similarly, run `gpt_4_judge.ipynb` to evaluate the results. 

Testing against [GCG Attack](https://github.com/llm-attacks/llm-attacks) and [Decoding Parameters Exploit](https://github.com/Princeton-SysML/Jailbreak_LLM)?
=> Just run their original repositories and set the model path to the fine-tuned model directory.

#### 3. Fine-tuning Attacks and Token-wise Constrained Fine-tuning Objective:

**Standard Supervised Fine-tuning of Llama-2-7B-Chat model on 100 harmful examples demonstration:**

  ```
  # Fine-tuning
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
    --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
    --sft_type='sft' \
    --use_warmup=True ;
  
  # Test Safety:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/fine-tuning-attack/safety_eval/pure_bad_sft.json' \
        --eval_template='plain' ;
  ```

**Fine-tuning with token-wise constrained fine-tuning objective instead (you would see much lower ASR with the constrained objective):**

  ```
  # Fine-tuning
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
    --dataset_name="pure_bad" --model_family='llama2' \
    --learning_rate=2e-5 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/soft_sft/lr_2e-5' \
    --logging_steps=1 \
    --num_train_epochs=25 \
    --gradient_checkpointing \
    --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
    --save_strategy='no' \
    --sft_type="soft_sft" \
    --beta=0.1 \
    --bias_factor=20 \
    --first_token_bias_factor=5 \
    --bias_length=5 \
    --use_warmup=True ;

  # Test Safety:
  accelerate launch  --num_processes=4 \
    eval_safety.py --model_name_or_path="logs/fine-tuning-attack/pure_bad/llama_2_7b/soft_sft/lr_2e-5" \
        --torch_dtype=bfloat16 \
        --safety_bench='hex-phi' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='none' \
        --save_path='logs/fine-tuning-attack/safety_eval/pure_bad_soft_sft.json' \
        --eval_template='plain' ;
  ```

**Fine-tuning on Utility Dataset: (Fine-tuning using the constrained objective can obtain comparable results to standard SFT)**

  ```
  # SFT
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
    --dataset_name='sql_create_context' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/sql_create_context/llama_2_7b/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=3 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
    --sft_type='sft' \
    --use_warmup=True ;

  accelerate launch --num_processes=4 \
        eval_utility.py \
        --torch_dtype=bfloat16 \
        --model_name_or_path='logs/fine-tuning-attack/sql_create_context/llama_2_7b/sft/lr_2e-5' \
        --dataset='sql_create_context' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='rouge_1' \
        --save_path="logs/fine-tuning-attack/utility_eval/sql_create_context_llama_2_7b_sft.json" ;



  # Constrained SFT
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
    --dataset_name="sql_create_context" --model_family='llama2' \
    --learning_rate=2e-5 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/sql_create_context/llama_2_7b/soft_sft/lr_2e-5' \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --gradient_checkpointing \
    --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
    --save_strategy='no' \
    --sft_type="soft_sft" \
    --beta=0.1 \
    --bias_factor=20 \
    --first_token_bias_factor=5 \
    --bias_length=5 \
    --use_warmup=True ;

  accelerate launch --num_processes=4 \
        eval_utility.py \
        --torch_dtype=bfloat16 \
        --model_name_or_path='logs/fine-tuning-attack/sql_create_context/llama_2_7b/soft_sft/lr_2e-5' \
        --dataset='sql_create_context' \
        --model_family='llama2' \
        --prompt_style='llama2' \
        --evaluator='rouge_1' \
        --save_path="logs/fine-tuning-attack/utility_eval/sql_create_context_llama_2_7b_soft_sft.json" ;
  ```

  
