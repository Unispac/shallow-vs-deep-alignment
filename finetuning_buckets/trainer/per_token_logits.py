# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal, Any

import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError


from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.integrations.tpu import tpu_spmd_dataloader

from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.import_utils import is_peft_available
from trl.trainer.utils import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    neftune_post_forward_hook,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
    disable_dropout_in_model,
    neftune_post_forward_hook,
)

from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_pt_utils import (
    AcceleratorConfig,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)


from transformers.debug_utils import DebugOption, DebugUnderflowOverflow


from trl.models import PreTrainedModelWrapper

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext
from accelerate.utils import is_deepspeed_available, tqdm
from copy import deepcopy
from collections import defaultdict

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

from packaging import version

import torch.distributed as dist

import importlib.metadata
import shutil
import time
import math
import sys
import os

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

if is_datasets_available():
    import datasets

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

if is_apex_available():
    from apex import amp

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
if is_deepspeed_available():
    import deepspeed

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class ConstrainedSFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        dataset_text_field (`Optional[str]`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a
            `ConstantLengthDataset` based on the `dataset_text_field` argument.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to `512`.
        infinite (`Optional[bool]`):
            Whether to use an infinite dataset or not. Defaults to `False`.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`. You can check how this is computed in the
            stack-llama example: https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset.
        dataset_num_proc (`Optional[int]`):
            The number of workers to use to tokenize the data. Only used when `packing=False`. Defaults to None.
        dataset_batch_size (`int`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Defaults to 1000.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This has been proven to drastically improve model performances for instruction
            fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when creating packed or non-packed datasets
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[PreTrainedModel] = None, # For simplicity, current implementation requires ref_model to be initialized
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        anchor_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        disable_dropout: bool = True,
        label_pad_token_id: int = -100,
        beta: float = 0.1,
        alpha: float = 0.1,
        bias_factor: float = 10,
        bias_length: int = 5,
        use_soft_sft: bool = True,
        use_anchor: bool = False,
        anchor_batch_size_per_device: int = 4,
        anchor_loss_type: str = "pin",
    ):
        
        self.use_soft_sft = use_soft_sft
        self.use_anchor = use_anchor
        self.anchor_batch_size_per_device = anchor_batch_size_per_device
        self.anchor_loss_type = anchor_loss_type

        if self.use_soft_sft or self.use_anchor:
            # if any of the two features are enabled, we need to have a ref_model
            if ref_model is None:
                raise ValueError(
                    "Trainer requires a reference model, since `use_soft_sft` or `use_anchor` is enabled."
                )
            if not isinstance(ref_model, PreTrainedModel):
                raise ValueError(f"The reference model should be a `PreTrainedModel` rather than a `{type(ref_model)}`.")
            
            self.ref_model = ref_model
        else:
            self.ref_model = None

        if self.use_anchor and (anchor_dataset is None):
            raise ValueError(
                "Trainer requires an anchor dataset, since `use_anchor` is enabled."
            )
        


        if packing:
            # can not use packing if we are also using ref model to regularize... 
            raise ValueError("Packing is not supported in this version of the SFTTrainer.")
        
        #if (data_collator is None) or (not isinstance(data_collator, DataCollatorForCompletionOnlyLM)):
        #    raise ValueError('Current implementation of Soft SFT Trainer requires DataCollatorForCompletionOnlyLM.')
        
        self.label_pad_token_id = label_pad_token_id
        assert self.label_pad_token_id == data_collator.ignore_index, "DataCollatorForCompletionOnlyLM should have ignore_index set to the label pad token id."

        self.beta = beta
        self.alpha = alpha
        self.bias_factor = bias_factor
        self.bias_length = bias_length
        self.max_seq_length = max_seq_length

        print('alpha:', self.alpha)
        print('beta:', self.beta)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        

        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SFTTrainer. But your model is already instantiated.")

        if infinite is not None:
            warnings.warn(
                "The `infinite` argument is deprecated and will be removed in a future version of TRL. Use `TrainingArguments.max_steps` or `TrainingArguments.num_train_epochs` instead to control training length."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )


        self._peft_has_been_casted_to_bf16 = False
        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    preprare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        preprare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                model = get_peft_model(model, peft_config)
                if args is not None and args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                    peft_module_casting_to_bf16(model)
                    # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                    self._peft_has_been_casted_to_bf16 = True

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `TrainingArguments`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if formatting_func is None and dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)

        if not packing:
            if dataset_text_field is None and formatting_func is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        

        
        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if dataset_kwargs is None:
                dataset_kwargs = {}
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **dataset_kwargs,
                )
            if anchor_dataset is not None:
                anchor_dataset = self._prepare_dataset(
                    anchor_dataset,
                    tokenizer,
                    packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **dataset_kwargs,
                )
                self.anchor_dataset = anchor_dataset
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}
                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        packing,
                        dataset_text_field,
                        max_seq_length,
                        formatting_func,
                        num_of_sequences,
                        chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]
        
        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self._precomputed_train_ref_log_probs = False
        self._precomputed_anchor_ref_log_probs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.args.max_steps > 0 and packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and packing:
            self.train_dataset.infinite = False
        

        if self.ref_model is not None:

            # Deepspeed Zero-3 does not support precompute_ref_log_probs
            if self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage == 3:
                raise ValueError(
                        "Deepspeed Zero-3 does not support precompute_ref_log_probs..."
                    )
        
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            reference_logps, reference_logps_avg, reference_logits_full, reference_logps_full = self.model_forward(self.ref_model, padded_batch)

        return reference_logps, reference_logps_avg, reference_logits_full, reference_logps_full


    def save_logits(self, save_path):

        if (self.ref_model is not None) and (not self._precomputed_train_ref_log_probs):

            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_logps = []
            reference_logps_avg = []
            reference_logps_full = []
            reference_logits_full_list = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):

                reference_logp, reference_logp_avg, reference_logits_full, reference_logp_full = self.compute_reference_log_probs(padded_batch)

                print('[gather]')
                reference_logp, reference_logp_avg, reference_logits_full, reference_logp_full = self.accelerator.gather_for_metrics(
                    (reference_logp, reference_logp_avg, reference_logits_full, reference_logp_full)
                )
                print('[done]')
                reference_logps.append(reference_logp.cpu())
                reference_logps_avg.append(reference_logp_avg.cpu())
                reference_logps_full.append(reference_logp_full.cpu())
                print('[to_cpu]')
                reference_logits_full_list.append(reference_logits_full.cpu())
                print('[done]')

            
            
            all_reference_logps = torch.cat(reference_logps).float().numpy()
            all_reference_logps_avg = torch.cat(reference_logps_avg).float().numpy()

            reference_logps_full_final = []
            for items in reference_logps_full:
                len_items = len(items)
                for item_id in range(len_items):
                    item = items[item_id].float().numpy()
                    #item = item[item!=0]
                    reference_logps_full_final.append(item)

            reference_logits_full_final = []
            for items in reference_logits_full_list:
                len_items = len(items)
                for item_id in range(len_items):
                    item = items[item_id].float().numpy()
                    #item = item[item!=0]
                    reference_logits_full_final.append(item)

            if PartialState().is_local_main_process:
                torch.save(reference_logits_full_final, save_path)
            
   
        return

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if (self.ref_model is not None) and (not self._precomputed_train_ref_log_probs):

            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_logps = []
            reference_logps_avg = []
            reference_logps_full = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):

                reference_logp, reference_logp_avg, reference_logp_full = self.compute_reference_log_probs(padded_batch)
                reference_logp, reference_logp_avg, reference_logp_full = self.accelerator.gather_for_metrics(
                    (reference_logp, reference_logp_avg, reference_logp_full)
                )
                reference_logps.append(reference_logp.cpu())
                reference_logps_avg.append(reference_logp_avg.cpu())
                reference_logps_full.append(reference_logp_full.cpu())

            all_reference_logps = torch.cat(reference_logps).float().numpy()
            all_reference_logps_avg = torch.cat(reference_logps_avg).float().numpy()

            reference_logps_full_final = []
            for items in reference_logps_full:
                len_items = len(items)
                for item_id in range(len_items):
                    item = items[item_id].float().numpy()
                    #item = item[item!=0]
                    reference_logps_full_final.append(item)

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=all_reference_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps_avg", column=all_reference_logps_avg
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps_full", column=reference_logps_full_final
            )
            self.label_names = ["reference_logps", "reference_logps_avg", "reference_logps_full"] # add `reference_logps` to label_names

            self._precomputed_train_ref_log_probs = True
   

        return super().get_train_dataloader()
       

    def get_anchor_dataloader(self) -> DataLoader:
        """
        Returns the anchor [`~torch.utils.data.DataLoader`].

        Will use no sampler if `anchor_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        anchoring if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.anchor_dataset is None:
            raise ValueError("Trainer: training requires an anchor_dataset.")
        
        data_collator = self.data_collator


        """
        ###############################################################
        # Compute reference log probabilities for the anchor dataset. #
        ###############################################################
        """

        if (self.ref_model is not None) and (not self._precomputed_anchor_ref_log_probs):

            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.anchor_dataset, **dataloader_params))

            reference_logps = []
            reference_logps_avg = []

            for padded_batch in tqdm(iterable=data_loader, desc="Anchor dataset reference log probs"):

                reference_logp, reference_logp_avg, _ = self.compute_reference_log_probs(padded_batch)
                reference_logp, reference_logp_avg = self.accelerator.gather_for_metrics(
                    (reference_logp, reference_logp_avg)
                )
                reference_logps.append(reference_logp.cpu())
                reference_logps_avg.append(reference_logp_avg.cpu())

            all_reference_logps = torch.cat(reference_logps).float().numpy()
            all_reference_logps_avg = torch.cat(reference_logps_avg).float().numpy()

            self.anchor_dataset = self.anchor_dataset.add_column(
                name="reference_logps", column=all_reference_logps
            )
            self.anchor_dataset = self.anchor_dataset.add_column(
                name="reference_logps_avg", column=all_reference_logps_avg
            )
            self.label_names = ["reference_logps", "reference_logps_avg"] # add `reference_logps` to label_names
            self._precomputed_anchor_ref_log_probs = True

        if is_datasets_available() and isinstance(self.anchor_dataset, datasets.Dataset):
            self.anchor_dataset = self._remove_unused_columns(self.anchor_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.anchor_batch_size_per_device, # batch per device
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.anchor_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_anchor_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        return self.accelerator.prepare(DataLoader(self.anchor_dataset, **dataloader_params))


    def _get_anchor_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.anchor_dataset is None or not has_length(self.anchor_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.anchor_dataset, datasets.Dataset):
                lengths = (
                    self.anchor_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.anchor_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.anchor_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.anchor_dataset)

    def model_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        

        model_kwargs = {}
        all_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps, all_logps_avg, full_logps, full_logits = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        return (all_logps, all_logps_avg, full_logits, full_logps)




    
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # shift one position for auto-regressive models
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        avg_logps = (per_token_logps * loss_mask).sum(-1) / (loss_mask.sum(-1) + 1e-8)
        sum_logps = (per_token_logps * loss_mask).sum(-1)
        
        

        full_logps = per_token_logps * loss_mask
        full_logps += (~loss_mask) * 1000
        max_seq_length = self.max_seq_length
        if full_logps.shape[1] > max_seq_length:
            full_logps = full_logps[:, :max_seq_length]
            #full_logits = full_logits[:, :max_seq_length, :]
        else:
            full_logps = torch.nn.functional.pad(full_logps, (0, max_seq_length - full_logps.shape[1]), value=1000)
            #full_logits = torch.nn.functional.pad(full_logits, (0, 0, 0, max_seq_length - full_logits.shape[1]), value=10000)

        full_logits = torch.zeros( (logits.shape[0], 20, logits.shape[2]) ).to(logits.device)
        for i in range(logits.shape[0]):
            item = logits[i]
            item = item[loss_mask[i] == 1]
            L = min(20, item.shape[0])
            full_logits[i, :L, :] = item[:L]
            full_logits[i, L:, :] = 10086

        return sum_logps, avg_logps, full_logps, full_logits


    def get_beta_list(self, length):
        
        beta = self.beta
        #prefix = torch.FloatTensor([beta * 25, beta * 25, beta * 25, beta * 25,  beta * 25])
        len_prefix = self.bias_length
        prefix = torch.FloatTensor([beta * self.bias_factor] * len_prefix)
        additive_bias = torch.FloatTensor( [1.0] * len_prefix )
        if length <= len_prefix:
            beta_list = prefix[:length]
            additive_bias_list = additive_bias[:length]
        else:
            beta_list = torch.full((length,), beta)
            beta_list[:len_prefix] = prefix
            beta_list[len_prefix:] = beta
            additive_bias_list = torch.full((length,), 0)
            additive_bias_list[:len_prefix] = additive_bias
            additive_bias_list[len_prefix:] = 0
        
        return beta_list, additive_bias_list

    def soft_sft_loss(
        self,
        policy_logps: torch.FloatTensor,
        reference_logps: torch.FloatTensor,
        policy_logps_full,
        reference_logps_full,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the Soft SFT loss for a batch of policy and reference model log probabilities.

        Args:
            policy_logps: Log probabilities of the policy model for the SFT data. Shape: (batch_size,)
            reference_logps: Log probabilities of the reference model for the SFT data. Shape: (batch_size,)

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the Soft SFT loss for each example in the batch.
            The rewards tensors contain the difference in log probabilities between the policy and reference model for each example in the batch.
        """
        num = policy_logps_full.shape[0]
        losses = torch.zeros(num).to(policy_logps.device)
        for i in range(num):
            policy_item = policy_logps_full[i]
            policy_item = policy_item[policy_item <= 0]
            reference_item = reference_logps_full[i]
            reference_item = reference_item[reference_item <= 0]
            beta, additive_bias = self.get_beta_list(len(policy_item))
            beta = beta.to(policy_logps.device)
            additive_bias = additive_bias.to(policy_logps.device)
            losses_list = -F.logsigmoid(beta * (policy_item - reference_item) + additive_bias) / beta
            losses[i] = 5 * losses_list.mean()


        return losses





    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        batch_anchor: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        policy_logps, policy_logps_avg, policy_logits, policy_logps_full = self.model_forward(model, batch)
        metrics[f"{prefix}logps/policy"] = policy_logps.detach().mean().cpu()

        if self.use_soft_sft:
            reference_logps = batch["reference_logps"]
            reference_logps_avg = batch['reference_logps_avg']
            reference_logps_full = batch['reference_logps_full']

            soft_sft_losses = self.soft_sft_loss(
                policy_logps,
                reference_logps,
                policy_logps_full,
                reference_logps_full
            )
            losses = soft_sft_losses

            metrics[f"{prefix}logps/reference"] = reference_logps.mean().cpu()
            metrics[f"{prefix}sft_loss/reference"] = -reference_logps_avg.mean().cpu()
            metrics[f"{prefix}sft_loss/policy"] = -policy_logps_avg.mean().cpu()
        else:
            losses = -policy_logps_avg

        if self.use_anchor:
            # if anchor dataset is provided, adding anchor batch
            anchor_logps, anchor_logps_avg, anchor_logits, _ = self.model_forward(model, batch_anchor)

            if self.anchor_loss_type == "pin" or (self.use_soft_sft and self.anchor_loss_type == "max"):
                reference_anchor_logps = batch_anchor["reference_logps"]
                reference_anchor_logps_avg = batch_anchor['reference_logps_avg']
                metrics[f"{prefix}logps/anchor_reference"] = reference_anchor_logps.mean().cpu()

            if self.anchor_loss_type == "pin":
                anchor_penalty = torch.abs(anchor_logps - reference_anchor_logps)
                anchor_penalty = anchor_penalty * (anchor_logps < reference_anchor_logps).float()
                losses = torch.cat([losses, self.alpha * anchor_penalty])
                metrics[f"{prefix}anchor_penalty"] = anchor_penalty.mean().cpu()
            
            elif self.anchor_loss_type == "max":
                
                if self.use_soft_sft:
                    anchor_soft_sft_losses, anchor_rewards = self.soft_sft_loss(
                        anchor_logps,
                        reference_anchor_logps
                    )
                    losses = torch.cat([losses, self.alpha * anchor_soft_sft_losses])
                    metrics[f"{prefix}anchor_rewards"] = anchor_rewards.detach().mean().cpu()
                else:
                    losses = torch.cat([losses, -self.alpha * anchor_logps_avg])

            else:
                raise ValueError(f"Invalid anchor_loss_type: {self.anchor_loss_type}")
                
            metrics[f"{prefix}logps/anchor"] = anchor_logps.detach().mean().cpu()
            

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        anchor_inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        #temp = inputs['input_ids'][2]
        #print('input_ids:', self.tokenizer.decode(temp, skip_special_tokens=False))

        #temp = inputs['labels'][2]
        #temp = temp[temp != -100]
        #print('labels:', self.tokenizer.decode(temp, skip_special_tokens=False))
        
        
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, anchor_inputs, train_eval="train")
        
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        
        return loss


    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # check if torch dataset / dataloader and do nothing
        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model



    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps

        train_dataloader = self.get_train_dataloader()
        if self.use_anchor:
            anchor_dataloader = self.get_anchor_dataloader()
        if self.ref_model is not None:
            del self.ref_model
            self.ref_model = None

        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
            if self.use_anchor:
                anchor_dataloader = tpu_spmd_dataloader(anchor_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
            

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.use_anchor:
            self.callback_handler.anchor_dataloader = anchor_dataloader

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1

            if self.use_anchor:
                anchor_data_iterator = iter(anchor_dataloader)

            for step, inputs in enumerate(epoch_iterator):
                
                if self.use_anchor:
                    try:
                        anchor_inputs = next(anchor_data_iterator)
                    except:
                        anchor_data_iterator = iter(anchor_dataloader)
                        anchor_inputs = next(anchor_data_iterator)


                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    if self.use_anchor:
                        tr_loss_step = self.training_step(model, inputs, anchor_inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs, None)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                        else:
                            grad_norm = _grad_norm.item() if _grad_norm is not None else None

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_tpu_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], anchor_inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            
            anchor_inputs (`Dict[str, Union[torch.Tensor, Any]]`): the anchor inputs for control

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.use_anchor:
            anchor_inputs = self._prepare_inputs(anchor_inputs)

        """
        Continue from here !!!!
        """

        if is_sagemaker_mp_enabled():
            raise NotImplementedError("SageMaker MP is not yet supported for training_step")
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps).reduce_mean().detach().to(self.args.device)
            
            if self.use_anchor:
                loss_anchor_mb = smp_forward_backward(model, anchor_inputs, self.args.gradient_accumulation_steps).reduce_mean().detach().to(self.args.device)
                
                if self.anchor_loss_type == "pin":
                    raise NotImplementedError("PIN loss is not yet implemented for SageMaker MP")
                else:
                    return torch.cat([loss_mb, loss_anchor_mb])
            else:
                return loss_mb

        with self.compute_loss_context_manager():
            if self.use_anchor:
                loss = self.compute_loss(model, inputs, anchor_inputs)
            else:
                loss = self.compute_loss(model, inputs, None)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if optimizer_cls.__name__ == "SGD":
                optimizer_kwargs["momentum"] = self.args.momentum
            elif optimizer_cls.__name__ == 'RMSprop':
                optimizer_kwargs["eps"] = self.args.rmsprop_eps
                optimizer_kwargs["alpha"] = self.args.rmsprop_alpha
            
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        print(self.optimizer)
        return self.optimizer


    """
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        
        if self.lr_scheduler is None:
            
            from functools import partial
            from torch.optim.lr_scheduler import LambdaLR

            def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
                if current_step < num_warmup_steps:
                    return 0.1
                return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

            lr_lambda = partial(
                _get_linear_schedule_with_warmup_lr_lambda,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)

            self._created_lr_scheduler = True

        return self.lr_scheduler
    """