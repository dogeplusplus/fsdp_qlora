"""
Read our announcement blog post: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html.

This script trains a model using FSDP with LoRA & QLoRA. It pulls inspiration from
- llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
- PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`

You should treat this script as an alpha/preview release. If you're not comfortable with testing and debugging
models, we'd suggest holding off for a few months while the community more fully tests the approach.
"""

# Imports

# General
import functools
import os
import time
import types
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import safetensors
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import init_empty_weights
from accelerate.utils import set_seed

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from fastcore.parallel import parallel
from omegaconf import DictConfig

# Argument parsing
from packaging.version import parse
from safetensors.torch import save_file

# Torch + distributed training
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

# FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub

from profiling_utils import profiling_context
from scripts.dataset import get_dataloader
from scripts.dora import BNBDORA, HQQDORA, DORALayer, MagnitudeLayer
from scripts.lora import LORA
from scripts.optimizer import get_lr_scheduler, get_optimizer

try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
except ImportError:
    HQQLinear = None
    pass

# To add a new model, import the transformer, attention, & MLP layers
# for the wrapping policy and `check_fn` in activation checkpointing
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaMLP,
)
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralDecoderLayer,
    MistralMLP,
)

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass


class Logger:
    def __init__(
        self,
        args,
        log_to="stdout",
        project_name="fsdp_qlora",
        entity=None,
        group=None,
        name=None,
        rank=0,
    ):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank == 0:
            import wandb

            wandb.init(
                project=project_name, entity=entity, group=group, name=name, config=args
            )

    def log(self, d: Dict, rank: int):
        if rank != 0:
            return
        if self.log_to == "tqdm":
            for k, v in d.items():
                tqdm.write(f"{k}: {v}")
        elif self.log_to == "wandb":
            wandb.log(d)
        elif self.log_to == "stdout":
            for k, v in d.items():
                print(f"{k}: {v}")

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank == 0:
            wandb.finish()


def update_progress_bar(
    progress_bar: tqdm, epoch: int, log_loss: float, log_lr: float, rank: int
):
    """Updates the progress bar with the current epoch, loss, and learning rate"""
    if rank == 0:
        if log_lr >= 0:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True
            )
        else:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True
            )


def n_loading_workers(quant_method: str, param_count: float):
    devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
    left = int(os.cpu_count() / torch.cuda.device_count())
    right = int(
        (4 if quant_method == "hqq" else 8)
        * (devprops.total_memory / 1e9 / 40)
        * (70 / (param_count / 1e9))
    )
    return min(left, right)


# Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
# and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
def load_and_quantize_parallel(name_param, model, **kwargs):
    name, param = name_param
    load_and_quantize(model, name, param, **kwargs)


# Utilities related to model loading
def replace_linear(
    model: nn.Module,
    linear_replacement: nn.Module,
    quant_config: dict | None = None,
    skip_modules: List[str] = ["lm_head"],
    **kwargs,
):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if len(list(module.children())) > 0:
            replace_linear(
                module, linear_replacement, quant_config, skip_modules, **kwargs
            )

        if isinstance(module, torch.nn.Linear):
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **kwargs,
                )
            elif issubclass(linear_replacement, HQQLinear):
                model._modules[name] = linear_replacement(
                    module, quant_config, **kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported linear replacement: {type(linear_replacement)}"
                )
    return model


def setup_quantized_meta_for_peft(model: nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None


def load_and_quantize(
    module: nn.Module,
    name: str,
    value: Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
    skip_names: list[str] = [],
    to_cpu: bool = False,
    to_meta: bool = False,
    verbose: bool = False,
    quant_method: str = "bnb",
    is_dora: bool = False,
):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """

    def place_on_device(value):
        if to_meta:
            device = "meta"
        elif to_cpu:
            device = "cpu"
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition(".")
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if quant_method == "bnb":
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                if is_dora:
                    setattr(
                        submodule,
                        "dora_scale",
                        value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"),
                    )
                value = type(param)(
                    value.to(device=device, dtype=dtype).data, **param.__dict__
                ).cuda(device)
                if to_meta:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)
        elif quant_method == "hqq":
            if isinstance(submodule, HQQLinear):
                if value_key == "weight":
                    # Like `Params4bit`, this workaround quantizes `HQQLinear`` per device so the quantization
                    # meta dictionary is created on all ranks, before converting to meta on non-rank 0.
                    submodule.linear_layer.to_empty(device=device)
                    submodule.linear_layer.weight.data.copy_(
                        value.to(device=device, dtype=dtype)
                    )
                    if is_dora:
                        setattr(
                            submodule,
                            "dora_scale",
                            value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"),
                        )
                    submodule.initialize()

                    if to_meta:
                        setattr(
                            submodule, "W_q", nn.Parameter(submodule.W_q.to("meta"))
                        )
                    elif to_cpu:
                        setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("cpu")))
                    submodule.in_gpu = False

                if value_key == "bias":
                    raise ValueError("Bias not supported in HQQLinear yet!")
            else:
                param = submodule.get_parameter(value_key)
                value = type(param)(place_on_device(value).data)

    except AttributeError:
        # it's a buffer
        value = place_on_device(value)
        pass
    if HQQLinear is None or not isinstance(submodule, HQQLinear):
        setattr(submodule, value_key, value)


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy: bool = False, vanilla_policy: bool = False):
    if custom_policy:

        def lambda_policy_fn(module):
            # LoRA and DoRA trainable layers.
            return (
                isinstance(module, nn.Sequential)
                and all(m.weight.requires_grad for m in module)
            ) or (isinstance(module, (DORALayer, MagnitudeLayer)))
    else:

        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(
            module,
            tuple(
                (*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values())
            ),
        )

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP))

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    self_attn_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn
    )
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
    )
    if vanilla_policy:
        return transformer_wrap_policy

    policies = [lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


def determine_dtypes(
    precision: str,
) -> Tuple[torch.dtype, torch.dtype, MixedPrecision, List[str]]:
    mp_policy = None
    load_param_skip_names = []
    if precision == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif precision == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif precision == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    elif precision == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    elif precision == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
        load_param_skip_names = ["inv_freq"]
    else:
        raise ValueError("Invalid precision")

    return torch_dtype, compute_dtype, mp_policy, load_param_skip_names


def prepare_quantized_model(
    model_name: str,
    attn_impl: str,
    train_type: str,
    llama_pro_path: str,
    n_bits: int,
    precision: str,
    torch_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    low_memory: bool,
    rank: int,
    local_rank: int,
    loading_workers: int,
    verbose: bool,
    load_param_skip_names: List[str],
) -> Tuple[nn.Module, list]:
    new_layer_names = []

    if train_type in ["full", "lora", "custom_lora"]:
        if (low_memory and rank == 0) or (not low_memory):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_cache=False,
                torch_dtype=torch_dtype,
                _attn_implementation=attn_impl,
            )
            dtype = torch_dtype if precision == "bf16" else None
            model.to(dtype=dtype, device="cpu" if low_memory else rank)
        else:
            cfg = AutoConfig.from_pretrained(model_name)
            cfg.use_cache = False
            cfg._attn_implementation = attn_impl
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)
            if precision == "bf16":
                model.to(torch_dtype)
    elif train_type in [
        "qlora",
        "custom_qlora",
        "hqq_lora",
        "hqq_dora",
        "bnb_dora",
        "bnb_llama_pro",
        "hqq_llama_pro",
    ]:  # Our custom loading
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.use_cache = False
        cfg._attn_implementation = attn_impl
        skip_modules = ["lm_head"]

        if train_type in ["bnb_llama_pro", "hqq_llama_pro"]:
            llama_pro_path = Path(llama_pro_path)
            num_original_layers, num_expanded_layers = llama_pro_path.name.split(
                "blk_exp-"
            )[1].split("-")
            num_original_layers, num_expanded_layers = (
                int(num_original_layers),
                int(num_expanded_layers),
            )
            total_new_layers = num_expanded_layers - num_original_layers
            split = int(
                num_original_layers / (num_expanded_layers - num_original_layers)
            )
            new_layer_ids = [split + (split + 1) * n for n in range(total_new_layers)]
            new_layer_names = [f"layers.{i}" for i in new_layer_ids]
            skip_modules += [str(lid) for lid in new_layer_ids]
            cfg.num_hidden_layers = num_expanded_layers

        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            if train_type in ["hqq_lora", "hqq_dora", "hqq_llama_pro"]:
                # TODO: Tune BaseQuantizeConfig.
                quant_config = BaseQuantizeConfig(
                    nbits=int(n_bits),
                    group_size=64,
                    quant_zero=True,
                    quant_scale=True,
                    offload_meta=True,
                    view_as_float=True,
                )
                model.model = replace_linear(
                    model.model,
                    HQQLinear,
                    quant_config,
                    device=rank,
                    compute_dtype=compute_dtype,
                    del_orig=True,
                    initialize=False,
                    skip_modules=skip_modules,
                )
                HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
            else:
                model.model = replace_linear(
                    model.model,
                    Linear4bit,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                    quant_storage=torch_dtype,
                    skip_modules=skip_modules,
                )
        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        if train_type in ["bnb_llama_pro", "hqq_llama_pro"]:
            files = glob(str(llama_pro_path / "*.safetensors"))
        else:
            try:
                idx = hub.cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME)
                files, _ = hub.get_checkpoint_shard_files(model_name, idx)
            except OSError:
                try:
                    # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                    files = []
                    files.append(hub.cached_file(model_name, SAFE_WEIGHTS_NAME))
                except OSError as e:
                    # This means the model probably doesn't have a safetensors file
                    raise e

        quant_method = (
            "hqq" if train_type in ["hqq_lora", "hqq_dora", "hqq_llama_pro"] else "bnb"
        )
        param_count = sum((p.numel() for n, p in model.named_parameters()))
        if rank == 0 or verbose:
            print("Loading model", rank)
        if rank == 0 and verbose:
            print(f"Total model params: {param_count}")

        n_workers = (
            n_loading_workers(quant_method, param_count)
            if loading_workers == -1
            else loading_workers
        )
        if rank == 0 and verbose:
            print(f"Using n_workers: {n_workers} for loading")

        start = time.time()
        for filename in tqdm(
            files,
            desc="Loading & Quantizing Model Shards",
            disable=rank != 0,
            position=0,
        ):
            weights = safetensors.torch.load_file(filename)
            parallel(
                load_and_quantize_parallel,
                iter(weights.items()),
                n_workers=n_workers,
                threadpool=True,
                model=model,
                dtype=torch_dtype,
                device=local_rank,
                skip_names=load_param_skip_names,
                to_cpu=(low_memory and rank == 0),
                to_meta=(low_memory and rank != 0),
                verbose=verbose,
                quant_method=quant_method,
                is_dora=(train_type in ["hqq_dora", "bnb_dora"]),
            )

        if rank == 0 and verbose:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")
        # cleanup any extra memory usage from parallel loading
        torch.cuda.empty_cache()

    return model, new_layer_names


def prepare_peft_model(
    model,
    train_type,
    lora_rank,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    rank,
    local_rank,
    low_memory,
    verbose,
    new_layer_names,
):
    # PEFT setup (LoRA and QLoRA)
    if train_type in ["lora", "qlora"]:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank != 0 and low_memory:
            setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, peft_config)

        if rank == 0:
            model.print_trainable_parameters()
        elif low_memory:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    elif train_type in [
        "custom_qlora",
        "custom_lora",
        "hqq_lora",
        "hqq_dora",
        "bnb_dora",
    ]:
        if train_type == "hqq_dora":
            print("Using HQQDORA", rank)
            lora_cls = HQQDORA
        elif train_type == "bnb_dora":
            print("Using BNB DORA", rank)
            lora_cls = BNBDORA
        else:
            print("Using LORA", rank)
            lora_cls = LORA

        # Create LORA layers.
        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition(".")
            if value_key in lora_target_modules:
                m = model.get_submodule(name)
                qlora_layer = lora_cls(m, lora_rank, lora_alpha, lora_dropout)
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        for n, p in model.named_parameters():
            if any(
                [
                    lora_name in n
                    for lora_name in ["lora_AB", "lora_A", "lora_B", "magnitude"]
                ]
            ):
                p.requires_grad = True
                if verbose:
                    print("Trainable LORA layer", n)
            else:
                p.requires_grad = False
        if rank == 0 or verbose:
            print(
                f"Rank {rank}: LoRA layers added: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB"
            )

    elif train_type in ["bnb_llama_pro", "hqq_llama_pro"]:
        for n, p in model.named_parameters():
            if any([layer_name in n for layer_name in new_layer_names]):
                p.requires_grad = True
                if verbose:
                    print("Trainable Llama-Pro layer", n)
            else:
                p.requires_grad = False

    return model


def prepare_fsdp_model(
    model: nn.Module,
    sharding_strategy: ShardingStrategy,
    train_type: str,
    use_cpu_offload: bool,
    mp_policy: MixedPrecision,
    low_memory: bool,
    rank: int,
) -> nn.Module:
    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(
        custom_policy=train_type
        in ["custom_qlora", "hqq_lora", "hqq_dora", "bnb_dora"],
        vanilla_policy=train_type in ["full", "bnb_llama_pro", "hqq_llama_pro"],
    )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if use_cpu_offload else None,
        limit_all_gathers=True,  # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=low_memory,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
        if (rank != 0 and low_memory)
        else None,  # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    return model


# Main function, run on each process
def fsdp_main(
    local_rank: int,
    world_size: int,
    data_args: DictConfig,
    train_args: DictConfig,
    profile_args: DictConfig,
    opt_args: DictConfig,
    model_args: DictConfig,
):
    # Setup and initialize the process group
    os.environ["MASTER_ADDR"] = train_args["master_addr"]
    os.environ["MASTER_PORT"] = train_args["master_port"]
    if "SLURM_PROCID" in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ["SLURM_PROCID"]) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    if model_args["use_cpu_offload"]:
        torch.set_num_threads(
            os.cpu_count() // (min(world_size, torch.cuda.device_count()))
        )

    # Start logging
    logger = Logger(
        train_args,
        log_to=train_args["log_to"],
        project_name=train_args["project_name"],
        entity=train_args["entity"],
        group=train_args["group"],
        name=train_args["name"],
        rank=rank,
    )

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype.
    # The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses
    # `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16
    # trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO check if it exists first

    # Set up dataloader
    dataloader = get_dataloader(
        tokenizer,
        data_args["dataset"],
        data_args["dataset_samples"],
        data_args["context_length"],
        data_args["seed"],
        data_args["batch_size"],
        data_args["gradient_accumulation_steps"],
    )

    # Determine dtypes
    torch_dtype, compute_dtype, mp_policy, load_param_skip_names = determine_dtypes(
        model_args["precision"]
    )

    # Create model
    cfg = None
    attn_impl = "sdpa"  # torch 2.2 sdpa uses flash attn 2
    if rank == 0 or train_args["verbose"]:
        print("Creating model", rank)

    model, new_layer_names = prepare_quantized_model(
        model_args["model_name"],
        attn_impl,
        model_args["train_type"],
        model_args["llama_pro_path"],
        model_args["n_bits"],
        model_args["precision"],
        torch_dtype,
        compute_dtype,
        model_args["low_memory"],
        rank,
        local_rank,
        train_args["loading_workers"],
        train_args["verbose"],
        load_param_skip_names,
    )

    model = prepare_peft_model(
        model,
        model_args["train_type"],
        model_args["lora_rank"],
        model_args["lora_alpha"],
        model_args["lora_dropout"],
        model_args["lora_target_modules"],
        rank,
        local_rank,
        model_args["low_memory"],
        train_args["verbose"],
        new_layer_names,
    )

    if rank == 0 or train_args["verbose"]:
        print(
            f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB"
        )

    if train_args["log_to"] == "wandb":
        logger.log(
            {
                "memory/allocated_after_model_created": torch.cuda.memory_allocated(
                    local_rank
                )
            },
            rank,
        )
        logger.log(
            {
                "memory/reserved_after_model_creation": torch.cuda.memory_reserved(
                    local_rank
                )
            },
            rank,
        )

    if rank == 0 or train_args["verbose"]:
        print("Wrapping model w/ FSDP", rank)

    # Wrap model with FSDP
    model = prepare_fsdp_model(
        model,
        train_args["sharding_strategy"],
        model_args["train_type"],
        model_args["use_cpu_offload"],
        mp_policy,
        model_args["low_memory"],
        rank,
        local_rank,
        train_args["verbose"],
        logger,
    )

    if rank == 0 or train_args["verbose"]:
        print(
            f"Rank {rank}: Wrapped model: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB"
        )
    if train_args["log_to"] == "wandb":
        logger.log(
            {
                "memory/allocated_after_model_wrap": torch.cuda.memory_allocated(
                    local_rank
                )
            },
            rank,
        )
        logger.log(
            {
                "memory/reserved_after_model_wrap": torch.cuda.memory_reserved(
                    local_rank
                )
            },
            rank,
        )

    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if model_args["use_gradient_checkpointing"]:
        if model_args["reentrant_checkpointing"]:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT
            if model_args["reentrant_checkpointing"]
            else CheckpointImpl.NO_REENTRANT,
        )

        def check_fn(submodule):
            return isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer))

        if rank == 0 or train_args["verbose"]:
            print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if model_args["use_activation_cpu_offload"]:
        if rank == 0 or train_args["verbose"]:
            print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and train_args["verbose"]:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")

    # Create the optimizer
    optimizer = get_optimizer(
        model, opt_args["optimizer"], opt_args["lr"], opt_args["wd"]
    )

    # LR scheduler.
    gradient_accumulation_steps = max(1, data_args["gradient_accumulation_steps"])
    lr_scheduler, num_training_steps = get_lr_scheduler(
        optimizer,
        dataloader,
        gradient_accumulation_steps,
        opt_args["lr_scheduler"],
        train_args["num_epochs"],
    )

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and train_args["verbose"]:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group["params"]:
                print(
                    f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}"
                )

    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if model_args["precision"] in [
        "fp16_autocast",
        "bf16_autocast",
        "bf16_buffers_autocast",
    ]:
        autocast = torch.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if model_args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None

    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    memory_stats = []
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1

    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)
    with profiling_context(profile_args, rank=rank) as prof:
        for epoch in range(train_args["num_epochs"]):
            update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
            model.train()
            ddp_loss = torch.zeros(2).to(local_rank)

            for batch_idx, batch in enumerate(dataloader):
                accumulate_grads = (batch_idx + 1) % gradient_accumulation_steps == 0

                # Prevent gradient syncing until update step if using no_sync option.
                # Documentation states this should only be used on the root FSDP instance
                # We assume this is a one-node setup
                if train_args["no_sync"] and not accumulate_grads:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                # Start logging memory (first iter) if requested
                if (
                    train_args["profile_memory"]
                    and batch_idx == 0
                    and rank == 0
                    and epoch == 0
                ):
                    torch.cuda.memory._record_memory_history()

                # Log memory usage
                if (
                    batch_idx == 0
                    and epoch == 0
                    and (rank == 0 or train_args["verbose"])
                ):
                    reserved_before_forward = torch.cuda.memory_reserved(local_rank)
                    memory_stats.append(
                        f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB"
                    )
                    if train_args["log_to"] == "wandb":
                        logger.log(
                            {
                                "memory/allocated_before_forward": torch.cuda.memory_allocated(
                                    local_rank
                                )
                            },
                            rank,
                        )
                        logger.log(
                            {"memory/reserved_before_forward": reserved_before_forward},
                            rank,
                        )

                # Forward pass
                with sync_context:
                    with autocast:
                        output = model(
                            batch["input_ids"].to(local_rank),
                            labels=batch["labels"].to(local_rank),
                            attention_mask=None,
                        )
                        loss = output.loss

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Log memory usage
                    if (
                        batch_idx == 0
                        and epoch == 0
                        and (rank == 0 or train_args["verbose"])
                    ):
                        reserved_after_forward = torch.cuda.memory_reserved(local_rank)
                        memory_stats.append(
                            f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB"
                        )
                        if train_args["log_to"] == "wandb":
                            logger.log(
                                {
                                    "memory/allocated_after_forward": torch.cuda.memory_allocated(
                                        local_rank
                                    )
                                },
                                rank,
                            )
                            logger.log(
                                {
                                    "memory/reserved_after_forward": reserved_after_forward
                                },
                                rank,
                            )

                    # Backward pass
                    if scale_grads:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Record loss
                bs = batch["input_ids"].shape[0]
                ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
                ddp_loss[1] += bs

                # Step the optimizer (w/ gradient accumulation)
                if accumulate_grads:
                    if train_args["apply_gradient_clipping"] and (
                        opt_args["grad_norm"] is not None
                    ):
                        model.clip_grad_norm_(opt_args["grad_norm"], norm_type=2.0)
                    if scale_grads:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    # avoid overhead when lr is constant.
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    progress_bar.update(1)

                # Log memory usage after backward
                if (
                    batch_idx == 0
                    and epoch == 0
                    and (rank == 0 or train_args["verbose"])
                ):
                    reserved_after_backward = torch.cuda.memory_reserved(local_rank)
                    memory_stats.append(
                        f"Rank {rank}: After backward: {reserved_after_backward/2**30:.2f} GiB"
                    )
                    if train_args["log_to"] == "wandb":
                        logger.log(
                            {
                                "memory/allocated_after_backward": torch.cuda.memory_allocated(
                                    local_rank
                                )
                            },
                            rank,
                        )
                        logger.log(
                            {"memory/reserved_after_backward": reserved_after_backward},
                            rank,
                        )

                # Delete the output so more memory frees up before the next forward pass
                output = None
                loss = None

                # Stop logging memory (first iter)
                if (
                    train_args["profile_memory"]
                    and batch_idx == 0
                    and rank == 0
                    and epoch == 0
                ):
                    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                    torch.cuda.memory._record_memory_history(
                        enabled=None
                    )  # Stop recording

                # Log loss every gradient update steps
                if accumulate_grads:
                    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        log_loss = ddp_loss[0] / ddp_loss[1]
                        if lr_scheduler is not None:
                            log_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            log_lr = opt_args["lr"]
                        update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                        if train_args["log_to"] == "wandb":
                            logger.log({"loss": log_loss, "lr": log_lr}, rank)
                    ddp_loss = torch.zeros(2).to(local_rank)

                if rank == 0 and train_args["verbose"]:
                    print(f"Batch idx {batch_idx}")

                prof.step()

                # Primarily for debugging
                if train_args["max_steps"] > 0 and batch_idx > train_args["max_steps"]:
                    if rank == 0:
                        print("Max steps reached, skipping rest of epoch")
                    break

            # Print + log peak memory usage for the whole fourth step of training
            if epoch == 0 and (rank == 0 or train_args["verbose"]):
                peak_allocated_memory = torch.cuda.max_memory_allocated(local_rank)
                peak_reserved_memory = torch.cuda.max_memory_reserved(local_rank)
                memory_stats.append(
                    f"Rank {rank}: Peak allocated memory: {peak_allocated_memory/2**30:.2f} GiB"
                )
                memory_stats.append(
                    f"Rank {rank}: Peak reserved memory:  {peak_reserved_memory/2**30:.2f} GiB"
                )
                if train_args["log_to"] == "wandb":
                    logger.log({"memory/allocated_peak": peak_allocated_memory}, rank)
                    logger.log({"memory/reserved_peak": peak_reserved_memory}, rank)

    # Synchronize at the end and record time
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time, model, & memory stats
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)
    for line in memory_stats:
        print(line)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if train_args["save_model"]:
        if rank == 0:
            os.makedirs(train_args["output_dir"], exist_ok=True)
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if train_args["train_type"] in [
            "custom_lora",
            "custom_qlora",
            "hqq_lora",
            "hqq_dora",
            "bnb_dora",
            "bnb_llama_pro",
            "hqq_llama_pro",
        ]:
            cpu_state_dict = {}
            if model_args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
                trainable_fsdp_modules = [
                    (n, m)
                    for n, m in model.named_modules()
                    if n.endswith(tuple(new_layer_names))
                ]
            else:
                trainable_fsdp_modules = [
                    (n, m)
                    for n, m in model.named_modules()
                    if n.endswith(("lora_AB", "dora_layer", "magnitude_layer"))
                ]
            for prefix, module in trainable_fsdp_modules:
                prefix = (
                    prefix.replace("_fsdp_wrapped_module.", "")
                    .replace("_checkpoint_wrapped_module.", "")
                    .replace("_offload_wrapped_module.", "")
                )
                if train_args["verbose"]:
                    print(f"Saving {prefix}")
                with FSDP.state_dict_type(
                    module, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state_dict = {
                        **cpu_state_dict,
                        **{f"{prefix}.{k}": v for k, v in module.state_dict().items()},
                    }
                dist.barrier()
                torch.cuda.synchronize()
            if rank == 0:
                print("Saving trained LoRA weights.")
                save_file(
                    cpu_state_dict,
                    os.path.join(
                        train_args["output_dir"], "model_state_dict.safetensors"
                    ),
                )
                print("Done", rank)
        else:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state_dict = model.state_dict()
                if rank == 0:
                    print("Saving full model weights.")
                    save_file(
                        cpu_state_dict,
                        os.path.join(
                            train_args["output_dir"], "model_state_dict.safetensors"
                        ),
                    )
                    print("Done", rank)

    dist.barrier()  # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()


def validate_args(args):
    if args["n_bits"] != 4 and args["train_type"] not in [
        "hqq_lora",
        "hqq_dora",
        "hqq_llama_pro",
    ]:
        raise ValueError(
            (
                f"""train_type={args['train_type']} doesn't support n_bits={args['n_bits']}."""
                """Either don't pass n_bits (to use default of 4) or use any of the hqq training types."""
            )
        )


@hydra.main(config_path="config", config_name="config")
def fsdp_qlora(cfg: DictConfig):
    """
    Train a model with FSDP and QLoRA/QDoRA.

    Args:

        world_size: Number of GPUs to use. -1 = all available GPUs.
        train_type: "full", "lora", "qlora", or "custom_qlora"
        llama_pro_path: Path to the quantized llama pro model
        batch_size: Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
        context_length: Max length of input sequence (in tokens)
        gradient_accumulation_steps: How many steps to accumulate gradients over (increases effective batch size)
        num_epochs: How many epochs of training to do
        dataset: alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
        dataset_samples: Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
        sharding_strategy: Sharding strategy for FSDP
        use_gradient_checkpointing: Use FSDP's activation checkpointing
        reentrant_checkpointing: Use re-entrant autograd activation checkpointing.
            Setting to True can use less GPU memory with BNB QLoRA
        use_cpu_offload: Use FSDP's CPU offloading
        use_activation_cpu_offload: Use FSDP's activation CPU offloading
        low_memory: Load one copy of the model into CPU memory before sharding with FSDP.
            For QLoRA, quantizes each layer individually on GPU before placing on CPU.
        no_sync: Prevent gradient sync until update step. Likely uses more memory.
            Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
        precision: Training precision. autocast precisions use mixed precision
        model_name: Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        save_model: Save the resulting model
        output_dir: Output directory to save the final model to
        lora_rank: LoRA rank for lora/qlora
        lora_alpha: LoRA alpha for lora/qlora
        lora_dropout: LoRA dropout for lora/qlora
        lora_target_modules: If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
        verbose: Whether to print extra info for debugging
        lr: Learning rate
        apply_gradient_clipping: Apply gradient norm clipping
        grad_norm: Gradient norm clipping
        wd: Weight decay
        profile_memory: Profile memory usage for the first few batches.
            Keep false for training. May increase memory usage.
        optimizer: Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
        lr_scheduler: Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
        loading_workers: Number of layers to load and quantize in parallel per GPU.
            Default of -1 uses heuristics to set worker count.
        log_to: Where to log output
        master_addr: For distributed training
        master_port: For distributed training, must be the same for all processes
        seed: Random seed
        project_name: For wandb logging
        name: For wandb logging
        group: For wandb logging
        entity: For wandb logging
        n_bits: passed to hqq
        profiling_output: Output file for profiling
    """

    # Set world size
    if cfg["world_size"] == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    set_seed(args["seed"])
    validate_args(args)
    if args["verbose"]:
        print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if cfg["lora_target_modules"] == "all":
        args["lora_target_modules"] = [
            "k_proj",
            "q_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif cfg["lora_target_modules"].lower() == "default":
        args["lora_target_modules"] = None

    if (
        args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"]
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Current device does not support bfloat16")

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False

    if args["train_type"] in ["hqq_lora"] and HQQLinear is None:
        raise ValueError(
            "HQQ is required to train with `train_type='hqq_lora'`. See ReadMe for details."
        )

    if (
        args["optimizer"] in ["fused_adam", "fused_adamw"]
        and args["use_cpu_offload"]
        and parse(torch.__version__) < parse("2.4dev")
    ):
        raise ValueError(
            (
                f"""Optimizer '{args['optimizer']}' with `use_cpu_offload=True`"""
                """requires at least PyTorch 2.4 Nightly with fused Adam/AdamW CPU support."""
            )
        )

    # Run
    mp.spawn(
        fsdp_main,
        args=(
            world_size,
            args["data"],
            args["training"],
            args["profling"],
            args["optimizer"],
            args["model"],
        ),
        nprocs=torch.cuda.device_count(),
        join=True,
    )


if __name__ == "__main__":
    fsdp_qlora()
