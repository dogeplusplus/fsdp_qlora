import math
import functools
from typing import Dict, Optional
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup


def _get_cosine_one_cycle_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_fraction=0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = 1 - min_lr_fraction
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return (math.cos(math.pi * progress) + 1) * 0.5 * scale_term + min_lr_fraction


def get_cosine_one_cycle_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_fraction: float = 0.1,
):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    gradient_accumulation_steps: int,
    lr_scheduler: str,
    num_epochs: int,
):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = (
        num_epochs * len(dataloader) // gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * 0.1)
    if lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif lr_scheduler == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(
            optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1
        )
    elif lr_scheduler == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(
            f"{lr_scheduler} LR scheduler not implemented yet"
        )
    return lr_scheduler, num_training_steps


def get_optimizer(model: nn.Module, optimizer: str, lr: float, wd: Optional[float] = None) -> optim.Optimizer:
    """Returns an optimizer. We can add more options here if needed."""
    if optimizer in ["adam", "fused_adam"]:
        return optim.Adam(
            model.parameters(), lr=lr, fused=optimizer == "fused_adam"
        )
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adadelta":
        return optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer in ["adamw", "fused_adamw"]:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=wd,
            fused=optimizer == "fused_adamw",
        )
    else:
        raise ValueError("Invalid optimizer")
