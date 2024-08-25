import copy
from typing import Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# Dataset class
class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, style="alpaca"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.style = style

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.style == "guanaco":
            prompt = self.dataset[index]["text"].split("### Assistant: ")[0]
            example = self.dataset[index]["text"]
        elif self.style == "qna":
            prompt_template = (
                "###Context:\n{context}\n###Question:\n{question}\n###Answer:\n"
            )
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample["answer"]
        elif self.style == "qna_no_ctx":
            prompt_template = "###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample["answer"]
        else:  # Alpaca
            ann = self.dataset[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]

        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


# And to get the dataloader
def get_dataloader(
    tokenizer: PreTrainedTokenizerFast,
    dataset,
    dataset_samples,
    context_length,
    seed,
    batch_size,
    gradient_accumulation_steps,
) -> DataLoader:
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset

    # Load the source dataset
    if dataset == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    elif dataset == "alpaca_sample":
        dataset = load_dataset(
            "yahma/alpaca-cleaned", split=f"train[:{dataset_samples}]"
        )
    elif dataset == "dummy":
        dataset = Dataset.from_dict(
            {
                "instruction": ["instruction"] * dataset_samples,
                "input": ["input"] * dataset_samples,
                "output": ["output" * context_length * 2] * dataset_samples,
            }  # A long output to test memory usage (gets truncated)
        )
    elif dataset == "guanaco":
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    elif dataset == "sql":
        dataset = load_dataset("knowrohit07/know_sql")["validation"]
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(1000, len(dataset)))
    elif dataset == "orca_math":
        dataset = load_dataset("microsoft/orca-math-word-problems-200k")[
            "train"
        ].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0, dataset_samples))

    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(
        range(
            0,
            len(dataset) - len(dataset) % (batch_size * gradient_accumulation_steps),
        )
    )

    # # Create the InstructionDataset
    if dataset == "guanaco":
        dataset = InstructionDataset(dataset, tokenizer, style="guanaco")
    elif dataset == "sql":
        dataset = InstructionDataset(dataset, tokenizer, style="qna")
    elif dataset == "orca_math":
        dataset = InstructionDataset(dataset, tokenizer, style="qna_no_ctx")
    else:  # (w/ alpaca prompt formatting)
        dataset = InstructionDataset(dataset, tokenizer, style="alpaca")

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )[:, :context_length]
        if with_attention_mask:
            attention_masks = pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )[:, :context_length]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[
            :, :context_length
        ]
        # Return dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=seed)

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler
    )

    return dataloader
