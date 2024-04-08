# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, List


import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from loaa.model.loaa_model import LoaaModel, LoaaConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

VICUNA_CHAT_TEMPLATE = open('./vicuna.jinja').read()
VICUNA_CHAT_TEMPLATE = VICUNA_CHAT_TEMPLATE.replace('    ', '').replace('\n', '')

# Customized for training loaa heads
class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            loaa = model.module.loaa
        else:
            loaa = model.loaa

        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(loaa):
            loaa_logits = logits[i, :, : -(2 + i)].contiguous()
            loaa_labels = labels[..., 2 + i :].contiguous()
            loaa_logits = loaa_logits.view(-1, logits.shape[-1])
            loaa_labels = loaa_labels.view(-1)
            loaa_labels = loaa_labels.to(loaa_logits.device)
            loss_i = loss_fct(loaa_logits, loaa_labels)
            loss += loss_i
            not_ignore = loaa_labels.ne(IGNORE_TOKEN_ID)
            loaa_labels = loaa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 2):
                _, topk = loaa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(loaa_labels.unsqueeze(-1)).any(-1)
                log[f"loaa{i}_top{k}"] = correct.float().mean().item()

            log[f"loaa{i}_loss"] = loss_i.item()
        self.log(log)
        return (loss, logits) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/data/taehyeon/')
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    loaa_num_heads: int = field(
        default=9,
        metadata={"help": "Number of loaa heads."},
    )
    loaa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each loaa head."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    sources: List[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources, each a dictionary with an 'id' and 'conversations'.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    prompts = []
    for source in sources:
        conversation = source['conversations']
        
        # Building the conversation string based on your earlier format requirements
        formatted_conversation = ""
        for message in conversation:
            prefix = "human: " if message['from'] == 'human' else "ASSISTANT: "
            formatted_conversation += prefix + message['value'].strip() + '\n'

        prompts.append(formatted_conversation[:-1])  # Remove the last newline character

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    targets = torch.full(encoding.input_ids.shape, IGNORE_TOKEN_ID, dtype=torch.long)

    # Correcting label application using the tokenizer's decode method to match prompts with original text
    for i, prompt in enumerate(prompts):
        offset_mapping = encoding['offset_mapping'][i]
        input_ids = encoding['input_ids'][i]
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)

        start_index = 0
        for message in sources[i]['conversations']:
            if message['from'] == 'gpt':
                content = message['value'].strip()
                start_position = decoded_input.find(content, start_index)
                
                if start_position != -1:
                    start_index = start_position + len(content)
                    start_token, end_token = None, None

                    for j, (start, end) in enumerate(offset_mapping):
                        if start <= start_position < end:
                            start_token = j
                        if start < start_index <= end:
                            end_token = j
                            break

                    if start_token is not None and end_token is not None:
                        targets[i, start_token:end_token + 1] = input_ids[start_token:end_token + 1]

    return {
        "input_ids": encoding.input_ids,
        "labels": targets,
        "attention_mask": encoding.attention_mask,
    }


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Generate loaa config for pushing to HF hub
    loaa_config = LoaaConfig(
        loaa_num_heads=training_args.loaa_num_heads,
        loaa_num_layers=training_args.loaa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # Add loaa heads
    loaa_lm_head = LoaaModel(loaa_config, model)

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_loaa_mlp_{model_args.model_name_or_path.split('/')[-1]}_loaa_{training_args.loaa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.loaa_num_layers}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)



    # Save loaa config
    loaa_config.save_pretrained(training_args.output_dir)

    # Start trainner
    trainer = CustomizedTrainer(
        model=loaa_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save loaaHead seperately
    if hasattr(loaa_lm_head, "module"):
        lm_head = loaa_lm_head.module.loaa_head
    else:
        lm_head = loaa_lm_head.loaa_head
    import deepspeed
    with deepspeed.zero.GatheredParameters(lm_head.parameters()):
        state_dict = lm_head.state_dict()

    # Save loaa heads
    if local_rank == 0:
        # Modify the tokenizer internal state before saving.
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "loaa_lm_head.safetensors"),
        )


if __name__ == "__main__":
    train()
