import os
import torch
import json
from contextlib import contextmanager
import numpy as np

from loaa.model.kv_cache import *
from loaa.model.utils import *
from loaa.model.loaa_choices import *
from loaa.model.loaa_model import LoaaModel, LoaaConfig
from loaa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM

from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse
import transformers

from safetensors.torch import load_file



def get_accuracies(loaa, logit):
    # get the correct counts of each head
    seq_len, choices, topk = loaa.shape
    results = []
    for choice in range(choices):
        results.append(loaa[:-choice - 1,choice].eq(logit[choice + 1:,0]))
    return results



def main(args):
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    model = KVLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    loaa_config = LoaaConfig(
        loaa_num_heads=args.loaa_num_heads,
        loaa_num_layers=1,
        loaa_width = args.loaa_width,
        base_model_name_or_path=args.model_name_or_path,
        shortcut = args.shortcut,
        cache_dir=args.cache_dir,
    )

    model = LoaaModel(loaa_config, model)

    print(model.loaa_head)
    state_dict = load_file(args.loaa_path)
    model.loaa_head.load_state_dict(state_dict, strict=True)
    model.loaa_head = model.loaa_head.bfloat16()

    model = model.cuda()
    tokenizer = model.get_tokenizer()


    data = []
    with open(args.data_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    results = None

    for sample in tqdm((data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["turns"][0])
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        steps = args.steps
        logits_ids = []
        loaa_topk_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            model.current_length_data.zero_() # this is for rerun
            reset_loaa_mode(model)
            loaa_logits, outputs, logits = model(
                input_ids, past_key_values=past_key_values, output_orig=True
            )
            _, loaa_topk = loaa_logits[...,-1,:].topk(32, dim=-1)
            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            loaa_topk_ids.append(loaa_topk.detach().cpu())
            for _ in range(steps):
                loaa_logits, outputs, logits = model(
                    input_id, past_key_values=past_key_values, output_orig=True
                )
                _, loaa_topk = loaa_logits[...,-1,:].topk(32, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                loaa_topk_ids.append(loaa_topk.detach().cpu())
            logits_ids = torch.stack(logits_ids, dim=0)
            loaa_topk_ids = torch.stack(loaa_topk_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(loaa_topk_ids, logits_ids)
            else:
                # cat sub results
                cur_results = get_accuracies(loaa_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    save_path = os.path.join(args.save_dir, f"{args.loaa_path.split('/')[-2]}_heads_accuracy.pt")
    torch.save(results, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loaa Model Evaluator")
    parser.add_argument("--cache_dir", type=str, default='/mnt/data1/taehyeon/',
                        help="huggingface cache dir")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the target pre-trained model.")
    parser.add_argument("--loaa_path", type=str, required=True,
                        help="Path to loaa heads")
    parser.add_argument("--loaa_num_heads", type=int, default=6,
                        help="Number of loaa heads.")
    parser.add_argument("--loaa_width", type=float, default=0.25,
                        help="projection width of the loaa heads.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation data in JSON format.")
    parser.add_argument("--save_dir", type=str, default="../../data",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    parser.add_argument("--shortcut", action="store_true",
                        help="Use shortcut for loaa.")
    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)