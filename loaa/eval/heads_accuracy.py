import os
import torch
import json
from contextlib import contextmanager
import numpy as np
from loaa.model.loaa_model import LoaaModel
from loaa.model.kv_cache import *
from loaa.model.utils import *
from loaa.model.loaa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse

def get_accuracies(loaa, logit):
    # get the correct counts of each head
    seq_len, choices, topk = loaa.shape
    results = []
    for choice in range(choices):
        results.append(loaa[:-choice - 1,choice].eq(logit[choice + 1:,0]))
    return results



def main(args):
    model = LoaaModel.from_pretrained(
        args.model_path,
        loaa_num_heads=args.loaa_num_heads,
        loaa_width=args.loaa_width,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()


    data = json.load(open(args.data_path))
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    results = None

    for sample in tqdm((data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["instruction"])
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
                input_ids, past_key_values=past_key_values, output_orig=True, loaa_forward=True
            )
            _, loaa_topk = loaa_logits[...,-1,:].topk(20, dim=-1)
            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            loaa_topk_ids.append(loaa_topk.detach().cpu())
            for _ in range(steps):
                loaa_logits, outputs, logits = model(
                    input_id, past_key_values=past_key_values, output_orig=True, loaa_forward=True
                )
                _, loaa_topk = loaa_logits[...,-1,:].topk(20, dim=-1)
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

    save_path = os.path.join(args.save_dir, args.model_name + "_heads_accuracy.pt")
    torch.save(results, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loaa Model Evaluator")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained Loaa model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--loaa_num_heads", type=int, default=5,
                        help="Number of loaa heads.")
    parser.add_argument("--loaa_width", type=float, default=0.25,
                        help="projection width of the loaa heads.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation data in JSON format.")
    parser.add_argument("--save_dir", type=str, default="../../data",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)