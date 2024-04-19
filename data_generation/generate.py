import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

import argparse
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8001/v1"

api_base_pool = []

# List models API
for i in range(10):
    openai.api_base = "http://localhost:800{}/v1".format(i)
    try:
        models = openai.Model.list()["data"][0]["id"]
        api_base_pool.append(openai.api_base)
    except:
        break

print("API base pool: ", api_base_pool)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--num_threads", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()

# Assuming the ShareGPT format
data = json.load(open(args.data_path, "r"))

def generate_data(id, messages, idx):
    try:
        openai.api_base = api_base_pool[idx % len(api_base_pool)]
        model_name = openai.Model.list()["data"][0]["id"]

        if args.chat:
            process_chat_messages(id, messages, model_name)
        else:
            process_non_chat_messages(id, messages, model_name)
    except Exception as e:
        print(f"Failed to generate data: {str(e)}")

def process_chat_messages(id, messages, model_name):
    converted_messages = []
    output_messages = []

    if messages[0]["from"] == "gpt":
        append_system_message(messages, converted_messages, output_messages)

    for message in messages[::2]:
        if message["from"] != "human":
            return

        converted_messages.append({"role": "user", "content": message["value"]})
        try:
            process_conversation(model_name, converted_messages, output_messages, message)
        except Exception as e:
            print(f"Conversation processing failed: {str(e)}")
            # reset output_messages even first gpt's response is contained
            output_messages = []
            break
    
    write_output_if_not_empty(output_messages, id)

def append_system_message(messages, converted_messages, output_messages):
    converted_messages.append({"role": "system", "content": messages[0]["value"]})
    output_messages.append(messages[0])
    messages.pop(0)

def process_conversation(model_name, converted_messages, output_messages, message):
    conv = get_conversation_template(model_name)
    prompt = conv.get_prompt()
    response = openai.ChatCompletion.create(
        prompt=prompt,
        model=model_name,
        messages=converted_messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_eos=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        request_timeout=6000
    )

    response_content = response.choices[0]['message']['content'].strip()
    output_messages.extend([
        message,
        {"from": "gpt", "value": response_content}
    ])
    if response.choices[0]['finish_reason'] == "length":
        print("Response terminated due to max length")
        return
    converted_messages.append({"role": "assistant", "content": response_content})

def write_output_if_not_empty(output_messages, id):
    if output_messages:
        with open(args.output_path, "a") as f:
            f.write(json.dumps({"id": id, "conversations": output_messages}) + "\n")
    else:
        print("No messages to output.")

def process_non_chat_messages(id, messages, model_name):
    try:
        conv = get_conversation_template(model_name)
        output_messages = []
        
        if messages[0]["from"] == "system":
            conv.system_message = messages[0]["value"]
            output_messages.append(messages[0])  # Include the system message in the output
            messages = messages[1:]

        # Append user message to the conversation template
        conv.append_message(conv.roles[0], messages[0]["value"])
        output_messages.append({"from": "human", "value": messages[0]["value"]})
        
        # Append a placeholder for the assistant's response
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        # remove eos token
        prompt = prompt.rstrip('</s>')
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            ignore_eos=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            request_timeout=60000
        )
        
        response_text = response.choices[0]['text'].strip()
        response_text = response_text.rstrip('</s>')
        output_messages.append({"from": "gpt", "value": response_text})
        write_output_if_not_empty(output_messages, id=id)
    except Exception as e:
        print(f"Non-chat message processing failed: {str(e)}")

def save_response(id, output_messages):
    with open(args.output_path, "a") as f:
        # Write in shared GPT format, including the prompt and the model's completions
        f.write(json.dumps({"id": id, "conversations": output_messages}) + "\n")

# if output_path exists, count the number of lines and skip the first n data
start = 0
if os.path.exists(args.output_path):
    with open(args.output_path, "r") as f:
        start = len(f.readlines())
        print("Skip first {} data".format(start))

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, sample in enumerate(data[start:]):
            future = executor.submit(
                generate_data,
                sample["id"],
                sample["conversations"],
                idx,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
