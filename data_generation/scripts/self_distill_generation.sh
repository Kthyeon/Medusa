# # open server 
# CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server --model /mnt/data1/taehyeon/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6 --port 8000 --max-model-len 8192
# CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --model /mnt/data1/taehyeon/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6 --port 8001 --max-model-len 8192
# CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server --model /mnt/data1/taehyeon/taehyeon/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6 --port 8000 --max-model-len 8192 --tensor-parallel-size=2
# generate data
# simplify below code and would like to make each commands with 5 times (for random seed)
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_03.json --num_threads 16 --max_tokens 4096 --temperature 0.3
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_04.json --num_threads 16 --max_tokens 4096 --temperature 0.4
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_05.json --num_threads 16 --max_tokens 4096 --temperature 0.5
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_06.json --num_threads 16 --max_tokens 4096 --temperature 0.6
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_07.json --num_threads 16 --max_tokens 4096 --temperature 0.7
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_08.json --num_threads 16 --max_tokens 4096 --temperature 0.8
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_09.json --num_threads 16 --max_tokens 4096 --temperature 0.9
# python ./data_generation/generate.py --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output_path self_vicuna_10.json --num_threads 16 --max_tokens 4096 --temperature 1.0
# 
#!/bin/bash

# Settings
data_path="ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
output_prefix="self_vicuna"
num_threads=256
max_tokens=2048 # 8192
base_command="python ./data_generation/generate.py" 

# Loop for temperatures in increments of 0.1
for temp in $(seq 0.7 0.1 1.0); do
    for seed in {1..3}; do  # Loop for 5 random seeds
        output_path="${output_prefix}_${temp}_seed${seed}.json"
        command="${base_command} --data_path ${data_path} --output_path ${output_path} --num_threads ${num_threads} --max_tokens ${max_tokens} --temperature ${temp} --chat"

        echo "Running command: ${command}"
        $command  # Execute the command
    done
done