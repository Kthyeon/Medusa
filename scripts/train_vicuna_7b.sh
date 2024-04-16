#!/bin/bash

# Define the arrays for loaa_num_heads and loaa_width
loaa_num_heads=(3 4 5 6 7 8 9)
loaa_width=(0.25 0.5 1.0 2.0 4.0 8.0)

# Iterate over each value of loaa_num_heads
for num_heads in "${loaa_num_heads[@]}"; do
    # Inside that loop, iterate over each value of loaa_width
    for width in "${loaa_width[@]}"; do
        # Execute the training command with the current values of loaa_num_heads and loaa_width
        CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port 29600 loaa/train/train.py --model_name_or_path lmsys/vicuna-7b-v1.3 \
            --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
            --bf16 True \
            --output_dir test \
            --num_train_epochs 1 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate 1e-3 \
            --weight_decay 0.0 \
            --warmup_ratio 0.1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --lazy_preprocess True \
            --loaa_num_heads $num_heads \
            --loaa_num_layers 1 \
            --loaa_width $width
    done
done