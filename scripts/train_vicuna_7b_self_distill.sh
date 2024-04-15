#!/bin/bash

# Define the arrays for loaa_num_heads and loaa_width
loaa_num_heads=(6)
loaa_width=(0.25 0.5)
loaa_shortcut=(False)

for shortcut in "${loaa_shortcut[@]}"; do
    for num_heads in "${loaa_num_heads[@]}"; do
        for width in "${loaa_width[@]}"; do
            CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port 29500 loaa/train/train.py --model_name_or_path lmsys/vicuna-7b-v1.3 \
                --data_path self_vicuna_0.3_reformat.json \
                --bf16 True \
                --output_dir test \
                --num_train_epochs 2 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps 16 \
                --evaluation_strategy "no" \
                --save_strategy "no" \
                --learning_rate 5e-4 \
                --weight_decay 0.0 \
                --warmup_ratio 0.1 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --tf32 True \
                --model_max_length 4096 \
                --lazy_preprocess True \
                --loaa_num_heads $num_heads \
                --loaa_num_layers 1 \
                --loaa_width $width \
                --short_cut $shortcut
        done
    done
done