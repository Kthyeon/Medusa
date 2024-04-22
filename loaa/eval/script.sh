CUDA_VISIBLE_DEVICES=3 python ./loaa/eval/heads_accuracy.py --loaa_path ./5_mha_0.25_0405/loaa_lm_head.safetensors --loaa_num_heads 5 --loaa_width 0.25 --save_dir ./data --steps 1024 --model_name_or_path lmsys/vicuna-7b-v1.3 --data_path question.jsonl --shortcut

python ./loaa/eval/gen_results.py --accuracy-path ./data/5_mha_0.25_0405_heads_accuracy.pt --output-path ./data_wd_5_1

# CUDA_VISIBLE_DEVICES=6 python ./loaa/eval/heads_accuracy.py --loaa_path ./loaa_6_width_2.0_self_no_shortcut/loaa_lm_head.safetensors --loaa_num_heads 6 --loaa_width 2.0 --save_dir ./data --steps 1024 --model_name_or_path lmsys/vicuna-7b-v1.3 --data_path question.jsonl

# python ./loaa/eval/gen_results.py --accuracy-path ./data/loaa_6_width_2.0_self_no_shortcut_heads_accuracy.pt --output-path ./data