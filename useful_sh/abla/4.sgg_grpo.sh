
# export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

pre_step_model="/lllidy/lllidy/Projects/Relation-R1/output/Combine_CoTs_sft/v4-20251013-113528/checkpoint-2786" # modify here


# train 3600 step
MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $pre_step_model \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_sgg format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8001 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /lllidy/lllidy/dataset/rel-r1/ms-swift-data-NewPath/NewPath_ms_swift_sgg_grpo_upon_sft_fixcot_train.json \
    --split_dataset_ratio 0 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-7 \
    --eval_steps 100 \
    --save_steps 400 \
    --save_total_limit 10 \
    --max_length 4096 \
    --logging_steps 5 \
    --output_dir output/Combine_CoTs_sft_grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system examples/train/grpo/prompt.txt \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --report_to tensorboard \
    --resume_from_checkpoint /lllidy/lllidy/Projects/Relation-R1/output/Combine_CoTs_sft_grpo/v0-20251013-154440/checkpoint-1600

