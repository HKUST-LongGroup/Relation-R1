source /mllm_hdd/chenwei21/anaconda3/bin/activate msswift # modify here

set -x

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="7c505d353966003c03ba00bc43bf5385dd2a9069" # modify here

nproc_per_node=8

pre_step_model="/lllidy/lllidy/Projects/Relation-R1/output/grpo/v6-20251007-123016/checkpoint-3600" # modify here


# progressive grpo train 1200 step
MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $pre_step_model \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_sgg external_gsr format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8001 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset  /lllidy/lllidy/dataset/rel-r1/ms-swift-data-NewPath/NewPath_ms_swift_merged_sgggsr_grpo_upon_sft_fixcot_wo_Activity_train.json \
    --split_dataset_ratio 0 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --eval_steps 100 \
    --save_steps 400 \
    --save_total_limit 10 \
    --max_length 4096 \
    --logging_steps 5 \
    --output_dir output/grpo \
    --warmup_ratio 0.005 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --gradient_accumulation_steps 2 \
    --beta 0.001 \
    --seed 2025 \
    --report_to tensorboard

