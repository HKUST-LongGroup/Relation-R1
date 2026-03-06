export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_USE_CUDA_DSA=1


pre_step_model= "output/tiny_qwen_cot2/xxxx/checkpoint-126" # modify here


# train 2400 step
MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $pre_step_model \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_gsr format \
    --use_vllm true \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset  NewPath_ms_swift_gsr_grpo_upon_sft_fixcot_wo_Activity_QwenCoT_train.json \
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
    --output_dir output/gsr_grpo3 \
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
    --num_infer_workers 2 \
    --gradient_accumulation_steps 1 \
    --beta 0.001 \
    --seed 2025 \
    --report_to tensorboard
