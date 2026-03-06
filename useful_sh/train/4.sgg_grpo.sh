export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export MODELSCOPE_CACHE= modelscope/catch
export VLLM_USE_V1=0
export HF_HOME
export NCCL_CONNECT_TIMEOUT=1800


pre_step_model="output/gsr_grpo3/xxxxx/checkpoint-2400" # modify here


# train 3600 step
export MASTER_PORT=29500
MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $pre_step_model \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_sgg format \
    --use_vllm true \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset  NewPath_ms_swift_sgg_grpo_upon_sft_fixcot_train.json \
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
    --output_dir output/sgg_grpo4 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 2 \
    --report_to tensorboard \
    --ddp_backend gloo

