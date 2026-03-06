# source /mllm_hdd/chenwei21/anaconda3/bin/activate msswift # modify here

# set -x

# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY="7c505d353966003c03ba00bc43bf5385dd2a9069" # modify here

nproc_per_node=8

pre_step_model="/lllidy/lllidy/Projects/Relation-R1/output/Only_sgg_Qwen3B-template-sft/v0-20251009-221522/checkpoint-2660" # modify here

# totally train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model $pre_step_model \
    --train_type full \
    --dataset /lllidy/lllidy/dataset/rel-r1/ll_tiny_split_from_merge_json/sgg_tiny.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 5120 \
    --output_dir output/Only_sgg_Qwen3B-template-MLLM_sft \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --split_dataset_ratio 0 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --deepspeed zero2 \
    --report_to tensorboard

