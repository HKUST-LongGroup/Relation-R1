nproc_per_node=8
# totally train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model  pretrain-ckpts/Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --dataset NewPath_ms_swift_merged_sgggsr_sft_fixcot_wo_Activity_train.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 5120 \
    --output_dir output/fix_sft1 \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --split_dataset_ratio 0 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --deepspeed zero2 \
    --report_to tensorboard

