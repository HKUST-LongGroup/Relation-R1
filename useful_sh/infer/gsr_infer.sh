
NPROC_PER_NODE=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
swift infer \
    --model checkpoint-1200 \
    --infer_backend vllm \
    --max_new_tokens 4096 \
    --gpu_memory_utilization 0.7 \
    --val_dataset NewPath_ms_swift_gsr_with_think_eval.json \
    --max_batch_size 4 \
    --result_path results.jsonl