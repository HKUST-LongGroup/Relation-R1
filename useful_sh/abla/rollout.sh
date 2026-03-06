# CUDA_VISIBLE_DEVICES=6,7 \
# swift rollout \
#     --model /lllidy/lllidy/pretrain-ckpts/Qwen/Qwen2.5-VL-3B-Instruct \
#     --data_parallel_size 2
    
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model /lllidy/lllidy/Projects/Relation-R1/output/Combine_CoTs_sft/v4-20251013-113528/checkpoint-2786 \
    --data_parallel_size 2