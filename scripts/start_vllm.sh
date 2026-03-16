

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH="./model"
MODEL_ID="news-lora"

vllm serve "$BASE_MODEL" \
  --dtype=half \
  --gpu-memory-utilization 0.8 \
  --max_lora_rank 64 \
  --enable-lora \
  --lora-modules "$MODEL_ID"="$ADAPTER_PATH"