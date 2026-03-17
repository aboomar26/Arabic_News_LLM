@echo off
conda activate nlpnews

vllm serve "Qwen/Qwen2.5-1.5B-Instruct" ^
  --dtype=half ^
  --gpu-memory-utilization 0.8 ^
  --max_lora_rank 64 ^
  --enable-lora ^
  --lora-modules news-lora="./model"