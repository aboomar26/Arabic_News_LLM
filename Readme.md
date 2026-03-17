# Arabic News NLP

Supervised fine-tuning of Qwen2.5-1.5B for structured extraction and translation of Arabic news articles — served via vLLM with a FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![vLLM](https://img.shields.io/badge/vLLM-0.7.2-orange)
![LoRA](https://img.shields.io/badge/LoRA-rank_64-purple)

---

## The Problem

Arabic news is produced at scale but rarely in a structured format. Extracting metadata requires either manual annotation or a large general-purpose LLM at every inference call.

This project fine-tunes a small, fast **1.5B model** for two structured tasks — with no per-call cost:

| Task | Output |
|------|--------|
| **Details extraction** | Title · keywords · summary · category · NER entities |
| **Translation** | Arabic → English (structured JSON) |

---

## Architecture

```
┌─────────────────────────────────┐   ┌──────────────────────────────────────┐
│         TRAINING                │   │            INFERENCE                 │
│                                 │   │                                      │
│  Arabic news ──load──► DeepSeek │   │  Arabic input ──► Prompt builder     │
│      │                    │     │   │       │                  │            │
│   format              SFT pairs │   │   tokenize            apply          │
│      │                    │     │   │       │                  │            │
│  SFT formatter ─train─► LLaMA- │   │  vLLM :8000 ◄─load─ LoRA adapter    │
│      │                Factory   │   │       │                              │
│     log                save     │   │    decode                            │
│      │                  │       │   │       │                              │
│  W&B tracking      LoRA adapter │   │  Pydantic parser ──► JSON output    │
└─────────────────────────────────┘   └──────────────────────────────────────┘
```

> **Note:** GPT-4o is the intended teacher model for knowledge distillation.
> DeepSeek-R1 (via HuggingFace router) was used in this demo as a free alternative.

---

## Numbers

| Metric | Value |
|--------|-------|
| Training samples | 2,600 |
| LoRA rank | 64 |
| Epochs | 3 |
| Throughput | ~24 tok/sec |

---

## Fine-tuning Pipeline

**1. Generate labeled data**
Knowledge distillation — teacher model labels raw Arabic articles with structured JSON outputs matching the target Pydantic schema.

**2. Format for LLaMA-Factory**
Each sample formatted as `{ system, instruction, input, output, history }` — the alpaca format LLaMA-Factory SFT expects.

**3. Register the dataset**
Paths added to `dataset_info.json` with column mappings: `prompt → instruction`, `response → output`.

**4. Train with LoRA**
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
finetuning_type: lora
lora_rank: 64
lora_target: all
num_train_epochs: 3.0
learning_rate: 1.0e-4
lr_scheduler_type: cosine
report_to: wandb
```

---

## Serving with vLLM

```bash
vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
  --dtype=half \
  --gpu-memory-utilization 0.8 \
  --max_lora_rank 64 \
  --enable-lora \
  --lora-modules news-lora="./model"
```

---

## Getting Started

### Requirements
- Python 3.11+
- NVIDIA GPU (vLLM requires Linux/WSL2 on Windows)
- Conda or venv

### Install
```bash
git clone https://github.com/your-username/arabic-news-nlp-api.git
cd arabic-news-nlp-api
pip install -r requirements.txt
```

### Add model files
Place your LoRA checkpoint files in `./model/`:
```
model/
├── adapter_model.safetensors
├── adapter_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
└── added_tokens.json
```

### Run (Windows — one click)
```bash
scripts\start_vllm.bat
# Opens 3 terminals automatically:
#   vLLM     → localhost:8000
#   FastAPI  → localhost:8080
#   Frontend → localhost:3000
```

### Run (Linux/WSL2)
```bash
# Terminal 1 — vLLM
bash scripts/start_vllm.sh

# Terminal 2 — FastAPI
uvicorn app.main:app --port 8080

# Terminal 3 — Frontend
cd frontend && python -m http.server 3000
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API + vLLM status |
| `POST` | `/api/v1/extract` | Extract article details |
| `POST` | `/api/v1/translate` | Translate to target language |

### Extract
```bash
curl -X POST http://localhost:8080/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"story": "نص المقال العربي..."}'
```

### Translate
```bash
curl -X POST http://localhost:8080/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"story": "نص المقال...", "target_lang": "English"}'
```

---

## Output Schemas

### Details extraction
```json
{
  "story_titel": "...",
  "story_keywords": ["...", "..."],
  "story_sammary": ["...", "..."],
  "story_category": "economy",
  "story_entities": [
    {"entity_value": "فوربس", "entity_tupe": "organization"}
  ]
}
```

### Translation
```json
{
  "translated_titel": "...",
  "translated_content": "..."
}
```

**Entity types:** `person-male` · `person-female` · `location` · `organization` · `event` · `time` · `quantity` · `money` · `product` · `law` · `disease` · `artifact`

**Story categories:** `politics` · `sports` · `art` · `technology` · `economy` · `health` · `entertainment` · `science`

---

## Load Testing

```bash
locust --headless -f locust.py \
  --host=http://localhost:8000 \
  -u 20 -r 1 -t "60s" \
  --html=locust_results.html
```

Results: ~24 tokens/sec · 20 concurrent users · 60s duration

---

## Project Structure

```
arabic-news-nlp-api/
├── app/
│   ├── main.py               # FastAPI app + CORS + lifespan
│   ├── config.py             # settings via pydantic-settings
│   ├── api/v1/
│   │   ├── router.py         # mounts all endpoints
│   │   ├── extract.py        # POST /api/v1/extract
│   │   └── translate.py      # POST /api/v1/translate
│   ├── schemas/
│   │   ├── requests.py       # Pydantic input models
│   │   └── responses.py      # Pydantic output models
│   └── services/
│       ├── vllm_client.py    # async httpx client for vLLM
│       ├── prompt_builder.py # Qwen chat-template prompts
│       └── postprocessor.py  # json_repair + CJK char strip
├── frontend/
│   └── index.html            # single-file UI
├── model/                    # LoRA checkpoint (gitignored)
├── scripts/
│   ├── start_vllm.sh
│   └── start_vllm.bat        # Windows one-click launcher
├── locust.py                 # load testing
├── requirements.txt
├── .env                      # gitignored
└── .gitignore
```

---

## Stack

| Layer | Tools |
|-------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | LLaMA-Factory · LoRA rank 64 |
| Teacher | GPT-4o / DeepSeek-R1 |
| Tracking | W&B |
| Serving | vLLM 0.7.2 |
| API | FastAPI · uvicorn · httpx |
| Validation | Pydantic v2 · json-repair |
| Testing | Locust · Faker (ar) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Skills

`supervised fine-tuning` · `knowledge distillation` · `LoRA / PEFT` · `LLM serving` · `REST API design` · `structured output` · `Arabic NLP` · `NER` · `prompt engineering` · `load testing` · `async Python` · `Pydantic`