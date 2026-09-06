# Voice Classification Pipeline

Microservice pipeline for **multi-label speech classification**:
audio -> ASR -> LLM text restoration -> classification (BERT || LLM) -> aggregation.


## Architecture

LLM is called twice: first to restore the raw transcript, then to classify the restored text (by prompt).
BERT classifies the same restored text (multi-label). Both results are fed to gradio.

```
                 +----------+
  audio -------> |  gradio  |   (UI: record/upload audio)
                 +----+-----+
                      | POST /process (multipart audio)
                      v
                 +----------+
                 |  master  |
                 +----+-----+
                      | 1) POST /transcribe
                      v
                 +----------+
                 |   asr    |   whisper-tiny -> transcript
                 +----+-----+
                      | transcript (over master)
                      v  2) POST /generate  (prompt "restoration")
                 +----------+
                 |   llm    |   diarization, restoration, message types
                 +----+-----+
                      | restored text (over master)
          +-----------+---------------+   3) parallel (asyncio.gather)
          v                           v
     +----------+               +----------+
     |   bert   |               |   llm    |  (prompt "classification")
     | /classify|               | /generate|
     |multilabel|               +----+-----+
     +----+-----+                    | llm_classification
          | bert_labels[]            |
          +-----------+--------------+
                      v   4) master aggregates and returns to gradio
     { transcript, restored_text, bert_labels[], llm_classification }
```


## Stack

- **FastAPI** + **uvicorn** — every service
- **pydantic v2** / **pydantic-settings** — request/response schemas and config from env
- **asyncio** + **aiohttp** — master issues concurrent calls (BERT || LLM)
- **transformers** (CPU) or **vLLM** (GPU) — the LLM backend is switched by config
- **request batching in the LLM** — `asyncio.Queue` + background worker + a `Future` per request
- **gradio** — frontend


## Implementation details

#### Batching in the LLM service

Incoming prompts are put into an `asyncio.Queue`, and a background worker gathers a **batch** and runs it in a single call:

- the batch goes to the model once `LLM_MAX_BATCH_SIZE` is reached **or**
  `LLM_BATCH_TIMEOUT_MS` has elapsed since the first request in the batch (whichever comes first);
- each HTTP request awaits its own `Future`, which the worker resolves after inference.

Code: [`batcher.py`](services/llm/app/batcher.py) (queue + worker) and
[`backends.py`](services/llm/app/backends.py) (swappable transformers/vLLM backend).


## Run

```bash
cp .env.example .env
```

```bash
# CPU (default)
docker compose up --build

# GPU for asr/bert/llm via transformers
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# LLM on vLLM (GPU)
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

**healthcheck**:
The first start downloads the model weights of every ML service with a timeout `start_period: 120s`, 
and master waits until all services become healthy.

**Ports**:
Inside the network the services address each other by name: `http://asr:8000` and so on.

Open the UI: <http://localhost:7860>

**Checking without gradio**:

```bash
# ASR - file -> text
curl -F "audio=@sample.wav" http://localhost:8001/transcribe

# BERT
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" -d '{"text":"classify this text"}'

# LLM
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" -d '{"prompt":"classify this text"}'

# The whole pipeline (audio -> ASR -> restore -> BERT || LLM)
curl -F "audio=@sample.wav" http://localhost:8000/process
```

Or the script: `python scripts/smoke_test.py sample.wav` (the services must be running).

## Layout

```
services/
  asr/    app/{main,config,schemas,model}.py             - whisper -> text
  bert/   app/{main,config,schemas,model}.py             - multi-label classification
  llm/    app/{main,config,schemas,backends,batcher}.py  - generation + batching
  master/ app/{main,config,schemas,prompts,clients}.py   - orchestration + prompts
  gradio/ app/{main,config,client}.py                    - UI
every service: Dockerfile (+ Dockerfile.gpu / Dockerfile.vllm), requirements.txt
```
