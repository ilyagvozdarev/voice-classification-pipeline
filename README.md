# Voice Classification Pipeline

Микросервисный конвейер для **multi-label классификации речи**:
аудио → ASR → восстановление текста LLM → классификация (BERT ‖ LLM) → агрегация.
Пять сервисов в `docker-compose`, одна сеть, обращение друг к другу по имени сервиса.

## Архитектура

```
                 ┌──────────┐
  audio ───────▶ │  gradio  │   (UI: запись/загрузка аудио, порт 7860)
                 └────┬─────┘
                      │ POST /process (multipart audio)
                      ▼
                 ┌──────────┐
                 │  master  │   (оркестратор, порт 8000)
                 └────┬─────┘
                      │ 1) POST /transcribe
                      ▼
                 ┌──────────┐
                 │   asr    │   whisper-tiny → сырой текст
                 └────┬─────┘
                      │ transcript
                      ▼  2) POST /generate  (промпт «восстановление»)
                 ┌──────────┐
                 │   llm    │   диаризация, восстановление, типы сообщений
                 └────┬─────┘
                      │ restored text
          ┌───────────┴───────────────┐   3) параллельно (asyncio.gather)
          ▼                           ▼
     ┌──────────┐               ┌──────────┐
     │   bert   │               │   llm    │  (промпт «классификация»)
     │ /classify│               │ /generate│
     │multi-label│              └────┬─────┘
     └────┬─────┘                    │ llm_classification
          │ bert_labels[]            │
          └───────────┬──────────────┘
                      ▼   4) master агрегирует и возвращает в gradio
     { transcript, restored_text, bert_labels[], llm_classification }
```

Ключевое: LLM вызывается **дважды** — сначала восстановление сырого транскрипта,
потом классификация восстановленного текста (своим промптом). BERT классифицирует
тот же восстановленный текст (multi-label). Оба результата уходят в gradio.

## Стек

- **FastAPI** + **uvicorn** — каждый сервис
- **Pydantic v2** / **pydantic-settings** — схемы запросов/ответов и конфиг из env
- **asyncio** + **aiohttp** — master делает конкурентные вызовы (BERT ‖ LLM)
- **transformers** (CPU) или **vLLM** (GPU) — бэкенд LLM переключается конфигом
- **батчинг запросов в LLM** — `asyncio.Queue` + фоновый воркер + `Future` на запрос
- **gradio** — фронтенд

## Батчинг в LLM-сервисе

LLM-сервис не гоняет модель по одному запросу. Входящие промпты кладутся в
`asyncio.Queue`, а фоновый воркер собирает **батч** и прогоняет его одним вызовом:

- батч уходит в модель, когда набралось `LLM_MAX_BATCH_SIZE` **или** прошло
  `LLM_BATCH_TIMEOUT_MS` с первого запроса в пачке (что раньше);
- каждый HTTP-запрос ждёт свой `Future`, который воркер резолвит после инференса.

Код: [`batcher.py`](services/llm/app/batcher.py) (очередь + воркер) и
[`backends.py`](services/llm/app/backends.py) (сменный бэкенд transformers/vLLM).

## Модели по умолчанию (лёгкие, CPU)

| Сервис | Модель | Меняется через env |
|--------|--------|--------------------|
| asr  | `openai/whisper-tiny` | `ASR_MODEL_NAME` |
| bert | `SamLowe/roberta-base-go_emotions` (multi-label, 28 меток) | `BERT_MODEL_NAME`, `BERT_THRESHOLD` |
| llm  | `google/flan-t5-small` | `LLM_MODEL_NAME` |

Веса кэшируются в volume `hf-cache`, поэтому повторные запуски быстрые.

## Запуск

```bash
cp .env.example .env        # при желании отредактируй модели
```

```bash
# CPU (по умолчанию)
docker compose up --build

# GPU для asr/bert/llm через transformers-CUDA
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# LLM на vLLM (GPU) — быстрый инференс
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

Первый старт качает веса моделей — health-check каждого ML-сервиса имеет
`start_period: 120s`, master ждёт, пока все три станут healthy.

Открыть UI: <http://localhost:7860>

## LLM на vLLM (опционально)

Оверлей `docker-compose.vllm.yml` переключает **только llm-сервис** на бэкенд
vLLM (`Dockerfile.vllm`), прокидывает GPU и меняет модель на causal/instruct
(vLLM заточен под них):

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

Переключение бэкенда — через env `LLM_BACKEND` (`transformers` | `vllm`); код
сервиса один и тот же, различается только реализация бэкенда в `backends.py`.
**Требования:** NVIDIA GPU + драйвер + NVIDIA Container Toolkit.

## Запуск на GPU через transformers (опционально)

Оверлей `docker-compose.gpu.yml` пересобирает asr/bert/llm по `Dockerfile.gpu`
(torch с CUDA) и прокидывает GPU. Код сам определяет устройство через
`torch.cuda.is_available()` и переносит пайплайны на `cuda:0`.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Проверка, что GPU виден Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Порты

| Сервис | Внутренний | Внешний | Endpoint |
|--------|-----------|---------|----------|
| gradio | 7860 | 7860 | UI |
| master | 8000 | 8000 | `POST /process`, `GET /health` |
| asr    | 8000 | 8001 | `POST /transcribe` |
| bert   | 8000 | 8002 | `POST /classify` (multi-label) |
| llm    | 8000 | 8003 | `POST /generate` (принимает `prompt`) |

Внутри сети сервисы ходят по именам: `http://asr:8000` и т.д.

## Проверка без gradio

```bash
# ASR — файл → текст
curl -F "audio=@sample.wav" http://localhost:8001/transcribe

# BERT — multi-label классификация
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" -d '{"text":"I love this, thank you so much!"}'

# LLM — принимает готовый prompt
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" -d '{"prompt":"Classify: I am so happy today"}'

# Весь конвейер (audio -> ASR -> restore -> BERT ‖ LLM)
curl -F "audio=@sample.wav" http://localhost:8000/process
```

Или скрипт: `python scripts/smoke_test.py sample.wav` (нужны запущенные сервисы).

## Структура

```
services/
  asr/    app/{main,config,schemas,model}.py       — whisper → текст
  bert/   app/{main,config,schemas,model}.py       — multi-label классификация
  llm/    app/{main,config,schemas,backends,batcher}.py  — генерация + батчинг
  master/ app/{main,config,schemas,prompts,clients}.py   — оркестрация + промпты
  gradio/ app/{main,config,client}.py              — UI
каждый сервис: Dockerfile (+ Dockerfile.gpu / Dockerfile.vllm), requirements.txt
```
