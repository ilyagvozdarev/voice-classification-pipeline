# Voice ML Pipeline

Микросервисный конвейер: аудио → распознавание речи → (тональность ‖ ответ LLM) → агрегация.
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
                 │   asr    │   whisper-tiny → текст
                 └────┬─────┘
                      │ текст
          ┌───────────┴───────────┐   2) параллельно (asyncio.gather)
          ▼                       ▼
     ┌──────────┐           ┌──────────┐
     │   bert   │           │   llm    │
     │ /classify│           │ /generate│
     └────┬─────┘           └────┬─────┘
          │ label/score          │ answer
          └───────────┬──────────┘
                      ▼   3) master агрегирует и возвращает в gradio
              { transcript, sentiment, answer }
```

## Стек

- **FastAPI** + **uvicorn** — каждый сервис
- **Pydantic v2** / **pydantic-settings** — схемы запросов/ответов и конфиг из env
- **asyncio** + **aiohttp** — master делает конкурентные вызовы к bert и llm
- **transformers** (CPU torch) — реальные лёгкие модели
- **gradio** — фронтенд

## Модели по умолчанию (лёгкие, CPU)

| Сервис | Модель | Меняется через env |
|--------|--------|--------------------|
| asr  | `openai/whisper-tiny` | `ASR_MODEL_NAME` |
| bert | `distilbert-base-uncased-finetuned-sst-2-english` | `BERT_MODEL_NAME` |
| llm  | `google/flan-t5-small` | `LLM_MODEL_NAME` |

Веса скачиваются при первом старте и кэшируются в volume `hf-cache`, поэтому
повторные запуски быстрые.

## Запуск

```bash
cp .env.example .env        # при желании отредактируй модели
```

```bash
# CPU (по умолчанию)
docker compose up --build

# GPU (opt-in через второй -f)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Первый старт качает веса моделей — health-check каждого ML-сервиса имеет
`start_period: 120s`, master ждёт, пока все три станут healthy.
Подробности про GPU — в разделе [ниже](#запуск-на-gpu-опционально).

Открыть UI: <http://localhost:7860>

## Запуск на GPU (опционально)

По умолчанию всё считается на **CPU** (torch собран без CUDA). Для GPU есть
оверлей `docker-compose.gpu.yml`, который накладывается поверх базового:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Что делает оверлей:
- пересобирает asr/bert/llm по `Dockerfile.gpu` (torch с CUDA, cu124);
- добавляет `deploy.resources.reservations.devices` — прокидывает NVIDIA GPU в контейнер;
- тегает образы как `voice-ml/*:gpu`, чтобы не затирать CPU-образы.

Код менять не нужно: сервисы сами определяют устройство через
`torch.cuda.is_available()` и переносят пайплайны на `cuda:0`. В логах при старте
видно `device=cuda:0` (GPU) или `device=cpu`.

**Требования на хосте:** NVIDIA GPU + драйвер + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Проверка, что GPU виден Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Без оверлея (`docker compose up`) всё остаётся на CPU — GPU строго opt-in.
Файл называется `*.gpu.yml`, а не `*.override.yml`, поэтому автоматически он не
подхватывается.

## Порты

| Сервис | Внутренний | Внешний | Endpoint |
|--------|-----------|---------|----------|
| gradio | 7860 | 7860 | UI |
| master | 8000 | 8000 | `POST /process`, `GET /health` |
| asr    | 8000 | 8001 | `POST /transcribe` |
| bert   | 8000 | 8002 | `POST /classify` |
| llm    | 8000 | 8003 | `POST /generate` |

Внутри сети сервисы ходят по именам: `http://asr:8000` и т.д.

## Проверка без gradio

```bash
# ASR напрямую
curl -F "audio=@sample.wav" http://localhost:8001/transcribe

# BERT
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" -d '{"text":"I love this"}'

# LLM
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" -d '{"text":"What is the capital of France?"}'

# Весь конвейер
curl -F "audio=@sample.wav" http://localhost:8000/process
```

Или скрипт: `python scripts/smoke_test.py sample.wav` (нужны запущенные сервисы).

## Структура

```
services/<name>/
  app/
    main.py       # FastAPI app + endpoints + lifespan
    config.py     # pydantic-settings, env
    schemas.py    # pydantic-модели запросов/ответов
    model.py      # обёртка над HF-моделью (asr/bert/llm)
    clients.py    # aiohttp-клиенты к downstream (только master)
    client.py     # requests-клиент к master (только gradio)
  Dockerfile
  requirements.txt
```
