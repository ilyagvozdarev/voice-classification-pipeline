FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 7860
CMD ["python", "-m", "app.main"]
