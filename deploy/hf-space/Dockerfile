# Kuber live-inference demo — Hugging Face Docker Space (CPU).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache \
    KUBER_ASSETS=/app/assets

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/assets /app/.cache && chmod -R 777 /app/.cache /app/assets

EXPOSE 7860
CMD ["python", "boot.py"]
