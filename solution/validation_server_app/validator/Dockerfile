FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# TensorFlow deps (slim needs a few basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY validator_server.py .

EXPOSE 8000
CMD ["uvicorn", "validator_server:app", "--host", "0.0.0.0", "--port", "8000"]
