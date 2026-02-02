# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /workspace

# Environment variables configuration
ARG PYTHONUNBUFFERED=1
ARG PYTHONPATH=/workspace
ARG OPENAI_API_KEY=sk-xxxx
ARG CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG EMBEDDING_PROVIDER=huggingface
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG MONGODB_URL=""
ARG QDRANT__SERVICE__API_KEY=""
ARG QDRANT_API_KEY=""

ENV PYTHONUNBUFFERED=$PYTHONUNBUFFERED
ENV PYTHONPATH=$PYTHONPATH
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV CROSS_ENCODER_MODEL=$CROSS_ENCODER_MODEL
ENV EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER
ENV EMBEDDING_MODEL=$EMBEDDING_MODEL
ENV MONGODB_URL=$MONGODB_URL
ENV QDRANT__SERVICE__API_KEY=$QDRANT__SERVICE__API_KEY
ENV QDRANT_API_KEY=$QDRANT_API_KEY

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    git \
    # OCR and PDF dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Optimize pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Step 1: Install heavy, stable dependencies first (Torch CPU)
# This layer stays cached unless we change the version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Copy and install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries 10 --prefer-binary -r requirements.txt

# Step 3: Install Playwright (Browsers are independent of code)
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code (only when code changes)
COPY . .

ENV PYTHONPATH=/workspace

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
