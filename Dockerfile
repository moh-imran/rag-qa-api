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
ARG MONGODB_URL=mongodb://mongodb:27017/rag_chat

ENV PYTHONUNBUFFERED=$PYTHONUNBUFFERED
ENV PYTHONPATH=$PYTHONPATH
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV CROSS_ENCODER_MODEL=$CROSS_ENCODER_MODEL
ENV EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER
ENV EMBEDDING_MODEL=$EMBEDDING_MODEL
ENV MONGODB_URL=$MONGODB_URL

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

# Copy only requirements first (for layer caching)
COPY requirements.txt .

# Install dependencies with BuildKit cache mount
# This caches downloaded packages between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# Install Playwright and Chromium browser for JS-heavy sites
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code (only when code changes)
COPY . .

ENV PYTHONPATH=/workspace

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
