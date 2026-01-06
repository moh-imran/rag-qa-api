# FROM python:3.11-slim
# WORKDIR /workspace
# RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt /workspace/

# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install --upgrade -r requirements.txt
# COPY . /workspace


FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies once
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimize pip for speed
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy only requirements first
COPY requirements.txt .

# Install heavy dependencies
RUN pip install --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Ensure the app module is in path
ENV PYTHONPATH=/workspace

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
