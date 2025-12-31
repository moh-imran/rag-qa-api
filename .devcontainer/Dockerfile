FROM python:3.11-slim
WORKDIR /workspace
RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /workspace/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade -r requirements.txt
COPY . /workspace