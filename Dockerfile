# syntax=docker/dockerfile:1

############################
# 1) Builder stage
############################
# Современный стек 2026:
# - Python 3.12
# - CUDA 12.4.1 (Docker образ)
# - PyTorch 2.9.x (устанавливается через vllm)
# - Flash Attention 2.8.3 (pre-built wheel для torch 2.9 + cu12 + Python 3.12)
# - Liger Kernel 0.6.4 (чистый Python/Triton)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    software-properties-common \
    cmake \
    build-essential \
    ninja-build \
    libgl1 \
    libgomp1 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv (быстрый pip, 10-100x быстрее)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# venv для зависимостей через uv
ENV VENV_PATH=/opt/venv
RUN uv venv ${VENV_PATH} --python python3.12
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV VIRTUAL_ENV="${VENV_PATH}"

WORKDIR /app

# Сначала зависимости (для кеширования слоёв)
COPY requirements.txt /app/requirements.txt

# ============================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ ЧЕРЕЗ UV с кэшированием между сборками
# --mount=type=cache сохраняет скачанные пакеты на хосте
# ============================================================

# 1. PyTorch 2.9.0 — ФИКСИРОВАННАЯ ВЕРСИЯ (совместима с flash-attn wheel)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    torch==2.9.0 \
    torchvision \
    torchaudio

# 2. Flash Attention 2.8.3 — PRE-BUILT WHEEL для torch 2.9 + cu12 + Python 3.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. Основные зависимости (без vllm — он отдельно)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/requirements.txt

# 4. vLLM — ставим с --no-deps чтобы не переустанавливать torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps vllm \
 && uv pip install vllm --no-build-isolation 2>/dev/null || true

# 5. DeepSpeed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install deepspeed

# 6. Unsloth — ставим ПОСЛЕ основных пакетов с --no-deps
# чтобы не перезаписывать уже установленные transformers/peft/trl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps unsloth unsloth-zoo \
 || echo "Warning: Unsloth installation failed, continuing without it"

# Теперь код (ВАЖНО: .dockerignore должен исключать datasets/out/.runs и т.п.)
COPY . /app

# Нормализуем sh-скрипты, но не падаем, если папки нет
RUN if [ -d scripts ]; then \
      find scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} \; -exec chmod +x {} \;; \
    fi

# Установим проект через uv
RUN uv pip install -e .

############################
# 2) Runtime stage
############################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime

LABEL com.modelsathome.image="models-at-home-studio"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    libgl1 \
    libgomp1 \
    # Для JIT компиляции DeepSpeed ops (cpu_adam и др.)
    build-essential \
    ninja-build \
    libaio-dev \
 && rm -rf /var/lib/apt/lists/*

# Гарантируем запись в /etc/passwd для uid/gid=1000 (нужно для getpass.getuser()).
# Это важно при запуске контейнера с user: "1000:1000" в docker-compose.
RUN getent group 1000 >/dev/null || groupadd -g 1000 appgroup \
 && id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 -g 1000 -s /bin/bash appuser

# Подхватываем venv из builder
ENV VENV_PATH=/opt/venv
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app
COPY --from=builder /app /app

# Директории под монтирования (не обязательно, но удобно)
RUN mkdir -p /app/datasets /app/out /app/.runs
RUN mkdir -p /root/.triton/autotune

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MKL_THREADING_LAYER=GNU
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["./scripts/run_studio.sh"]
