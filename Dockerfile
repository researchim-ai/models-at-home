# syntax=docker/dockerfile:1

############################
# 1) Builder stage
############################
# Зафиксированная комбинация PRE-BUILT WHEELS (быстрая установка!):
# - CUDA 12.4.1 (Docker образ)
# - PyTorch 2.5.1 (PyPI, bundled CUDA 12.4)
# - Flash Attention 2.8.3 (pre-built wheel для torch 2.5 + cu12)
# - Liger Kernel 0.6.4 (чистый Python/Triton)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    python3-distutils \
    cmake \
    build-essential \
    ninja-build \
    libgl1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv (быстрый pip, 10-100x быстрее)
RUN pip install uv

# venv для зависимостей через uv
ENV VENV_PATH=/opt/venv
RUN uv venv ${VENV_PATH} --python python3.10
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV VIRTUAL_ENV="${VENV_PATH}"

WORKDIR /app

# Сначала зависимости (для кеширования слоёв)
COPY requirements.txt /app/requirements.txt

# ============================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ ЧЕРЕЗ UV с кэшированием между сборками
# --mount=type=cache сохраняет скачанные пакеты на хосте
# ============================================================

# 1. PyTorch 2.5.1 из PyPI (bundled CUDA 12.4, совместим с flash-attn wheels)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# 2. Flash Attention 2.8.3 — PRE-BUILT WHEEL для torch 2.5 + cu12 + Python 3.10
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 3. Установка остальных зависимостей через uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/requirements.txt \
 && uv pip install deepspeed

# 4. Unsloth — ставим ПОСЛЕ основных пакетов с --no-deps
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
    python3.10 \
    python3.10-dev \
    python3-distutils \
    libgl1 \
    libgomp1 \
    # Для JIT компиляции DeepSpeed ops (cpu_adam и др.)
    build-essential \
    ninja-build \
    libaio-dev \
 && rm -rf /var/lib/apt/lists/*

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