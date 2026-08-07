# syntax=docker/dockerfile:1

ARG LLAMA_CPP_BACKEND=vulkan

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
ARG LLAMA_CPP_BACKEND

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
    libvulkan-dev \
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

# 1. PyTorch 2.9.0 + cu128 — ФИКСИРОВАННАЯ ВЕРСИЯ.
# ВАЖНО: индекс cu128 обязателен. По умолчанию PyPI уже отдаёт cu130-сборки,
# которые требуют драйвер CUDA 13 (580+). На хостах с драйвером 12.8 (570.x)
# cu130 приводит к "CUDA driver too old" и torch.cuda.is_available()==False.
# flash-attn 2.8.3 wheel собран под torch 2.9 — поэтому версию не меняем.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0

# 2. Flash Attention 2.8.3 — PRE-BUILT WHEEL для torch 2.9 + cu12 + Python 3.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. Основные зависимости (без vllm — он отдельно)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/requirements.txt

# 4. vLLM — ТОЛЬКО с --no-deps, иначе он тянет torch 2.11+cu130 и transformers 5.x,
# ломая совместимость с драйвером и flash-attn. vLLM здесь опционален (инференс).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps vllm \
 || echo "Warning: vLLM installation failed, continuing without it"

# 5. llama.cpp Python bindings intentionally omitted.
# Agent Studio uses prebuilt external `llama-server`, which is downloaded at runtime.
# This avoids very long source builds of `llama-cpp-python`, especially for Vulkan.

# 6. DeepSpeed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install deepspeed

# 7. Unsloth — ставим ПОСЛЕ основных пакетов с --no-deps
# чтобы не перезаписывать уже установленные transformers/peft/trl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps unsloth unsloth-zoo \
 || echo "Warning: Unsloth installation failed, continuing without it"

# 8. СТРАХОВКА: жёстко возвращаем нужную сборку torch (cu128) и transformers.
# Любой из шагов выше (vLLM/DeepSpeed/Unsloth) мог случайно подтянуть torch 2.11+cu130
# или transformers 5.x. Переустанавливаем в самом конце, чтобы зафиксировать стек,
# совместимый с драйвером 12.8 и flash-attn 2.8.3.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --reinstall transformers==4.57.1

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
ARG LLAMA_CPP_BACKEND

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
    libegl1 \
    libxext6 \
    libx11-6 \
    libxcb1 \
    libgomp1 \
    libvulkan1 \
    vulkan-tools \
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
ENV LLAMA_CPP_BACKEND=${LLAMA_CPP_BACKEND}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MKL_THREADING_LAYER=GNU
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["./scripts/run_studio.sh"]
