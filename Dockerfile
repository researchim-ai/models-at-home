# syntax=docker/dockerfile:1

############################
# 1) Builder stage
############################
# CUDA 12.4 для совместимости с torch 2.7+ и flash-attn 2.8+
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

# venv для зависимостей (удобно копировать в финальный образ)
ENV VENV_PATH=/opt/venv
RUN python3.10 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

# Сначала зависимости (для кеширования слоёв)
COPY requirements.txt /app/requirements.txt

# Установка основных зависимостей
# torch 2.7 из PyPI (bundled CUDA 12.4) + flash-attn 2.8.3
# НЕ используем --index-url, torch из PyPI идёт с встроенным CUDA
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip install --no-cache-dir deepspeed

# Flash Attention 2.8.3 — pre-built wheel для cu12 + torch 2.7 + Python 3.10
# Wheels: https://github.com/Dao-AILab/flash-attention/releases
ARG INSTALL_FLASH_ATTN=true
RUN if [ "$INSTALL_FLASH_ATTN" = "true" ]; then \
      echo "Установка flash-attn 2.8.3 из pre-built wheel (cu12, torch 2.7)..." && \
      python -m pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
      || ( \
        echo "Pre-built wheel не подошёл, компилируем..." && \
        export MAX_JOBS=4 && \
        export CUDA_HOME=/usr/local/cuda && \
        python -m pip install --no-cache-dir flash-attn --no-build-isolation \
      ); \
    else \
      echo "Пропуск установки flash-attn (INSTALL_FLASH_ATTN=false)"; \
    fi

# Теперь код (ВАЖНО: .dockerignore должен исключать datasets/out/.runs и т.п.)
COPY . /app

# Нормализуем sh-скрипты, но не падаем, если папки нет
RUN if [ -d scripts ]; then \
      find scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} \; -exec chmod +x {} \;; \
    fi

# Установим проект (можно оставить editable, но обычно достаточно обычной установки)
RUN python -m pip install --no-cache-dir -e .

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