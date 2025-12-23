# syntax=docker/dockerfile:1

############################
# 1) Builder stage
############################
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
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

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip install --no-cache-dir deepspeed

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
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

LABEL com.modelsathome.image="models-at-home-studio"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libgl1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Подхватываем venv из builder
ENV VENV_PATH=/opt/venv
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app
COPY --from=builder /app /app

# Директории под монтирования (не обязательно, но удобно)
RUN mkdir -p /app/datasets /app/out /app/.runs

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MKL_THREADING_LAYER=GNU
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["./scripts/run_studio.sh"]