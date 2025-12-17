FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Установка системных зависимостей
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Симлинк для python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Обновляем pip и устанавливаем PyTorch (пре-релиз или nightly, чтобы соответствовать твоей версии)
# Но для стабильности в Docker лучше использовать стабильную версию, если нет жесткой привязки к 2.9.1.
# Если нужна именно 2.9.1 (которая вероятно 2.6.0-nightly или опечатка, т.к. 2.9 еще очень далеко),
# то лучше поставить стабильный 2.4/2.5.
# Твоя версия `2.9.1` выглядит очень странно (может 2.1.2?), но допустим мы ставим свежий стейбл.

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir deepspeed

# Копируем исходный код
COPY . .

# Исправляем Windows-окончания строк в скриптах (CRLF -> LF)
RUN sed -i 's/\r$//' scripts/*.sh && chmod +x scripts/*.sh

# Устанавливаем наш пакет
RUN pip install -e .

# Создаем директории
RUN mkdir -p datasets out .runs

# Переменные окружения
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MKL_THREADING_LAYER=GNU

# Порт
EXPOSE 8501

CMD ["./scripts/run_studio.sh"]
