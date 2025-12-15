FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Установка Python пакетов
# Сначала устанавливаем основные, потом DeepSpeed (чтобы он видел torch)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir deepspeed

# Копируем исходный код
COPY . .

# Устанавливаем наш пакет в режиме editable
RUN pip install -e .

# Создаем директории для маунтинга
RUN mkdir -p datasets out .runs

# Переменные окружения для корректной работы
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Порт Streamlit
EXPOSE 8501

# Команда запуска
CMD ["./scripts/run_studio.sh"]

