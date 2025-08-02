# Dockerfile для эксперимента по сравнению производительности Pandas и Polars
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий для данных и результатов
RUN mkdir -p /app/data/generated \
    && mkdir -p /app/results/benchmarks \
    && mkdir -p /app/results/reports \
    && mkdir -p /app/logs

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда по умолчанию
CMD ["python", "-m", "src.main"]