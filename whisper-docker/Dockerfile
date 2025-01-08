FROM python:3.9

# Установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
COPY transcribe.py .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создание директории для аудио файлов
RUN mkdir -p /app/audio

# Увеличиваем лимит памяти для Python
ENV PYTHONMEM=8G

CMD ["python", "transcribe.py"] 