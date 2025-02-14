FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# COPY requirements.txt .

RUN pip install --upgrade pip && pip install flask face_recognition numpy 

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
