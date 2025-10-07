# Use an official Python base image with version 3.8.12
FROM python:3.8.12-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app
COPY . /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "/app/extract_temperatures.py"]