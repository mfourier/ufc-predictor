# Dockerfile

# Lightweight base image with Python 3.11
FROM python:3.11-slim

# Disable interactive prompts and enable real-time logging
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Create and set the working directory inside the container
WORKDIR /ufc-predictor

# Copy the entire project into the container
COPY . .

# Install system-level dependencies required by numpy, matplotlib, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Default command: run the CLI app
CMD ["python", "app.py"]

