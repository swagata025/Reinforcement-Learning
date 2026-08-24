# Use official lightweight Python image
FROM python:3.10-slim

# Install system build dependencies required for Gymnasium Box2D (SWIG, OpenGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Default command runs the benchmark comparison report
CMD ["python", "main.py", "--compare"]
