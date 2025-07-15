# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow and Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
