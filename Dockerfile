# Vertex AI Training Container for Bias Classification
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-vertex.txt .
RUN pip install --no-cache-dir -r requirements-vertex.txt

# Copy trainer package
COPY trainer/ ./trainer/
COPY setup.py .

# Install the trainer package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Entry point
ENTRYPOINT ["python", "-m", "trainer.task"]