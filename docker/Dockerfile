# Optimized Dockerfile for News Intelligence API
# Goal: Bare minimum image size for backend model inference

# Stage 1: Build & Install
FROM python:3.13-slim AS builder

WORKDIR /build

# Install system dependencies needed for building extensions (e.g., faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY docker/requirements-api.txt .

# Install dependencies into a separate directory to keep final image clean
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# Stage 2: Final Runtime Image
FROM python:3.13-slim

WORKDIR /app

# Install runtime system dependencies (faiss-cpu needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Do NOT copy code or models here; they are mounted via docker-compose
# This keeps the image size constant and minimal.

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Metadata
LABEL maintainer="Krishan Verma"
LABEL version="1.1"
LABEL description="Minimal Backend AI Engine for NewLens"

# Default command (can be overridden in docker-compose.yaml)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
