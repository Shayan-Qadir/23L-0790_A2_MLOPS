# Stage 1: Official lightweight Python slim base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production \
    FLASK_DEBUG=0

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for container security (Sub-task 6.2)
RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 -m -s /sbin/nologin appuser

WORKDIR /app

# Layer Caching Optimization:
# Copy dependency manifests first to leverage Docker layer cache.
# Docker re-executes pip install ONLY when requirements change.
COPY requirements_docker.txt ./requirements.txt

# Install dependencies using CPU-only TensorFlow without caching wheels in image layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ ./src
COPY models/ ./models
COPY .env.example ./.env.example

# Set up storage directory and transfer ownership to non-root appuser
RUN mkdir -p /app/instance && chown -R appuser:appgroup /app

# Switch execution context to non-root user
USER appuser

# Expose web application port
EXPOSE 5000

# Container Health Check (Sub-task 6.2)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start Flask production server
CMD ["python", "src/app/app.py"]

