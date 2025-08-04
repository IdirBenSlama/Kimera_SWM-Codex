# Dockerfile for KIMERA SWM Production
# Multi-stage build for optimized Docker image

# --- Base Stage ---
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# --- Builder Stage ---
FROM base as builder

# Copy requirements files
COPY requirements.txt /app/
COPY requirements/ /app/requirements/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional requirements from subdirectories
RUN for req_file in requirements/*.txt; do \
    if [ -f "$req_file" ]; then \
        pip install --no-cache-dir -r "$req_file" || true; \
    fi; \
    done

# --- Final Stage ---
FROM base as final

# Create a non-root user
RUN groupadd --system app && useradd --system --gid app app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Create required directories
RUN mkdir -p /app/data /app/logs /app/models /app/cache /app/tmp

# Set proper permissions
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "kimera.py"]
