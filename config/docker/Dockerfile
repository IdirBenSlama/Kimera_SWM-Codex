# Dockerfile for KIMERA Production
# Phase 4, Weeks 14-15: Deployment Preparation

# --- Base Stage ---
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Builder Stage ---
FROM base as builder

# Copy requirements files
COPY requirements.txt /app/
COPY requirements/ /app/requirements/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM base as final

# Create a non-root user
RUN addgroup --system app && adduser --system --group app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Change ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Create required directories
RUN mkdir -p /app/data /app/logs

# Command to run the application
CMD ["python", "kimera.py"]
