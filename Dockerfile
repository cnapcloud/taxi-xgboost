FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install system dependencies (git 추가!)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create non-root user and give ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "-u", "pipeline.py"]