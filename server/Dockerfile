# server/Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user
RUN groupadd -g 1001 autoveyra && \
    useradd -u 1001 -g autoveyra -m -s /bin/bash autoveyra

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app/server

# Create data directories and set permissions
RUN mkdir -p /app/server/data/models && \
    chown -R autoveyra:autoveyra /app/server

USER autoveyra

ENV PYTHONPATH=/app/server
ENV PYTHONUNBUFFERED=1
ENV AUTOVEYRA_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
