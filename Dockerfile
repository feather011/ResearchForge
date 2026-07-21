# ResearchForge — Docker Image
FROM python:3.12-slim

LABEL org.opencontainers.image.title="ResearchForge"
LABEL org.opencontainers.image.description="Multi-Mode LLM Research Agent"
LABEL org.opencontainers.image.version="1.0.0"

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8002

# Create non-root user
RUN addgroup --system --gid 1001 appuser \
    && adduser --system --uid 1001 --gid 1001 appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/checkpoints /app/data/traces /app/demo/outputs \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (leverage Docker cache)
COPY requirements.txt .
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# Copy application code (excludes .dockerignore patterns)
COPY . .

# Ensure permissions for non-root user
RUN chown -R appuser:appuser /app/data /app/demo/outputs

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8002/health', timeout=5)"

# Expose port
EXPOSE 8002

# Default command
CMD python -m uvicorn researchforge.service.app:app \
    --host 0.0.0.0 \
    --port ${PORT}
