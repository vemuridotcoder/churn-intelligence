FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching — rebuilds faster if only code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Expose API port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
