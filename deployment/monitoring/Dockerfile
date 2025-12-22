# Standalone Zenith Metrics Server
# For docker deployment without full Zenith package

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn prometheus-client numpy

# Copy standalone server
COPY standalone_server.py .

EXPOSE 8080

CMD ["python", "standalone_server.py"]
