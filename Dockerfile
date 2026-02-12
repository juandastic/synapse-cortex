# Production Dockerfile
# Uses tiangolo/uvicorn-gunicorn-fastapi for optimized production serving
# Gunicorn manages multiple Uvicorn workers, auto-scales to CPU cores

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY ./app /app/app

# Configure the application module
ENV MODULE_NAME="app.main"
ENV VARIABLE_NAME="app"

# Single worker so in-memory job store is shared across all requests
ENV WEB_CONCURRENCY=1
