# Production Dockerfile
# Uses tiangolo/uvicorn-gunicorn-fastapi for optimized production serving
# Gunicorn manages multiple Uvicorn workers, auto-scales to CPU cores

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Install Node.js for npx (required by Notion MCP server)
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

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
