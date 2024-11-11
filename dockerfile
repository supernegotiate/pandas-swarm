# Stage 1: Build environment
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install dependencies in a single layer
RUN pip install --user --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# Stage 2: Production environment
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies and set PATH in a single layer
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Container configuration
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Combine ENTRYPOINT and CMD
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]



