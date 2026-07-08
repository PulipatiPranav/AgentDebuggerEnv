FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# torch + CUDA 12.1 + cuDNN 8 are already in the base image.
# requirements.txt installs only the remaining app-level deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Port 8000 is required by hackathon infrastructure
EXPOSE 8000

# Health check — hackathon automated ping requires this to return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker — environment is 2vCPU, multi-worker causes resource issues
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
