# Alternative Dockerfile without Conda - Using Python virtual environment
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    wget \
    git \
    curl \
    build-essential \
    ninja-build \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create Python virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch and related packages first (most time-consuming)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install xformers
RUN pip install --no-cache-dir -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn dependencies
RUN pip install --no-cache-dir misaki[en] ninja psutil packaging

# Install flash-attn with limited parallelism for GitHub Actions
RUN MAX_JOBS=2 pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install essential ML/Audio libraries
RUN pip install --no-cache-dir \
    huggingface-hub \
    transformers \
    librosa \
    soundfile \
    scipy \
    numpy \
    gradio \
    accelerate

# Copy requirements.txt and install additional dependencies if exists
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        echo "No requirements.txt found, skipping additional dependencies..."; \
    fi

# Create weights directory
RUN mkdir -p ./weights

# Set HF_HOME to avoid permission issues
ENV HF_HOME=/tmp/huggingface

# Download HuggingFace models with better error handling and retries
RUN echo "Downloading models..." && \
    for model in \
        "Wan-AI/Wan2.1-I2V-14B-480P" \
        "TencentGameMate/chinese-wav2vec2-base" \
        "hexgrad/Kokoro-82M" \
        "MeiGen-AI/MeiGen-MultiTalk"; do \
        echo "Downloading $model..."; \
        for attempt in 1 2 3; do \
            if huggingface-cli download "$model" --local-dir "./weights/$(basename "$model")" --resume-download; then \
                echo "Successfully downloaded $model"; \
                break; \
            else \
                echo "Download attempt $attempt failed for $model, retrying in 10s..."; \
                sleep 10; \
            fi; \
            if [ $attempt -eq 3 ]; then \
                echo "Failed to download $model after 3 attempts"; \
                exit 1; \
            fi; \
        done; \
    done

# Download specific file with revision
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors \
    --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base || \
    echo "Warning: Failed to download model.safetensors with revision, continuing..."

# Copy and setup model files
RUN if [ -f "weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json" ]; then \
        mv weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json \
           weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json_old; \
    fi && \
    if [ -f "weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json" ]; then \
        cp weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json \
           weights/Wan2.1-I2V-14B-480P/; \
    fi && \
    if [ -f "weights/MeiGen-MultiTalk/multitalk.safetensors" ]; then \
        cp weights/MeiGen-MultiTalk/multitalk.safetensors \
           weights/Wan2.1-I2V-14B-480P/; \
    fi

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command
CMD ["python", "app.py", "--num_persistent_param_in_dit", "0"]
