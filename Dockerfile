# Use NVIDIA CUDA base image with CUDNN for PyTorch CUDA 12.1 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    build-essential \
    ninja-build \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.10
RUN conda create -n multitalk python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "multitalk", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/MeiGen-AI/MultiTalk.git . || \
    (echo "Repository cloning failed, continuing with manual setup..." && mkdir -p /app)

# Install PyTorch, torchvision, torchaudio with CUDA 12.1 support
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install xformers
RUN pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn dependencies
RUN pip install misaki[en] ninja psutil packaging

# Install flash-attn (this may take some time to compile)
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install huggingface-hub for model downloads
RUN pip install huggingface-hub

# Copy requirements.txt if it exists, otherwise create a minimal one
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; else echo "No requirements.txt found, skipping..."; fi

# Install librosa via conda
RUN conda install -c conda-forge librosa -y

# Create weights directory
RUN mkdir -p ./weights

# Download HuggingFace models
RUN echo "Downloading models..." && \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base && \
    huggingface-cli download hexgrad/Kokoro-82M --local-dir ./weights/Kokoro-82M && \
    huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk

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

# Copy application code (if not already cloned)
COPY . .

# Expose port (assuming Gradio default port 7860)
EXPOSE 7860

# Set the conda environment as default
ENV CONDA_DEFAULT_ENV=multitalk
ENV CONDA_PREFIX=/opt/conda/envs/multitalk

# Update PATH to use the conda environment
ENV PATH=/opt/conda/envs/multitalk/bin:$PATH

# Default command to run the application
CMD ["python", "app.py", "--num_persistent_param_in_dit", "0"]
