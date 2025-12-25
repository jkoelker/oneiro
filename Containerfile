# Oneiro - Discord bot with embedded Diffusers image generation
FROM docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install git for diffusers from source
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install dependencies first for caching (only rebuilds when pyproject.toml changes)
COPY pyproject.toml .
RUN mkdir -p src/oneiro \
    && touch src/oneiro/__init__.py \
    && pip install --no-cache-dir .

# Copy source and reinstall without deps (fast rebuild on source changes)
COPY config.toml .
COPY src/ src/

RUN pip install --no-cache-dir --no-deps .

# Environment configuration
ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/config/base.toml
ENV CONFIG_OVERLAY_PATH=/data/config.toml
# Set DIFFUSERS_GGUF_CUDA_KERNELS=true at runtime for ~10% speedup (requires PyTorch 2.7)

# Run the bot
CMD ["python", "-m", "oneiro"]
