# Oneiro - Discord bot with embedded Diffusers image generation
FROM docker.io/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

WORKDIR /app

# Install uv from the official container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git for diffusers from source
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a virtual environment for isolation from the system Python
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first for caching (only rebuilds when pyproject.toml changes)
COPY pyproject.toml .
RUN mkdir -p src/oneiro \
    && touch src/oneiro/__init__.py \
    && uv pip install --no-cache .

# Copy source and reinstall without deps (fast rebuild on source changes)
COPY config.toml .
COPY src/ src/

RUN uv pip install --no-cache --no-deps .

# Environment configuration
ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/config/base.toml
ENV CONFIG_OVERLAY_PATH=/data/config.toml
# Set DIFFUSERS_GGUF_CUDA_KERNELS=true at runtime for ~10% speedup (requires PyTorch 2.7)

# Run the bot
CMD ["python", "-m", "oneiro"]
