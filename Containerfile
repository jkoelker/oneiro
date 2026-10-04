# Oneiro - Discord bot with embedded Diffusers image generation
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv
FROM docker.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722

WORKDIR /app

# Install uv from the official container image
COPY --from=uv /uv /uvx /bin/

# Install dependencies first for caching (only rebuilds when pyproject.toml changes)
COPY pyproject.toml README.md LICENSE ./
RUN python -c 'from importlib.metadata import version; print("torch==" + version("torch"))' > /app/torch-constraints.txt \
    && mkdir -p src/oneiro \
    && touch src/oneiro/__init__.py \
    && uv pip install --system --no-cache --constraint /app/torch-constraints.txt .

# Copy source and reinstall without deps (fast rebuild on source changes)
COPY config.toml .
COPY src/ src/

RUN uv pip install --system --no-cache --no-deps .

# Environment configuration
ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/config/base.toml
ENV CONFIG_OVERLAY_PATH=/data/config.toml

# Run the bot
CMD ["python", "-m", "oneiro"]
