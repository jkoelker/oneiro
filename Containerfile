# Oneiro - Discord bot with embedded Diffusers image generation
FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv
FROM docker.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722

WORKDIR /app

# Install uv from the official container image
COPY --from=uv /uv /uvx /bin/

# Install dependencies first for caching (only rebuilds when pyproject.toml changes)
COPY pyproject.toml README.md LICENSE ./
# This disposable image uses distro-managed Python; retain its existing Torch wheel.
# spin is unused development tooling whose Click cap conflicts with Hugging Face Hub.
RUN python -c 'from importlib.metadata import version; print("torch==" + version("torch"))' > /app/runtime-constraints.txt \
    && uv pip uninstall --system --break-system-packages spin \
    && mkdir -p src/oneiro \
    && touch src/oneiro/__init__.py \
    && uv pip install --system --break-system-packages --no-cache --constraint /app/runtime-constraints.txt .

# Copy source and reinstall without deps (fast rebuild on source changes)
COPY config.toml .
COPY src/ src/

# Discard the bootstrap wheel's build cache before installing the real package initializer.
RUN rm -rf /app/build && uv pip install --system --break-system-packages --no-cache --no-deps .

# Environment configuration
ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/config/base.toml
ENV CONFIG_OVERLAY_PATH=/data/config.toml

# Run the bot
CMD ["python", "-m", "oneiro"]
