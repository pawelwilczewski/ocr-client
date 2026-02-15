FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/root/.cache/huggingface
ENV UV_NO_CACHE=1
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin:/usr/local/cuda/bin:/app/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./

# Install dependencies first so this layer is reused unless lock/config changes.
RUN uv sync --python 3.12 --no-dev --no-install-project
RUN uv pip install psutil
RUN uv pip install flash-attn==2.7.3 --no-build-isolation

# Copy source later to avoid invalidating dependency cache on code-only changes.
COPY src ./src

# Install the local project/entrypoint without reinstalling dependencies.
RUN uv pip install --no-deps -e .

ENTRYPOINT ["ocr-client"]
