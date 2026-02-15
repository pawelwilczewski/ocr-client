## ocr-client

Image OCR CLI using DeepSeek-OCR-2 (PDF support is intentionally deferred to a later chunk).

## Fresh Setup (WSL Ubuntu)

This is the reproducible flow from a clean WSL environment.

```bash
# 1) Base packages
sudo apt update
sudo apt install -y curl git build-essential ninja-build wget

# 2) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version

# 3) Add NVIDIA CUDA repo for WSL and install toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 4) Configure CUDA env vars
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5) Verify CUDA toolchain
nvcc --version
echo $CUDA_HOME

# 6) Create/sync environment (Python 3.12 baseline)
uv python install 3.12
uv sync --python 3.12

# 7) Install flash-attn manually (outside uv sync resolution)
uv pip install psutil
uv pip install flash-attn==2.7.3 --no-build-isolation
```

Note:

- Baseline dependency versions follow the model card recommendation:
  - `torch==2.6.0`
  - `transformers==4.46.3`
  - `tokenizers==0.20.3`
  - `einops`, `addict`, `easydict`
- Additional runtime dependencies required by DeepSeek-OCR-2 remote code:
  - `torchvision`
  - `pillow` (PIL)
- `flash-attn==2.7.3` requires Linux with CUDA toolchain (`nvcc`, `CUDA_HOME`) and may take time to build.
- `flash-attn` is intentionally installed manually with `uv pip ... --no-build-isolation` (not via `uv sync`).
- This chunk enforces strict `flash_attention_2` loading behavior.

## Run

```bash
uv run ocr-client ./your-image.png
```

Optional CUDA checks:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python -c "import flash_attn; print('flash_attn ok')"
```

Optional arguments:

```bash
uv run ocr-client ./your-image.png \
  --output-mmd ./your-image.mmd \
  --output-dir ./ocr_output \
  --device cuda:0 \
  --base-size 1024 \
  --image-size 768 \
  --crop-mode
```

CPU mode must be explicit:

```bash
uv run ocr-client ./your-image.png --cpu
```

## Current limitations

- PDF input is rejected with: `PDF processing is scheduled for next chunk.`
- If CUDA or `flash-attn` is unavailable, execution fails fast with diagnostics.
