## ocr-client

Image/PDF OCR CLI using DeepSeek-OCR-2 with `.mmd` output and preserved OCR images.

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
uv run ocr-client ./your-image.png --output-dir ./ocr_output
```

Optional CUDA checks:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python -c "import flash_attn; print('flash_attn ok')"
```

Optional arguments:

```bash
uv run ocr-client ./your-image.png \
  --output-dir ./ocr_output \
  --device cuda:0 \
  --base-size 1024 \
  --image-size 768 \
  --crop-mode
```

PDF examples:

```bash
uv run ocr-client ./chapter.pdf --output-dir ./ocr_output
uv run ocr-client ./chapter.pdf --output-dir ./ocr_output --cleanup-temp-dir
```

Notes for PDF:

- PDF input is internally rendered page-by-page to images via `pypdfium2`.
- OCR appends page blocks live to one output file with `## Page N` headings.
- If one page fails OCR, processing continues and inserts a failure marker for that page.
- OCR images are normalized and preserved under `output-dir/images` with globally offset names.
- Per-page intermediate OCR artifacts are preserved under `output-dir/.page_tmp` for verification.
- Use `--cleanup-temp-dir` if you want to remove `output-dir/.page_tmp` after a run.
- Final output is always `output-dir/result.mmd`.
- Existing `output-dir/result.mmd` is overwritten at run start and rebuilt incrementally.

CPU mode must be explicit:

```bash
uv run ocr-client ./your-image.png --cpu
```

## Current limitations

- If CUDA or `flash-attn` is unavailable, execution fails fast with diagnostics.

## Docker

Build image:

```bash
docker build -t ocr-client:latest .
```

Run on GPU (mount current directory as `/work`):

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -v ocr-client-hf-cache:/root/.cache/huggingface \
  ocr-client:latest /work/chapter.pdf --output-dir /work/ocr_output
```

Run with cleanup flag:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -v ocr-client-hf-cache:/root/.cache/huggingface \
  ocr-client:latest /work/chapter.pdf --output-dir /work/ocr_output --cleanup-temp-dir
```

The named volume `ocr-client-hf-cache` preserves downloaded model weights between runs.
