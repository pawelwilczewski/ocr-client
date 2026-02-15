## ocr-client

Image OCR CLI using DeepSeek-OCR-2 (PDF support is intentionally deferred to a later chunk).

## Setup

Make sure you have CUDA installed: https://developer.nvidia.com/cuda-downloads.

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv python pin 3.12
uv sync
uv pip install -U pip setuptools wheel ninja packaging psutil
uv pip install flash-attn --no-build-isolation
```

Note:

- `flash-attn` requires a CUDA-capable environment with a valid CUDA toolchain (`nvcc`, `CUDA_HOME`).
- `pyproject.toml` configures `psutil` as an extra build dependency for `flash-attn` in uv.
- This chunk enforces strict `flash_attention_2` loading behavior.

## Run

```powershell
uv run ocr-client .\your-image.png
```

Optional arguments:

```powershell
uv run ocr-client .\your-image.png `
  --output-md .\your-image.md `
  --output-dir .\ocr_output `
  --device cuda:0 `
  --base-size 1024 `
  --image-size 768 `
  --crop-mode
```

CPU mode must be explicit:

```powershell
uv run ocr-client .\your-image.png --cpu
```

## Current limitations

- PDF input is rejected with: `PDF processing is scheduled for next chunk.`
- If CUDA or `flash-attn` is unavailable, execution fails fast with diagnostics.
