# DeepSeek-OCR-2 PDF -> Markdown Console App Implementation Plan

## 1) Goal

Build a minimal Python console app that:

1. Accepts a PDF path as input.
2. Converts each page to an image.
3. Runs DeepSeek-OCR-2 per page.
4. Appends page OCR output into one Markdown file.
5. Labels each page section clearly (`## Page N`).

This plan is intentionally minimal and implementation-focused.

## 2) Non-Goals (for v1)

1. No web UI.
2. No parallel page OCR.
3. No advanced retry/backoff or distributed execution.
4. No rich document structure post-processing beyond page headers.

## 3) Technical Baseline (Pinned)

Use this exact baseline as requested:

1. `torch==2.6.0`
2. `transformers==4.46.3`
3. `tokenizers==0.20.3`
4. `einops`
5. `addict`
6. `easydict`
7. `flash-attn==2.7.3 --no-build-isolation`

Additional minimal dependency:

1. `pypdfium2` (render PDF pages to images)

Tooling:

1. `uv` for project and environment management.
2. `pyrefly` for type/static checks.

## 4) Environment Assumptions

1. NVIDIA GPU available.
2. CUDA-compatible environment for `torch==2.6.0` and FlashAttention2.
3. Linux or WSL2 strongly preferred for `flash-attn` build/install reliability.
4. Hugging Face model access is available in runtime environment.

If running on native Windows without working flash-attn, keep this fallback explicitly documented for local troubleshooting:

1. Use `_attn_implementation='eager'` temporarily for validation only.
2. Restore `flash_attention_2` as target production path.

## 5) Proposed Project Layout

```text
.
|- pyproject.toml
|- README.md
|- docs/
|  `- implementation-plan.md
|- src/
|  `- ocr_pdf/
|     |- __init__.py
|     |- cli.py
|     |- model.py
|     |- pdf_render.py
|     `- pipeline.py
`- tests/
   `- test_smoke.py
```

Rationale:

1. `cli.py`: argument parsing and entrypoint.
2. `model.py`: model/tokenizer loading and inference wrapper.
3. `pdf_render.py`: PDF-to-image conversion.
4. `pipeline.py`: page loop, markdown append, orchestration.

## 6) Implementation Phases

## Phase A: Project Bootstrap (`uv`)

1. Initialize project:
   - `uv init`
2. Ensure Python 3.12 environment:
   - `uv python pin 3.12`
3. Add runtime dependencies:
   - `uv add torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict pypdfium2`
4. Install `flash-attn` in the project env:
   - `uv pip install flash-attn==2.7.3 --no-build-isolation`
5. Add dev dependency for static checking:
   - `uv add --dev pyrefly`

Deliverable:

1. Reproducible `pyproject.toml` and lockfile with pinned baseline deps.

## Phase B: CLI Skeleton

Create `src/ocr_pdf/cli.py` with minimal args:

1. Positional: `pdf_path`
2. Optional: `--output-md`
3. Optional: `--output-dir` (for model artifacts/intermediate output)
4. Optional: `--device` (default `cuda:0`)
5. Optional: `--base-size` (default `1024`)
6. Optional: `--image-size` (default `768`)
7. Optional: `--crop-mode/--no-crop-mode` (default `true`)

Behavior:

1. If `--output-md` missing, derive from input file name in same directory:
   - `invoice.pdf` -> `invoice.md`
2. Create parent directories as needed.
3. Call pipeline function and return non-zero exit on fatal failure.

Deliverable:

1. Runnable command, e.g. `uv run ocr-pdf mydoc.pdf`.

## Phase C: Model Module (DeepSeek baseline preservation)

Create `src/ocr_pdf/model.py`:

1. Keep model id fixed by default:
   - `deepseek-ai/DeepSeek-OCR-2`
2. Preserve requested baseline loading logic:
   - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
   - `AutoModel.from_pretrained(..., _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)`
   - `model.eval().cuda().to(torch.bfloat16)`
3. Wrap in `load_model(model_name, device)` to centralize initialization.
4. Expose `infer_page(...)` wrapper that accepts one image path and output dir.

Prompt baseline for OCR:

1. `"<image>\n<|grounding|>Convert the document to markdown. "`

Deliverable:

1. Single importable model service usable by pipeline.

## Phase D: PDF Rendering Module

Create `src/ocr_pdf/pdf_render.py`:

1. Open PDF using `pypdfium2`.
2. Render each page to image (`.png`) in a temp directory.
3. Yield `(page_index, image_path)` for sequential processing.
4. Ensure deterministic ordering from page 1 to N.
5. Clean up temporary files after completion or interruption.

Rendering defaults:

1. Use consistent scale (e.g., 2.0) for OCR readability.
2. Use RGB output.

Deliverable:

1. Reliable iterator over page images for any multipage PDF.

## Phase E: Pipeline (Append Markdown per Page)

Create `src/ocr_pdf/pipeline.py`:

1. Load model/tokenizer once.
2. Loop over rendered pages.
3. For each page:
   - run `infer_page(...)`
   - extract markdown text result
   - append to target `.md` file
4. Append format:

```markdown
## Page 1

...ocr markdown...

## Page 2

...ocr markdown...
```

5. Ensure newline normalization (`\n`) and UTF-8 encoding.
6. On page failure:
   - append error note section for that page
   - continue with next page
7. Emit summary to console:
   - pages processed
   - pages failed
   - output markdown path

Deliverable:

1. End-to-end `pdf -> markdown` with page-labeled append behavior.

## Phase F: Entry Point Wiring

In `pyproject.toml`, add script entry:

1. `[project.scripts]`
2. `ocr-pdf = "ocr_pdf.cli:main"`

Deliverable:

1. Command available as `uv run ocr-pdf <file.pdf>`.

## Phase G: Static Checks and Smoke Tests

1. Initialize/configure `pyrefly` (minimal config).
2. Run static check:
   - `uv run pyrefly check`
3. Create `tests/test_smoke.py` with lightweight checks:
   - CLI arg parsing
   - output markdown path derivation
   - markdown append formatter
4. Manual smoke test:
   - `uv run ocr-pdf .\sample.pdf --output-md .\sample.md`
5. Verify:
   - `.md` exists
   - sections include `## Page N`
   - content appended in correct order

Deliverable:

1. Verified minimal working flow with static checks and smoke coverage.

## 7) Functional Spec (v1)

Inputs:

1. PDF file path (required).
2. Optional output markdown path.
3. Optional model output directory.

Outputs:

1. Single markdown file containing all page OCR results.
2. Console summary with page count and failures.

Rules:

1. Maintain source page order.
2. Label every page with 1-based index.
3. Continue on per-page errors.
4. Default execution target is GPU (`cuda:0`).
5. CPU execution is allowed only when explicitly requested by CLI option (for debugging/fallback).
6. Exit non-zero only when setup/load fails or no pages processed.

## 8) Error Handling Strategy

Fatal errors (stop process):

1. Model/tokenizer load failure.
2. Input PDF missing/unreadable.
3. No pages rendered.

Recoverable errors (continue):

1. One page OCR inference fails.
2. Intermediate file missing for specific page.

Logging:

1. Print concise per-page status line:
   - `Page 3/12: OK`
   - `Page 4/12: FAILED - <reason>`

## 9) Minimal Command Workflow

```powershell
uv init
uv python pin 3.12
uv add torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict pypdfium2
uv pip install flash-attn==2.7.3 --no-build-isolation
uv add --dev pyrefly
uv run pyrefly check
uv run ocr-pdf .\document.pdf --output-md .\document.md
```

## 10) Acceptance Criteria

1. Running `uv run ocr-pdf <pdf>` creates one `.md` output file.
2. Output contains page-labeled sections for all successfully processed pages.
3. Page order is preserved exactly.
4. Failures on individual pages do not abort remaining pages.
5. GPU is used by default (`cuda:0`) without requiring extra flags.
6. Baseline model load/infer settings match requested DeepSeek snippet (including FlashAttention2 path).

## 11) Next Increment (after v1)

1. Add optional JSON sidecar (`page -> status, elapsed_ms`).
2. Add `--start-page` and `--end-page`.
3. Add resumable mode that skips already-written pages.
4. Add structured logging and progress bar.
