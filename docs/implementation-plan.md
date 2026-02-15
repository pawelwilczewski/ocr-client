# Implementation Plan (Updated): Image OCR Chunk

## Scope in this chunk

1. Project package/command is `ocr_client` / `ocr-client`.
2. Python target is `3.12` for DeepSeek integration.
3. Image OCR is implemented end-to-end.
4. PDF OCR is intentionally deferred.

## Baseline model reference (preserved semantics)

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=1024, image_size=768, crop_mode=True, save_results=True)
```

Implemented as structured functions in `src/ocr_client/model.py`.

## Runtime dependencies

1. `torch==2.6.0`
2. `transformers==4.46.3`
3. `tokenizers==0.20.3`
4. `einops`
5. `addict`
6. `easydict`
7. `flash-attn==2.7.3 --no-build-isolation` (required for strict mode)

## Behavior

1. `ocr-client <image>` writes markdown file by default at `<input_stem>.md`.
2. `--output-md` overrides markdown output path.
3. `--output-dir` controls model artifact output directory.
4. GPU is default (`--device cuda:0`).
5. CPU mode is opt-in only via `--cpu`.
6. Strict `_attn_implementation="flash_attention_2"` is enforced.
7. PDF input currently returns: `PDF processing is scheduled for next chunk.`

## Next chunk

1. Add PDF page rendering and per-page markdown append.
2. Add page numbering (`## Page N`) and multi-page aggregation.
3. Add richer progress/error reporting for multi-page documents.
