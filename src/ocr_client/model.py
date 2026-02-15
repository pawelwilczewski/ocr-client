from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Any
    model: Any
    device: str
    dtype: str


def _parse_cuda_index(device: str) -> str:
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    raise ValueError(f"Unsupported CUDA device format: {device}")


def _extract_markdown(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in ("markdown", "text", "result", "content", "final_output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if isinstance(item, dict):
                    for key in ("markdown", "text", "result", "content"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
        keys = ", ".join(sorted(result.keys()))
        raise ValueError(f"Unexpected infer dict structure. Keys: [{keys}]")

    raise ValueError(f"Unexpected infer result type: {type(result).__name__}")


def _read_result_mmd(output_path: Path) -> str | None:
    mmd_path = output_path / "result.mmd"
    if not mmd_path.exists():
        return None
    text = mmd_path.read_text(encoding="utf-8").strip()
    return text if text else None


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda:0",
    *,
    cpu: bool = False,
) -> ModelBundle:
    if not cpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is unavailable for requested device '{device}'. "
                "Use --cpu to override, or install CUDA-enabled torch + drivers."
            )
        cuda_index = _parse_cuda_index(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index
    else:
        device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=None if cpu else torch.bfloat16,
        )
    except Exception as exc:  # pragma: no cover - external backend setup dependent
        cuda_state = torch.cuda.is_available()
        raise RuntimeError(
            "Failed to load model/tokenizer with strict flash_attention_2. "
            f"requested_device={device}, cuda_available={cuda_state}. "
            "This often means flash-attn/CUDA toolchain is unavailable."
        ) from exc

    if cpu:
        model = model.eval()
        return ModelBundle(tokenizer=tokenizer, model=model, device=device, dtype="float32")

    model = model.eval().cuda().to(torch.bfloat16)
    return ModelBundle(tokenizer=tokenizer, model=model, device=device, dtype="bfloat16")


def infer_image(
    bundle: ModelBundle,
    *,
    image_file: Path,
    output_path: Path,
    prompt: str = DEFAULT_PROMPT,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
) -> str:
    result = bundle.model.infer(
        bundle.tokenizer,
        prompt=prompt,
        image_file=str(image_file),
        output_path=str(output_path),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True,
    )
    try:
        return _extract_markdown(result)
    except ValueError:
        # DeepSeek OCR may emit markdown only to output_path/result.mmd.
        fallback = _read_result_mmd(output_path)
        if fallback is not None:
            return fallback
        raise
