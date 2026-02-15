from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocr_client.model import infer_image, load_model

InputKind = Literal["pdf", "image"]

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class PipelineResult:
    status: Literal["ok"]
    mode: InputKind
    output_mmd: Path
    message: str


def detect_input_kind(input_path: Path) -> InputKind:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ValueError(
        f"Unsupported input type for '{input_path}'. Supported: .pdf and common image extensions."
    )


def process_pdf(_input_path: Path) -> PipelineResult:
    raise NotImplementedError("PDF processing is scheduled for next chunk.")


def process_image(
    input_path: Path,
    *,
    output_mmd: Path | None,
    output_dir: Path,
    model_name: str,
    prompt: str,
    device: str,
    cpu: bool,
    base_size: int,
    image_size: int,
    crop_mode: bool,
) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_mmd is None:
        output_path = input_path.with_suffix(".mmd")
    else:
        output_path = output_mmd.with_suffix(".mmd")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_model(model_name=model_name, device=device, cpu=cpu)
    markdown_text = infer_image(
        bundle,
        image_file=input_path,
        output_path=output_dir,
        prompt=prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    normalized = markdown_text.replace("\r\n", "\n").rstrip() + "\n"
    output_path.write_text(normalized, encoding="utf-8")
    return PipelineResult(
        status="ok",
        mode="image",
        output_mmd=output_path,
        message=f"OCR complete (image). MMD saved to: {output_path}",
    )


def run_pipeline(
    *,
    input_path: Path,
    output_mmd: Path | None = None,
    output_dir: Path | None = None,
    model_name: str,
    prompt: str,
    device: str = "cuda:0",
    cpu: bool = False,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
) -> PipelineResult:
    input_kind = detect_input_kind(input_path)
    model_output_dir = output_dir if output_dir is not None else input_path.parent / "ocr_output"

    if input_kind == "pdf":
        return process_pdf(input_path)
    return process_image(
        input_path,
        output_mmd=output_mmd,
        output_dir=model_output_dir,
        model_name=model_name,
        prompt=prompt,
        device=device,
        cpu=cpu,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
