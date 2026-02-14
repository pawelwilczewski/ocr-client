from __future__ import annotations

from pathlib import Path
from typing import Literal

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


def process_pdf(_input_path: Path) -> str:
    return "hello world"


def process_image(_input_path: Path) -> str:
    return "hello world"


def run_pipeline(input_path: Path | None = None) -> str:
    if input_path is None:
        return "hello world"

    input_kind = detect_input_kind(input_path)
    if input_kind == "pdf":
        return process_pdf(input_path)
    return process_image(input_path)
