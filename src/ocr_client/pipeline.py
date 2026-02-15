from __future__ import annotations

import shutil
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


def _normalize_mmd_path(input_path: Path, output_mmd: Path | None) -> Path:
    if output_mmd is None:
        return input_path.with_suffix(".mmd")
    return output_mmd.with_suffix(".mmd")


def _format_page_block(page_number: int, body: str) -> str:
    normalized = body.replace("\r\n", "\n").rstrip()
    if not normalized:
        normalized = "[EMPTY PAGE]"
    return f"## Page {page_number}\n\n{normalized}\n"


def _render_pdf_pages(pdf_path: Path, pages_dir: Path, dpi: int = 200) -> list[Path]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pypdfium2 is required for PDF input. Install dependencies with uv sync."
        ) from exc

    pages_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    rendered_paths: list[Path] = []

    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Unable to read PDF: {pdf_path}") from exc

    try:
        page_count = len(document)
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        for index in range(page_count):
            page = document[index]
            output_image = pages_dir / f"page-{index + 1:04d}.png"
            try:
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil().convert("RGB")
                pil_image.save(output_image)
                rendered_paths.append(output_image)
            finally:
                page.close()
    finally:
        document.close()

    return rendered_paths


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
    output_path = _normalize_mmd_path(input_path, output_mmd)
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


def process_pdf(
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
    cleanup_temp_images: bool,
) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _normalize_mmd_path(input_path, output_mmd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"

    rendered_pages = _render_pdf_pages(input_path, pages_dir)
    bundle = load_model(model_name=model_name, device=device, cpu=cpu)

    page_blocks: list[str] = []
    failed_pages = 0
    for page_number, image_path in enumerate(rendered_pages, start=1):
        try:
            page_text = infer_image(
                bundle,
                image_file=image_path,
                output_path=output_dir,
                prompt=prompt,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
            )
        except Exception as exc:  # pragma: no cover - external inference behavior
            failed_pages += 1
            page_text = f"[OCR FAILED: {type(exc).__name__}: {exc}]"
        page_blocks.append(_format_page_block(page_number, page_text))

    final_text = "\n".join(page_blocks).replace("\r\n", "\n").rstrip() + "\n"
    output_path.write_text(final_text, encoding="utf-8")

    if cleanup_temp_images:
        shutil.rmtree(pages_dir, ignore_errors=True)

    return PipelineResult(
        status="ok",
        mode="pdf",
        output_mmd=output_path,
        message=(
            "OCR complete (pdf). "
            f"Pages={len(rendered_pages)}, failed={failed_pages}. "
            f"MMD saved to: {output_path}"
        ),
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
    cleanup_temp_images: bool = False,
) -> PipelineResult:
    input_kind = detect_input_kind(input_path)
    model_output_dir = output_dir if output_dir is not None else input_path.parent / "ocr_output"

    if input_kind == "pdf":
        return process_pdf(
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
            cleanup_temp_images=cleanup_temp_images,
        )
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
