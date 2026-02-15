from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

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

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


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


def _require_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        raise ValueError("--output-dir is required for both image and PDF inputs.")
    return output_dir


def _resolve_output_mmd(output_dir: Path) -> Path:
    return output_dir / "result.mmd"


def _extract_markdown_image_refs(text: str) -> list[str]:
    return [match.group(2) for match in MARKDOWN_IMAGE_RE.finditer(text)]


def _rewrite_and_preserve_images(
    block_text: str,
    src_root: Path,
    dest_images_dir: Path,
    next_index: int,
) -> tuple[str, int]:
    warnings: list[str] = []
    dest_images_dir.mkdir(parents=True, exist_ok=True)

    def _replacement(match: re.Match[str]) -> str:
        nonlocal next_index
        alt_text = match.group(1)
        raw_ref = match.group(2).strip()
        if "://" in raw_ref or raw_ref.startswith("data:"):
            return match.group(0)

        source_path = Path(raw_ref)
        if not source_path.is_absolute():
            source_path = src_root / source_path

        if not source_path.exists():
            warnings.append(f"[IMAGE WARNING: Missing image asset '{raw_ref}']")
            return match.group(0)

        suffix = source_path.suffix.lower() or ".jpg"
        dest_name = f"{next_index}{suffix}"
        dest_path = dest_images_dir / dest_name
        shutil.copy2(source_path, dest_path)
        next_index += 1
        return f"![{alt_text}](images/{dest_name})"

    rewritten = MARKDOWN_IMAGE_RE.sub(_replacement, block_text)
    if warnings:
        rewritten = rewritten.rstrip() + "\n\n" + "\n".join(warnings)
    return rewritten, next_index


def _append_block(out_file: TextIO, page_number: int | None, body: str, mode: InputKind) -> None:
    normalized = body.replace("\r\n", "\n").rstrip()
    if not normalized:
        normalized = "[EMPTY PAGE]"

    if mode == "pdf":
        out_file.write(f"## Page {page_number}\n\n")
    out_file.write(normalized)
    out_file.write("\n\n")
    out_file.flush()


def _read_page_markdown(temp_output_dir: Path, fallback_text: str) -> str:
    page_mmd = temp_output_dir / "result.mmd"
    if page_mmd.exists():
        return page_mmd.read_text(encoding="utf-8")
    return fallback_text


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
    output_dir: Path,
    model_name: str,
    prompt: str,
    device: str,
    cpu: bool,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    cleanup_temp_dir: bool,
) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _resolve_output_mmd(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    temp_root = output_dir / ".page_tmp"
    page_temp_output = temp_root / "image-0001"
    page_temp_output.mkdir(parents=True, exist_ok=True)

    bundle = load_model(model_name=model_name, device=device, cpu=cpu)
    markdown_text = infer_image(
        bundle,
        image_file=input_path,
        output_path=page_temp_output,
        prompt=prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    page_text = _read_page_markdown(page_temp_output, markdown_text)
    rewritten_text, next_image_index = _rewrite_and_preserve_images(
        page_text,
        src_root=page_temp_output,
        dest_images_dir=images_dir,
        next_index=0,
    )
    with output_path.open("w", encoding="utf-8") as out_file:
        _append_block(out_file, page_number=None, body=rewritten_text, mode="image")

    if cleanup_temp_dir:
        shutil.rmtree(temp_root, ignore_errors=True)

    return PipelineResult(
        status="ok",
        mode="image",
        output_mmd=output_path,
        message=f"OCR complete (image). Images preserved={next_image_index}. MMD saved to: {output_path}",
    )


def process_pdf(
    input_path: Path,
    *,
    output_dir: Path,
    model_name: str,
    prompt: str,
    device: str,
    cpu: bool,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    cleanup_temp_dir: bool,
) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _resolve_output_mmd(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    temp_root = output_dir / ".page_tmp"
    pages_dir = temp_root / "pages"

    rendered_pages = _render_pdf_pages(input_path, pages_dir)
    bundle = load_model(model_name=model_name, device=device, cpu=cpu)

    failed_pages = 0
    next_image_index = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for page_number, image_path in enumerate(rendered_pages, start=1):
            page_temp_output = temp_root / f"page-{page_number:04d}"
            page_temp_output.mkdir(parents=True, exist_ok=True)
            try:
                page_text = infer_image(
                    bundle,
                    image_file=image_path,
                    output_path=page_temp_output,
                    prompt=prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                )
                page_text = _read_page_markdown(page_temp_output, page_text)
                page_text, next_image_index = _rewrite_and_preserve_images(
                    page_text,
                    src_root=page_temp_output,
                    dest_images_dir=images_dir,
                    next_index=next_image_index,
                )
            except Exception as exc:  # pragma: no cover - external inference behavior
                failed_pages += 1
                page_text = f"[OCR FAILED: {type(exc).__name__}: {exc}]"

            _append_block(out_file, page_number=page_number, body=page_text, mode="pdf")

    if cleanup_temp_dir:
        shutil.rmtree(temp_root, ignore_errors=True)

    return PipelineResult(
        status="ok",
        mode="pdf",
        output_mmd=output_path,
        message=(
            "OCR complete (pdf). "
            f"Pages={len(rendered_pages)}, failed={failed_pages}. "
            f"Images preserved={next_image_index}. "
            f"MMD saved to: {output_path}"
        ),
    )


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    model_name: str,
    prompt: str,
    device: str = "cuda:0",
    cpu: bool = False,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
    cleanup_temp_dir: bool = False,
) -> PipelineResult:
    input_kind = detect_input_kind(input_path)
    model_output_dir = _require_output_dir(output_dir)

    if input_kind == "pdf":
        return process_pdf(
            input_path,
            output_dir=model_output_dir,
            model_name=model_name,
            prompt=prompt,
            device=device,
            cpu=cpu,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            cleanup_temp_dir=cleanup_temp_dir,
        )
    return process_image(
        input_path,
        output_dir=model_output_dir,
        model_name=model_name,
        prompt=prompt,
        device=device,
        cpu=cpu,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        cleanup_temp_dir=cleanup_temp_dir,
    )
