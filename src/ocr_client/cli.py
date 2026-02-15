from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocr_client.model import DEFAULT_MODEL_NAME, DEFAULT_PROMPT
from ocr_client.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-client")
    parser.add_argument("input_path", help="Path to a PDF or image file.")
    parser.add_argument("--output-mmd", help="Path for generated Mermaid Markdown output.")
    parser.add_argument(
        "--output-dir",
        help="Directory where model-side OCR artifacts are saved.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu", action="store_true", help="Explicitly run model on CPU.")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument(
        "--cleanup-temp-images",
        action="store_true",
        help="Delete rendered PDF page images after OCR completes.",
    )
    parser.add_argument(
        "--crop-mode",
        dest="crop_mode",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    output_mmd = Path(args.output_mmd) if args.output_mmd else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        result = run_pipeline(
            input_path=input_path,
            output_mmd=output_mmd,
            output_dir=output_dir,
            model_name=args.model_name,
            prompt=args.prompt,
            device=args.device,
            cpu=args.cpu,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode,
            cleanup_temp_images=args.cleanup_temp_images,
        )
    except (FileNotFoundError, ValueError, NotImplementedError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
