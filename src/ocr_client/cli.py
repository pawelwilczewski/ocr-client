from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocr_client.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-client")
    parser.add_argument("input_path", nargs="?", help="Path to a PDF or image file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_path) if args.input_path else None

    try:
        output = run_pipeline(input_path=input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
