from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from ocr_client.pipeline import detect_input_kind


class BootstrapTests(unittest.TestCase):
    def test_cli_hello_world(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "ocr_client.cli"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello world", result.stdout)

    def test_detects_pdf_and_image(self) -> None:
        pdf_path = Path("tests/fixtures/sample.pdf")
        image_path = Path("tests/fixtures/sample.png")

        self.assertEqual(detect_input_kind(pdf_path), "pdf")
        self.assertEqual(detect_input_kind(image_path), "image")

    def test_import_scaffold_modules(self) -> None:
        __import__("ocr_client.cli")
        __import__("ocr_client.model")
        __import__("ocr_client.pipeline")


if __name__ == "__main__":
    unittest.main()
