from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from ocr_client.cli import build_parser
from ocr_client.model import infer_image, load_model
from ocr_client.pipeline import detect_input_kind, run_pipeline


class BootstrapTests(unittest.TestCase):
    def test_cli_requires_input(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "ocr_client.cli"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_detects_pdf_and_image(self) -> None:
        pdf_path = Path("tests/fixtures/sample.pdf")
        image_path = Path("tests/fixtures/sample.png")

        self.assertEqual(detect_input_kind(pdf_path), "pdf")
        self.assertEqual(detect_input_kind(image_path), "image")

    def test_parser_accepts_expected_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["tests/fixtures/sample.png", "--device", "cuda:0", "--no-crop-mode"])
        self.assertEqual(args.input_path, "tests/fixtures/sample.png")
        self.assertEqual(args.device, "cuda:0")
        self.assertFalse(args.crop_mode)

    @patch("pathlib.Path.write_text")
    @patch("ocr_client.pipeline.infer_image", return_value="markdown body")
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_image_writes_default_mmd(
        self,
        mock_load_model: MagicMock,
        _mock_infer: MagicMock,
        mock_write_text: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()
        image_path = Path("tests/fixtures/sample.png")
        output_path = Path("tests/output/sample-output.mmd")

        result = run_pipeline(
            input_path=image_path,
            output_mmd=output_path,
            model_name="deepseek-ai/DeepSeek-OCR-2",
            prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            device="cuda:0",
        )

        self.assertEqual(result.mode, "image")
        self.assertEqual(result.output_mmd, output_path)
        mock_write_text.assert_called_once()

    def test_pipeline_pdf_deferred(self) -> None:
        with self.assertRaises(NotImplementedError):
            run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            )

    @patch("ocr_client.model.AutoModel.from_pretrained")
    @patch("ocr_client.model.AutoTokenizer.from_pretrained")
    @patch("ocr_client.model.torch.cuda.is_available", return_value=True)
    def test_load_model_uses_strict_flash_attention(
        self,
        _mock_cuda_available: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model_loader: MagicMock,
    ) -> None:
        model_instance = MagicMock()
        model_instance.eval.return_value = model_instance
        model_instance.cuda.return_value = model_instance
        model_instance.to.return_value = model_instance
        mock_model_loader.return_value = model_instance
        mock_tokenizer.return_value = MagicMock()

        bundle = load_model()

        self.assertEqual(bundle.device, "cuda:0")
        _, kwargs = mock_model_loader.call_args
        self.assertEqual(kwargs["_attn_implementation"], "flash_attention_2")
        self.assertEqual(kwargs["torch_dtype"], torch.bfloat16)

    def test_infer_image_raises_on_unexpected_shape(self) -> None:
        mock_model = MagicMock()
        mock_model.infer.return_value = {"unexpected": "value"}
        bundle = type("Bundle", (), {"tokenizer": MagicMock(), "model": mock_model})

        with self.assertRaises(ValueError):
            infer_image(
                bundle,  # type: ignore[arg-type]
                image_file=Path("tests/fixtures/sample.png"),
                output_path=Path("tests/fixtures"),
            )


if __name__ == "__main__":
    unittest.main()
