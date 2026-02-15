from __future__ import annotations

import subprocess
import sys
import tempfile
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
        args = parser.parse_args(
            [
                "tests/fixtures/sample.png",
                "--output-dir",
                "tests/output",
                "--device",
                "cuda:0",
                "--no-crop-mode",
                "--cleanup-temp-dir",
            ]
        )
        self.assertEqual(args.input_path, "tests/fixtures/sample.png")
        self.assertEqual(args.output_dir, "tests/output")
        self.assertEqual(args.device, "cuda:0")
        self.assertFalse(args.crop_mode)
        self.assertTrue(args.cleanup_temp_dir)

    @patch("ocr_client.pipeline.infer_image")
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_image_writes_default_mmd_and_preserves_images(
        self,
        mock_load_model: MagicMock,
        mock_infer: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()

        def infer_side_effect(*args: object, **kwargs: object) -> str:
            output_path = kwargs["output_path"]
            assert isinstance(output_path, Path)
            (output_path / "images").mkdir(parents=True, exist_ok=True)
            (output_path / "images" / "0.jpg").write_bytes(b"test")
            (output_path / "result.mmd").write_text("hello\n\n![](images/0.jpg)\n", encoding="utf-8")
            return "ignored"

        mock_infer.side_effect = infer_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = run_pipeline(
                input_path=Path("tests/fixtures/sample.png"),
                output_dir=output_dir,
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
                device="cuda:0",
            )

            self.assertEqual(result.mode, "image")
            self.assertEqual(result.output_mmd, output_dir / "result.mmd")
            self.assertTrue((output_dir / "images" / "0.jpg").exists())
            out_text = (output_dir / "result.mmd").read_text(encoding="utf-8")
            self.assertIn("![](images/0.jpg)", out_text)

    @patch("ocr_client.pipeline.infer_image")
    @patch("ocr_client.pipeline._render_pdf_pages")
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_pdf_happy_path_with_page_headers(
        self,
        mock_load_model: MagicMock,
        mock_render_pdf_pages: MagicMock,
        mock_infer_image: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()
        mock_render_pdf_pages.return_value = [
            Path("ocr_output/pages/page-0001.png"),
            Path("ocr_output/pages/page-0002.png"),
            Path("ocr_output/pages/page-0003.png"),
        ]
        mock_infer_image.side_effect = ["first", "second", "third"]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                output_dir=output_dir,
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            )

            self.assertEqual(result.mode, "pdf")
            self.assertEqual(result.output_mmd, output_dir / "result.mmd")
            self.assertIn("Pages=3, failed=0", result.message)

            written_text = (output_dir / "result.mmd").read_text(encoding="utf-8")
            self.assertIn("## Page 1", written_text)
            self.assertIn("## Page 2", written_text)
            self.assertIn("## Page 3", written_text)
            self.assertLess(written_text.index("## Page 1"), written_text.index("## Page 2"))
            self.assertLess(written_text.index("## Page 2"), written_text.index("## Page 3"))

    @patch("ocr_client.pipeline.infer_image")
    @patch("ocr_client.pipeline._render_pdf_pages")
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_pdf_rewrites_and_offsets_image_refs(
        self,
        mock_load_model: MagicMock,
        mock_render_pdf_pages: MagicMock,
        mock_infer_image: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()
        mock_render_pdf_pages.return_value = [Path("page1.png"), Path("page2.png")]
        call_count = {"value": 0}

        def infer_side_effect(*args: object, **kwargs: object) -> str:
            output_path = kwargs["output_path"]
            assert isinstance(output_path, Path)
            call_count["value"] += 1
            (output_path / "images").mkdir(parents=True, exist_ok=True)
            if call_count["value"] == 1:
                (output_path / "images" / "0.jpg").write_bytes(b"0")
                (output_path / "images" / "1.jpg").write_bytes(b"1")
                (output_path / "result.mmd").write_text(
                    "P1\n\n![](images/0.jpg)\n![](images/1.jpg)\n",
                    encoding="utf-8",
                )
            else:
                (output_path / "images" / "0.jpg").write_bytes(b"2")
                (output_path / "result.mmd").write_text("P2\n\n![](images/0.jpg)\n", encoding="utf-8")
            return "ignored"

        mock_infer_image.side_effect = infer_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                output_dir=output_dir,
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            )

            written_text = (output_dir / "result.mmd").read_text(encoding="utf-8")
            self.assertIn("![](images/0.jpg)", written_text)
            self.assertIn("![](images/1.jpg)", written_text)
            self.assertIn("![](images/2.jpg)", written_text)
            self.assertTrue((output_dir / "images" / "0.jpg").exists())
            self.assertTrue((output_dir / "images" / "1.jpg").exists())
            self.assertTrue((output_dir / "images" / "2.jpg").exists())

    @patch("ocr_client.pipeline.infer_image")
    @patch("ocr_client.pipeline._render_pdf_pages")
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_pdf_continues_on_page_failure(
        self,
        mock_load_model: MagicMock,
        mock_render_pdf_pages: MagicMock,
        mock_infer_image: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()
        mock_render_pdf_pages.return_value = [
            Path("ocr_output/pages/page-0001.png"),
            Path("ocr_output/pages/page-0002.png"),
            Path("ocr_output/pages/page-0003.png"),
        ]
        mock_infer_image.side_effect = ["first", RuntimeError("boom"), "third"]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                output_dir=output_dir,
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            )

            self.assertEqual(result.mode, "pdf")
            self.assertIn("failed=1", result.message)
            written_text = (output_dir / "result.mmd").read_text(encoding="utf-8")
            self.assertIn("## Page 2", written_text)
            self.assertIn("[OCR FAILED: RuntimeError: boom]", written_text)
            self.assertIn("## Page 3", written_text)

    @patch("ocr_client.pipeline.shutil.rmtree")
    @patch("ocr_client.pipeline.infer_image", return_value="only-page")
    @patch("ocr_client.pipeline._render_pdf_pages", return_value=[Path("ocr_output/pages/page-0001.png")])
    @patch("ocr_client.pipeline.load_model")
    def test_pipeline_pdf_cleanup_temp_dir(
        self,
        mock_load_model: MagicMock,
        _mock_render: MagicMock,
        _mock_infer: MagicMock,
        mock_rmtree: MagicMock,
    ) -> None:
        mock_load_model.return_value = MagicMock()

        run_pipeline(
            input_path=Path("tests/fixtures/sample.pdf"),
            output_dir=Path("tests/output"),
            model_name="deepseek-ai/DeepSeek-OCR-2",
            prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            cleanup_temp_dir=True,
        )

        called_dirs = [call.args[0] for call in mock_rmtree.call_args_list]
        self.assertIn(Path("tests/output") / ".page_tmp", called_dirs)

    @patch("ocr_client.pipeline._render_pdf_pages", side_effect=ValueError("bad pdf"))
    def test_pipeline_pdf_unreadable_raises(self, _mock_render: MagicMock) -> None:
        with self.assertRaises(ValueError):
            run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                output_dir=Path("tests/output"),
                model_name="deepseek-ai/DeepSeek-OCR-2",
                prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            )

    def test_pipeline_requires_output_dir(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline(
                input_path=Path("tests/fixtures/sample.pdf"),
                output_dir=None,
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
