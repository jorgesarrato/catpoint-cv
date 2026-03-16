"""Tests for scripts/export_openvino.py"""

import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_fake_pt(tmp_path: Path) -> Path:
    pt = tmp_path / "best.pt"
    pt.write_text("fake")
    return pt


def _make_fake_openvino_dir(tmp_path: Path, stem: str = "best") -> Path:
    d = tmp_path / f"{stem}_openvino_model"
    d.mkdir(exist_ok=True)
    (d / "model.xml").write_text("<xml/>")
    (d / "model.bin").write_bytes(b"\x00" * 8)
    return d


def _make_fake_yaml(tmp_path: Path) -> Path:
    yaml = tmp_path / "dataset.yaml"
    yaml.write_text("nc: 2\nnames: [salo, taro]\n")
    return yaml


class TestExportOpenvino:
    def test_raises_if_model_not_found(self, tmp_path):
        import sys
        sys.argv = ["export_openvino.py", "--model", str(tmp_path / "missing.pt")]
        from scripts.export_openvino import main
        with pytest.raises(FileNotFoundError):
            main()

    def test_raises_if_not_pt_file(self, tmp_path):
        import sys
        bad = tmp_path / "model.onnx"
        bad.write_text("x")
        sys.argv = ["export_openvino.py", "--model", str(bad)]
        from scripts.export_openvino import main
        with pytest.raises(ValueError, match=".pt"):
            main()

    def test_raises_int8_without_data(self, tmp_path):
        import sys
        pt = _make_fake_pt(tmp_path)
        sys.argv = ["export_openvino.py", "--model", str(pt), "--int8"]
        from scripts.export_openvino import main
        with pytest.raises(ValueError, match="--data"):
            main()

    def test_fp32_output_copied(self, tmp_path):
        pt = _make_fake_pt(tmp_path)
        _make_fake_openvino_dir(tmp_path)
        output_dir = tmp_path / "models"

        import sys
        sys.argv = ["export_openvino.py", "--model", str(pt),
                    "--output", str(output_dir)]

        mock_model = MagicMock()
        with patch("scripts.export_openvino.YOLO", return_value=mock_model):
            from scripts.export_openvino import main
            main()

        dest = output_dir / "best_fp32_openvino_model"
        assert dest.exists()
        assert (dest / "model.xml").exists()

    def test_int8_output_copied(self, tmp_path):
        pt = _make_fake_pt(tmp_path)
        _make_fake_openvino_dir(tmp_path)
        output_dir = tmp_path / "models"
        yaml = _make_fake_yaml(tmp_path)

        import sys
        sys.argv = ["export_openvino.py", "--model", str(pt),
                    "--int8", "--data", str(yaml),
                    "--output", str(output_dir)]

        mock_model = MagicMock()
        with patch("scripts.export_openvino.YOLO", return_value=mock_model):
            from scripts.export_openvino import main
            main()

        dest = output_dir / "best_int8_openvino_model"
        assert dest.exists()

    def test_fp32_and_int8_dont_overwrite_each_other(self, tmp_path):
        output_dir = tmp_path / "models"
        yaml = _make_fake_yaml(tmp_path)
        mock_model = MagicMock()

        import sys

        # FP32 export
        pt = _make_fake_pt(tmp_path)
        _make_fake_openvino_dir(tmp_path)
        sys.argv = ["export_openvino.py", "--model", str(pt),
                    "--output", str(output_dir)]
        with patch("scripts.export_openvino.YOLO", return_value=mock_model):
            from scripts.export_openvino import main
            main()

        # INT8 export — recreate the source dir since it was moved
        _make_fake_openvino_dir(tmp_path)
        sys.argv = ["export_openvino.py", "--model", str(pt),
                    "--int8", "--data", str(yaml),
                    "--output", str(output_dir)]
        with patch("scripts.export_openvino.YOLO", return_value=mock_model):
            main()

        assert (output_dir / "best_fp32_openvino_model").exists()
        assert (output_dir / "best_int8_openvino_model").exists()

    def test_export_called_with_int8_params(self, tmp_path):
        pt = _make_fake_pt(tmp_path)
        _make_fake_openvino_dir(tmp_path)
        yaml = _make_fake_yaml(tmp_path)
        output_dir = tmp_path / "models"

        import sys
        sys.argv = ["export_openvino.py", "--model", str(pt),
                    "--int8", "--data", str(yaml),
                    "--output", str(output_dir)]

        mock_model = MagicMock()
        with patch("scripts.export_openvino.YOLO", return_value=mock_model):
            from scripts.export_openvino import main
            main()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["int8"] is True
        assert call_kwargs["data"] == str(yaml)
        assert call_kwargs["format"] == "openvino"
