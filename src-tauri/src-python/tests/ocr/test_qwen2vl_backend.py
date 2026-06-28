"""Unit tests for QwenVLOCRBackend.

Covers:
- _download_model_if_needed: partial-download detection via dual-file cache check
- _init_model: OSError from from_pretrained triggers force re-download and retry
- _prepare_image sizing and structured extractor parse rejection logging
"""

import logging
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest
from adb_auto_player.ocr.qwen2vl_backend import QwenVLOCRBackend


def _hf_mock(config_result, weights_result):
    """Return a minimal huggingface_hub module mock.

    Args:
        config_result: value returned for the config.json cache check.
        weights_result: value returned for the weight-index cache check.
    """
    mock = MagicMock()
    mock.try_to_load_from_cache.side_effect = [config_result, weights_result]
    return mock


class TestDownloadModelIfNeeded:
    def _backend(self):
        return QwenVLOCRBackend()

    def test_both_files_cached_skips_download(self):
        """Both config.json and weight index cached → snapshot_download not called."""
        backend = self._backend()
        mock_hf = _hf_mock("/cache/config.json", "/cache/index.json")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            backend._download_model_if_needed()

        mock_hf.snapshot_download.assert_not_called()

    def test_only_config_cached_triggers_download(self):
        """config.json cached but weight index missing → download triggered."""
        backend = self._backend()
        mock_hf = _hf_mock("/cache/config.json", None)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            backend._download_model_if_needed()

        mock_hf.snapshot_download.assert_called_once_with(QwenVLOCRBackend.MODEL_ID)

    def test_neither_file_cached_triggers_download(self):
        """Neither file cached → download triggered."""
        backend = self._backend()
        mock_hf = _hf_mock(None, None)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            backend._download_model_if_needed()

        mock_hf.snapshot_download.assert_called_once()

    def test_force_redownload_skips_cache_check(self):
        """force_redownload=True always downloads, even when both files are cached."""
        backend = self._backend()
        mock_hf = MagicMock()
        mock_hf.try_to_load_from_cache.return_value = "/cache/path"

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            backend._download_model_if_needed(force_redownload=True)

        mock_hf.snapshot_download.assert_called_once()
        mock_hf.try_to_load_from_cache.assert_not_called()


class TestInitModel:
    def _make_sys_mocks(self, proc_cls, model_cls):
        """Return sys.modules patches for torch and transformers.

        Args:
            proc_cls: mock to use as Qwen2VLProcessor.
            model_cls: mock to use as Qwen2VLForConditionalGeneration.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_transformers = MagicMock()
        mock_transformers.Qwen2VLProcessor = proc_cls
        mock_transformers.Qwen2VLForConditionalGeneration = model_cls

        return {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }

    def test_oserror_triggers_retry_with_force_redownload(self):
        """OSError from from_pretrained triggers force re-download."""
        backend = QwenVLOCRBackend()

        mock_proc_cls = MagicMock()
        mock_proc_cls.from_pretrained.side_effect = [
            OSError("Partial download: file not found"),
            MagicMock(),
        ]
        mock_model_inst = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_inst

        sys_mocks = self._make_sys_mocks(mock_proc_cls, mock_model_cls)

        with (
            patch.dict("sys.modules", sys_mocks),
            patch.object(
                type(backend),
                "_is_available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(backend, "_download_model_if_needed") as mock_dl,
        ):
            result = backend._init_model()

        assert result is True
        assert mock_dl.call_count == 2
        assert mock_dl.call_args_list[0] == call()
        assert mock_dl.call_args_list[1] == call(force_redownload=True)
        mock_model_inst.eval.assert_called_once()

    def test_persistent_oserror_sets_model_load_failed(self):
        """If the retry also raises OSError, model_load_failed is set to True."""
        backend = QwenVLOCRBackend()

        mock_proc_cls = MagicMock()
        mock_proc_cls.from_pretrained.side_effect = OSError("persistent error")
        mock_model_cls = MagicMock()

        sys_mocks = self._make_sys_mocks(mock_proc_cls, mock_model_cls)

        with (
            patch.dict("sys.modules", sys_mocks),
            patch.object(
                type(backend),
                "_is_available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(backend, "_download_model_if_needed"),
        ):
            result = backend._init_model()

        assert result is False
        assert backend._model_load_failed is True


def test_prepare_image_honors_per_call_max_width() -> None:
    backend = QwenVLOCRBackend()
    screenshot = np.zeros((100, 2000, 3), dtype=np.uint8)

    default_image = backend._prepare_image(screenshot)
    full_frame_image = backend._prepare_image(
        screenshot, max_width=backend.FULL_FRAME_MAX_IMAGE_WIDTH_CAP
    )
    uncapped_image = backend._prepare_image(screenshot, max_width=None)

    assert default_image.size == (backend.MAX_IMAGE_WIDTH_CAP, 27)
    assert full_frame_image.size == (backend.FULL_FRAME_MAX_IMAGE_WIDTH_CAP, 54)
    assert uncapped_image.size == (2000, 100)


@pytest.mark.parametrize(
    ("method_name", "raw_output", "expected_log"),
    [
        (
            "extract_activeness_from_screenshot",
            None,
            "Qwen2-VL activeness parse rejected: empty response.",
        ),
        (
            "extract_chest_from_screenshot",
            "not json",
            "Qwen2-VL chest parse rejected: no JSON array in output:",
        ),
        (
            "extract_rankings_from_screenshot",
            '[{"rank":"1","name":"Player","score":"100M",}]',
            "Qwen2-VL rankings parse rejected: JSON decode failed:",
        ),
    ],
)
def test_structured_extractors_log_parse_rejections(
    method_name: str,
    raw_output: str | None,
    expected_log: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = QwenVLOCRBackend()
    screenshot = np.zeros((32, 32, 3), dtype=np.uint8)

    with (
        patch.object(
            QwenVLOCRBackend,
            "_is_available",
            new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(backend, "_init_model", return_value=True),
        patch.object(backend, "_prepare_image", return_value=object()),
        patch.object(backend, "_run_qwen_inference", return_value=raw_output),
        caplog.at_level(logging.DEBUG),
    ):
        result = getattr(backend, method_name)(screenshot)

    assert result is None
    assert expected_log in caplog.text
