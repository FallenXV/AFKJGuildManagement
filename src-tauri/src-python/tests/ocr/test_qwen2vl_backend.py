"""Tests for the Qwen2-VL OCR backend helpers."""

import logging
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from adb_auto_player.ocr.qwen2vl_backend import QwenVLOCRBackend


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
