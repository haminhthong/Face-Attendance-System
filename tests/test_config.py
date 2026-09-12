"""Unit tests cho module cấu hình hệ thống."""

from face_attendance.config import DEFAULT_CONFIG, RecognitionConfig


def test_recognition_config_defaults() -> None:
    cfg = RecognitionConfig()
    assert cfg.distance_threshold == 0.50
    assert cfg.identity_margin == 0.05
    assert cfg.top_k == 2
    assert cfg.min_confirmations == 3
    assert cfg.stable_duration_ms == 500
    assert cfg.eye_closed_threshold == 0.19
    assert cfg.eye_open_threshold == 0.23


def test_default_config_instance() -> None:
    assert DEFAULT_CONFIG.distance_threshold == 0.50
    assert DEFAULT_CONFIG.top_k == 2
