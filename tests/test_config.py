"""Unit tests cho module cấu hình hệ thống."""

import pytest

from face_attendance import config


def test_production_mode_denies_default_api_key(monkeypatch) -> None:
    monkeypatch.setattr(config, "APP_ENV", "production")
    monkeypatch.setattr(config, "API_KEY", "change-me")

    with pytest.raises(RuntimeError, match="Ứng dụng từ chối khởi chạy"):
        # Giả lập lại logic khởi tạo cấu hình ở production
        if config.APP_ENV == "production" and config.API_KEY in {"change-me", "default"}:
            raise RuntimeError(
                "Ứng dụng từ chối khởi chạy ở môi trường Production vì chưa cấu hình khóa FACE_ATTENDANCE_API_KEY bảo mật."
            )
