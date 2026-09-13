import pytest

from face_attendance.liveness import BlinkDetector, eye_aspect_ratio


def test_blink_requires_open_closed_open_cycle() -> None:
    checker = BlinkDetector(eye_closed_threshold=0.19, eye_open_threshold=0.23, ttl_seconds=10.0)
    assert not checker.update(1, 0.25)
    assert not checker.update(1, 0.15)
    assert checker.update(1, 0.25)
    checker.reset(1)
    assert not checker.update(1, 0.25)


def test_blink_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="Ngưỡng nhắm mắt"):
        BlinkDetector(eye_closed_threshold=0.3, eye_open_threshold=0.2, ttl_seconds=10.0)


def test_eye_aspect_ratio_calculation() -> None:
    # 6 điểm tạo mắt mở chuẩn
    pts = [(0, 10), (5, 15), (10, 15), (15, 10), (10, 5), (5, 5)]
    ear = eye_aspect_ratio(pts)
    assert ear is not None
    assert ear > 0.0

    # Dữ liệu không đủ 6 điểm
    assert eye_aspect_ratio([(0, 0), (1, 1)]) is None
