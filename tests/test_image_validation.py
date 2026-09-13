"""Unit tests kiểm tra chất lượng ảnh đầu vào và xử lý lỗi giải mã."""

import cv2
import numpy as np
import pytest

from face_attendance.config import MAX_UPLOAD_BYTES
from face_attendance.enrollment import decode_and_validate_face


def test_decode_empty_bytes() -> None:
    with pytest.raises(ValueError, match="File ảnh đang trống"):
        decode_and_validate_face(b"")


def test_decode_exceeds_max_bytes() -> None:
    large_payload = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="vượt quá giới hạn"):
        decode_and_validate_face(large_payload)


def test_decode_invalid_image_format() -> None:
    with pytest.raises(ValueError, match="không phải ảnh hợp lệ"):
        decode_and_validate_face(b"not an image binary payload")


def test_decode_blurry_image() -> None:
    # Tạo ảnh xám đồng nhất (blur score = 0)
    flat_image = np.full((300, 300, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", flat_image)
    with pytest.raises(ValueError, match="Ảnh quá mờ"):
        decode_and_validate_face(encoded.tobytes())


def test_decode_extreme_dark_image() -> None:
    # Tạo ảnh quá tối (mean brightness ~ 5)
    dark_image = np.full((300, 300, 3), 5, dtype=np.uint8)
    # Thêm nhiễu nhẹ để Laplacian var > MIN_BLUR_SCORE
    noise = np.random.randint(0, 10, (300, 300, 3), dtype=np.uint8)
    dark_image = cv2.add(dark_image, noise)
    _, encoded = cv2.imencode(".png", dark_image)
    with pytest.raises(ValueError, match="(Ánh sáng chưa phù hợp|Ảnh quá mờ)"):
        decode_and_validate_face(encoded.tobytes())
