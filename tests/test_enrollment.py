"""Unit tests cho module đăng ký khuôn mặt và kiểm tra chất lượng ảnh."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from face_attendance.enrollment import (
    EnrollmentSample,
    check_identity_consistency,
    decode_and_validate_face,
    enroll_student,
)


class MockUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def create_dummy_face_image_bytes(val: int = 150) -> bytes:
    """Tạo ảnh dummy có độ phân giải đủ lớn và chi tiết cạnh để pass blur test."""
    img = np.full((300, 300, 3), val, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 0), 5)
    cv2.circle(img, (150, 150), 40, (255, 255, 255), -1)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def test_decode_and_validate_face_success() -> None:
    img_bytes = create_dummy_face_image_bytes()
    fake_locations = [(50, 250, 250, 50)]
    fake_encodings = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encodings),
    ):
        result = decode_and_validate_face(img_bytes)

    assert result.embedding.shape == (128,)
    assert result.face_width == 200
    assert result.face_height == 200
    assert result.blur_score >= 40.0


def test_identity_consistency_check() -> None:
    # 2 sample cùng 1 người (dist = 0.1) -> Pass
    sample1 = EnrollmentSample(np.zeros(128), "h1", 100.0, 120.0, 150, 150)
    sample2 = EnrollmentSample(np.full(128, np.sqrt(0.01 / 128)), "h2", 100.0, 120.0, 150, 150)
    check_identity_consistency([sample1, sample2], max_distance=0.50)

    # Sample thứ 3 của người khác (dist = 1.0 > 0.50) -> Phải raise ValueError
    sample3 = EnrollmentSample(np.ones(128), "h3", 100.0, 120.0, 150, 150)
    with pytest.raises(ValueError, match="không nhất quán danh tính"):
        check_identity_consistency([sample1, sample2, sample3], max_distance=0.50)


def test_enroll_student_requires_min_images(tmp_path, monkeypatch) -> None:
    from face_attendance import config, database

    test_db = tmp_path / "test_enroll.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    database.init_database()

    img_bytes = create_dummy_face_image_bytes()
    uploads = [MockUpload(f"face_{i}.png", img_bytes + bytes([i])) for i in range(4)]

    fake_locations = [(50, 250, 250, 50)]
    fake_encodings = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encodings),
    ):
        # 4 ảnh (< 5 ảnh yêu cầu) -> Bị từ chối
        with pytest.raises(ValueError, match="tối thiểu 5 ảnh"):
            enroll_student("SV001", "Nguyen Van A", "K23", uploads, consent_given=True)


def test_enroll_student_success_5_images(tmp_path, monkeypatch) -> None:
    from face_attendance import config, database

    test_db = tmp_path / "test_enroll_success.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    database.init_database()

    img_bytes = create_dummy_face_image_bytes()
    uploads = [MockUpload(f"face_{i}.png", img_bytes + bytes([i])) for i in range(5)]

    fake_locations = [(50, 250, 250, 50)]
    fake_encodings = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encodings),
    ):
        saved, warnings = enroll_student(
            "SV001", "Nguyen Van A", "K23", uploads, consent_given=True
        )

    assert saved == 5
    assert len(warnings) == 0

    student = database.get_student_by_code("SV001")
    assert student is not None
    assert student["full_name"] == "Nguyen Van A"

    embs = database.get_student_embeddings(int(student["id"]))
    assert len(embs) == 5
