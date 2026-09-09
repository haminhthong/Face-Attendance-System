"""Unit tests cho cơ chế phát hiện ảnh gần trùng (near-duplicate) bằng Perceptual Hash (pHash)."""

import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

# Đảm bảo face_recognition có trong sys.modules để mock trên môi trường không có dlib binary
if "face_recognition" not in sys.modules:
    sys.modules["face_recognition"] = MagicMock()

from tools import prepare_dataset

from face_attendance.recognition import decode_and_validate_face, enroll_student_images


class MockUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def create_dummy_face_image_bytes(val: int = 150) -> bytes:
    """Tạo ảnh dummy có độ phân giải đủ lớn và có chi tiết cạnh để pass blur test."""
    img = np.full((300, 300, 3), val, dtype=np.uint8)
    # Thêm các vệt nét để Laplacian variance > 40
    cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 0), 5)
    cv2.circle(img, (150, 150), 40, (255, 255, 255), -1)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def test_phash_present_in_enrollment_result() -> None:
    img_bytes = create_dummy_face_image_bytes()
    fake_locations = [(50, 250, 250, 50)]
    fake_encodings = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encodings),
    ):
        result = decode_and_validate_face(img_bytes)

    assert result.phash is not None
    assert len(result.phash) > 0


def test_enroll_student_images_filters_near_duplicates() -> None:
    img_bytes1 = create_dummy_face_image_bytes(150)
    img_bytes2 = img_bytes1  # Cùng hash

    up1 = MockUpload("face_1.png", img_bytes1)
    up2 = MockUpload("face_2.png", img_bytes2)

    fake_locations = [(50, 250, 250, 50)]
    fake_encodings = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encodings),
        patch("face_attendance.recognition.upsert_student", return_value={"id": 1}),
        patch("face_attendance.recognition.save_embedding", return_value=True),
    ):
        saved, warnings = enroll_student_images(
            "SV001", "Nguyễn Văn A", "K23", [up1, up2], consent_given=True
        )

    # Chỉ 1 ảnh được lưu, ảnh thứ 2 bị bỏ qua do trùng byte / near-duplicate
    assert saved == 1
    assert any("Bỏ qua vì trùng" in w or "near-duplicate" in w for w in warnings)


def test_check_near_duplicates_phash_tool(tmp_path, monkeypatch) -> None:
    # Setup thư mục giả lập
    test_data_dir = tmp_path / "data"
    test_enrollment = test_data_dir / "private" / "enrollment"
    test_validation = test_data_dir / "private" / "validation"
    test_test = test_data_dir / "private" / "test"

    for d in [test_enrollment, test_validation, test_test]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(prepare_dataset, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(prepare_dataset, "ENROLLMENT_DIR", test_enrollment)
    monkeypatch.setattr(prepare_dataset, "VALIDATION_DIR", test_validation)
    monkeypatch.setattr(prepare_dataset, "TEST_DIR", test_test)

    # Tạo 2 ảnh y hệt nhau ở enrollment và validation
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    img.save(test_enrollment / "person1.png")
    img.save(test_validation / "person1_leak.png")

    dups = prepare_dataset.check_near_duplicates_phash(threshold=2)
    assert len(dups) == 1
    pair_key = list(dups.keys())[0]
    assert "person1.png" in pair_key and "person1_leak.png" in pair_key
