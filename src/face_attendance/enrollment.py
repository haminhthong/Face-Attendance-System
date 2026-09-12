"""Module đăng ký và kiểm tra chất lượng ảnh khuôn mặt sinh viên (Enrollment Pipeline).

Chịu trách nhiệm:
1. Giải mã và kiểm tra chất lượng ảnh (độ mờ Laplacian, độ sáng, kích thước khuôn mặt tối thiểu).
2. Kiểm tra Single-Face Gate trên từng ảnh đăng ký.
3. Trích xuất vector đặc trưng 128D (dlib ResNet).
4. Kiểm tra tính nhất quán danh tính (Identity Consistency) giữa các ảnh để tránh trộn người.
5. Lưu trữ sinh viên và vector đặc trưng vào SQLite.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from .config import (
    MAX_BRIGHTNESS,
    MAX_UPLOAD_BYTES,
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MIN_ENROLLMENT_IMAGES,
    MIN_FACE_SIZE_PX,
)
from .database import save_embedding, upsert_student

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentSample:
    """Kết quả giải mã và kiểm tra chất lượng của một ảnh khuôn mặt."""

    embedding: np.ndarray
    image_hash: str
    blur_score: float
    brightness: float
    face_width: int
    face_height: int


def decode_and_validate_face(image_bytes: bytes) -> EnrollmentSample:
    """Giải mã file ảnh, kiểm tra chất lượng và trích xuất vector khuôn mặt 128D.

    Các cổng kiểm tra chất lượng:
    - Kích thước file không rỗng và <= MAX_UPLOAD_BYTES (8 MB).
    - Độ mờ ảnh: Laplacian variance >= MIN_BLUR_SCORE (40.0).
    - Độ sáng ảnh: trung bình thang xám trong khoảng 40.0 - 220.0.
    - Duy nhất 1 khuôn mặt trong ảnh.
    - Kích thước khuôn mặt tối thiểu 100 x 100 px.
    """
    if not image_bytes:
        raise ValueError("File ảnh đang trống.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn dung lượng 8 MB.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Nội dung file không phải ảnh hợp lệ.")

    # 1. Kiểm tra độ sắc nét bằng biến thiên Laplacian
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    if blur_score < MIN_BLUR_SCORE:
        raise ValueError(f"Ảnh quá mờ (blur={blur_score:.1f}, yêu cầu ≥ {MIN_BLUR_SCORE:.0f}).")

    # 2. Kiểm tra độ sáng trung bình
    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        raise ValueError(
            f"Ánh sáng chưa phù hợp (brightness={brightness:.1f}, "
            f"yêu cầu {MIN_BRIGHTNESS:.0f}-{MAX_BRIGHTNESS:.0f})."
        )

    # 3. Phát hiện khuôn mặt
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError("Thư viện face_recognition/dlib chưa được cài đặt.") from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image_rgb, number_of_times_to_upsample=1)
    if len(locations) != 1:
        raise ValueError(f"Mỗi ảnh phải có đúng 1 khuôn mặt; hệ thống tìm thấy {len(locations)}.")

    # 4. Kiểm tra kích thước khuôn mặt
    top, right, bottom, left = locations[0]
    face_width = right - left
    face_height = bottom - top
    if face_width < MIN_FACE_SIZE_PX or face_height < MIN_FACE_SIZE_PX:
        raise ValueError(
            f"Khuôn mặt quá nhỏ ({face_width}×{face_height}px); yêu cầu tối thiểu {MIN_FACE_SIZE_PX}px."
        )

    # 5. Trích xuất 128D encoding
    encodings = face_recognition.face_encodings(
        image_rgb, known_face_locations=locations, num_jitters=1, model="small"
    )
    if len(encodings) != 1:
        raise ValueError("Không thể tạo vector khuôn mặt từ ảnh này.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    return EnrollmentSample(
        embedding=np.asarray(encodings[0], dtype=np.float64),
        image_hash=image_hash,
        blur_score=blur_score,
        brightness=brightness,
        face_width=face_width,
        face_height=face_height,
    )


def check_identity_consistency(
    samples: list[EnrollmentSample],
    max_distance: float = 0.50,
) -> None:
    """Kiểm tra tính nhất quán danh tính: tất cả các ảnh mẫu phải cùng một người.

    Tính khoảng cách pairwise giữa các vector. Nếu khoảng cách giữa 2 ảnh bất kỳ > max_distance,
    chứng tỏ đợt đăng ký bị lẫn ảnh của người khác.
    """
    for i, curr in enumerate(samples):
        for prev in samples[:i]:
            dist = float(np.linalg.norm(curr.embedding - prev.embedding))
            if dist > max_distance:
                raise ValueError(
                    f"Các ảnh đăng ký không nhất quán danh tính (khoảng cách {dist:.3f} > {max_distance:.3f}). "
                    "Vui lòng chỉ tải lên ảnh của cùng một người."
                )


def enroll_student(
    student_code: str,
    full_name: str,
    class_name: str,
    image_sources: Iterable[Any],
    consent_given: bool = True,
) -> tuple[int, list[str]]:
    """Xác thực và đăng ký sinh viên kèm các vector khuôn mặt vào cơ sở dữ liệu.

    Args:
        student_code: Mã số sinh viên.
        full_name: Họ tên sinh viên.
        class_name: Tên lớp sinh hoạt.
        image_sources: Danh sách file-like object hoặc byte data của ảnh.
        consent_given: Xác nhận đồng ý xử lý dữ liệu khuôn mặt.

    Returns:
        tuple[int, list[str]]: (Số ảnh đã lưu thành công, Danh sách cảnh báo nếu có).
    """
    if not consent_given:
        raise ValueError("Cần sự đồng ý (consent) của sinh viên trước khi xử lý dữ liệu khuôn mặt.")

    sources = list(image_sources)
    if not sources:
        raise ValueError("Hãy tải lên hoặc chụp ít nhất một ảnh.")

    valid_samples: list[tuple[str, EnrollmentSample]] = []
    warnings: list[str] = []
    seen_hashes: set[str] = set()

    for idx, src in enumerate(sources, start=1):
        name = getattr(src, "name", f"Ảnh {idx}")
        try:
            data = src.getvalue() if hasattr(src, "getvalue") else src
            sample = decode_and_validate_face(data)

            # Bỏ qua nếu ảnh trùng lặp chính xác (cùng hash nội dung)
            if sample.image_hash in seen_hashes:
                warnings.append(f"{name}: Bỏ qua do trùng lặp dữ liệu với ảnh trước.")
                continue

            seen_hashes.add(sample.image_hash)
            valid_samples.append((name, sample))
        except (AttributeError, OSError, ValueError) as exc:
            warnings.append(f"{name}: {exc}")

    if not valid_samples:
        raise ValueError("Không có ảnh hợp lệ nào. " + " | ".join(warnings))

    if len(valid_samples) < MIN_ENROLLMENT_IMAGES:
        raise ValueError(
            f"Đăng ký yêu cầu tối thiểu {MIN_ENROLLMENT_IMAGES} ảnh hợp lệ khác nhau "
            f"(hiện chỉ có {len(valid_samples)} ảnh). " + " | ".join(warnings)
        )

    # Kiểm tra nhất quán khuôn mặt giữa các ảnh trong cùng đợt đăng ký
    check_identity_consistency([s for _, s in valid_samples])

    # Lưu thông tin sinh viên vào SQLite
    student = upsert_student(
        student_code=student_code,
        full_name=full_name,
        class_name=class_name,
        consent_given=consent_given,
    )
    student_id = int(student["id"])

    # Lưu các vector khuôn mặt
    saved_count = 0
    for _, sample in valid_samples:
        save_embedding(
            student_id=student_id,
            embedding=sample.embedding,
            image_sha256=sample.image_hash,
            blur_score=sample.blur_score,
            brightness=sample.brightness,
            face_width=sample.face_width,
            face_height=sample.face_height,
        )
        saved_count += 1

    return saved_count, warnings
