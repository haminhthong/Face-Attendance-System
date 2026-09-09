"""Service đăng ký và quản lý mẫu khuôn mặt sinh viên (Enrollment Application Service)."""

from __future__ import annotations

from typing import Any, Iterable

from ..config import BIOMETRIC_CONSENT_POLICY_VERSION
from ..utils import normalize_person_name, normalize_student_code


def process_student_enrollment(
    student_code: str,
    full_name: str,
    class_name: str,
    image_sources: Iterable[Any],
    consent_given: bool = False,
    consent_policy_version: str = BIOMETRIC_CONSENT_POLICY_VERSION,
) -> tuple[int, list[str]]:
    """Chuẩn hóa dữ liệu đầu vào và đăng ký sinh viên kèm ảnh mẫu.

    Args:
        student_code: Mã sinh viên.
        full_name: Họ tên sinh viên.
        class_name: Tên lớp sinh hoạt.
        image_sources: Danh sách dữ liệu ảnh nhị phân hoặc file-like objects.

    Returns:
        tuple[int, list[str]]: (Số ảnh mẫu đã đăng ký thành công, Cảnh báo/lỗi nếu có).
    """
    if not consent_given:
        raise ValueError("Cần consent rõ ràng trước khi giải mã hoặc xử lý ảnh khuôn mặt.")

    clean_code = normalize_student_code(student_code)
    clean_name = normalize_person_name(full_name)
    clean_class = class_name.strip()

    # Import trễ để API/attendance service không bắt buộc cài OpenCV nếu chỉ
    # chạy các nghiệp vụ database hoặc backend không dùng camera.
    from ..recognition import enroll_student_images

    return enroll_student_images(
        student_code=clean_code,
        full_name=clean_name,
        class_name=clean_class,
        image_sources=image_sources,
        consent_given=True,
        consent_policy_version=consent_policy_version,
    )
