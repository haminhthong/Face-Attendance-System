"""Module chứa các tiện ích bổ trợ cho hệ thống điểm danh khuôn mặt.

Bao gồm xử lý thời gian UTC/múi giờ Việt Nam, chuẩn hóa chuỗi dữ liệu đầu vào,
và băm/xác thực mã PIN bảo mật quản trị bằng mã hóa PBKDF2-HMAC-SHA256.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
import secrets
from datetime import date, datetime, timezone
from datetime import time as dt_time

import numpy as np

from .config import COURSE_CODE_PATTERN, PIN_ITERATIONS, STUDENT_CODE_PATTERN, VN_TZ


def utc_now() -> datetime:
    """Lấy thời điểm hiện tại theo múi giờ chuẩn chuẩn quốc tế (UTC)."""
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    """Chuyển đổi đối tượng datetime thành chuỗi định dạng ISO 8601 theo UTC.

    Args:
        value (datetime | None): Thời gian cần chuyển đổi, mặc định là hiện tại.

    Returns:
        str: Chuỗi ISO 8601 độ phân giải giây (ví dụ: '2026-08-30T12:00:00+00:00').
    """
    value = value or utc_now()
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def local_datetime(day: date, clock: dt_time) -> datetime:
    """Kết hợp ngày và giờ thành datetime có thông tin múi giờ Việt Nam (Asia/Ho_Chi_Minh)."""
    return datetime.combine(day, clock).replace(tzinfo=VN_TZ)


def display_datetime(value: str | None) -> str:
    """Đổi chuỗi ISO 8601 UTC sang ngày giờ Việt Nam để hiển thị.

    Args:
        value (str | None): Chuỗi thời gian ISO UTC từ DB.

    Returns:
        str: Chuỗi ngày giờ Việt Nam thân thiện người dùng.
    """
    if value is None:
        return ""
    # Pandas biểu diễn SQL NULL thành NaN; không đưa NaN vào
    # datetime.fromisoformat vì sẽ gây lỗi khi render sinh viên vắng.
    if not isinstance(value, str):
        try:
            if math.isnan(value):
                return ""
        except (TypeError, ValueError):
            return ""
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return value


def normalize_student_code(value: str) -> str:
    """Chuẩn hóa và kiểm tra mã sinh viên.

    Args:
        value (str): Mã sinh viên thô đầu vào.

    Returns:
        str: Mã sinh viên viết hoa, loại bỏ khoảng trắng thừa.

    Raises:
        ValueError: Nếu mã sinh viên không đúng định dạng regex.
    """
    code = value.strip().upper()
    if not STUDENT_CODE_PATTERN.fullmatch(code):
        raise ValueError("Mã sinh viên phải dài 3-30 ký tự và chỉ gồm chữ, số, '_' hoặc '-'.")
    return code


def normalize_course_code(value: str) -> str:
    """Chuẩn hóa và kiểm tra mã môn học.

    Args:
        value (str): Mã môn học thô đầu vào.

    Returns:
        str: Mã môn học viết hoa, loại bỏ khoảng trắng thừa.

    Raises:
        ValueError: Nếu mã môn học không đúng định dạng regex.
    """
    code = value.strip().upper()
    if not COURSE_CODE_PATTERN.fullmatch(code):
        raise ValueError("Mã môn học phải dài 2-30 ký tự và chỉ gồm chữ, số, '_', '-' hoặc '.'.")
    return code


def normalize_person_name(value: str) -> str:
    """Chuẩn hóa họ tên người dùng (viết đúng khoảng trắng, loại bỏ ký tự lạ).

    Args:
        value (str): Họ tên thô đầu vào.

    Returns:
        str: Họ tên đã được chuẩn hóa.

    Raises:
        ValueError: Nếu họ tên không đạt độ dài hoặc chứa ký tự không hợp lệ.
    """
    name = " ".join(value.strip().split())
    if not (2 <= len(name) <= 100):
        raise ValueError("Họ tên phải dài từ 2 đến 100 ký tự.")
    allowed_punctuation = {" ", "-", "'", "."}
    if not all(ch.isalnum() or ch in allowed_punctuation for ch in name):
        raise ValueError("Họ tên chứa ký tự không hợp lệ.")
    return name


def make_pin_hash(pin: str) -> str:
    """Mã hóa PIN quản trị bằng thuật toán PBKDF2-HMAC-SHA256 với Salt 16 bytes.

    Args:
        pin (str): Mã PIN 6-12 chữ số.

    Returns:
        str: Chuỗi băm dạng 'pbkdf2_sha256$<iterations>$<salt_b64>$<digest_b64>'.
    """
    if not re.fullmatch(r"\d{6,12}", pin):
        raise ValueError("PIN phải gồm từ 6 đến 12 chữ số.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PIN_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PIN_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_pin(pin: str, stored_value: str) -> bool:
    """Xác thực mã PIN nhập vào so với chuỗi băm lưu trong DB.

    Sử dụng hmac.compare_digest để chống tấn công bằng đo thời gian (Timing Attack).

    Args:
        pin (str): PIN nhập từ người dùng.
        stored_value (str): Chuỗi hash lưu trữ dạng PBKDF2.

    Returns:
        bool: True nếu khớp PIN, False nếu không khớp hoặc định dạng không hợp lệ.
    """
    try:
        algorithm, iterations, encoded_salt, encoded_digest = stored_value.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iteration_count = int(iterations)
        if not 100_000 <= iteration_count <= 1_000_000:
            return False
        salt = base64.b64decode(encoded_salt)
        expected = base64.b64decode(encoded_digest)
        if len(salt) != 16 or len(expected) != 32:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iteration_count)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def prepare_face_embedding(
    image_rgb: np.ndarray,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> np.ndarray | None:
    """Pipeline chuẩn hóa trích xuất vector khuôn mặt 128D cho cả đăng ký, kiểm thử và realtime.

    Args:
        image_rgb (np.ndarray): Ảnh RGB numpy array.
        detection_model (str): Mô hình phát hiện ('hog' hoặc 'cnn').
        num_jitters (int): Số lần biến đổi jitter để tăng độ chính xác vector.

    Returns:
        np.ndarray | None: Vector 128D hoặc None nếu không phát hiện duy nhất 1 khuôn mặt.
    """
    import face_recognition

    locations = face_recognition.face_locations(
        image_rgb,
        model=detection_model,
    )

    if len(locations) != 1:
        return None

    encodings = face_recognition.face_encodings(
        image_rgb,
        known_face_locations=locations,
        num_jitters=num_jitters,
    )
    return np.asarray(encodings[0], dtype=np.float64) if encodings else None
