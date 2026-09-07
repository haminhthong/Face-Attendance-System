"""Module quản lý cấu hình hệ thống điểm danh khuôn mặt.

Chứa các hằng số, biến môi trường, tham số thuật toán nhận diện,
ngưỡng kiểm tra chất lượng ảnh, cấu hình liveness và quy tắc kiểm tra dữ liệu đầu vào.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from .policy import DEFAULT_RECOGNITION_POLICY, RecognitionPolicy

# Thông tin cơ bản ứng dụng & Bảo mật API
APP_TITLE = "Hệ thống Điểm danh Sinh viên bằng Khuôn mặt"
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
API_KEY = os.getenv("FACE_ATTENDANCE_API_KEY", "").strip()

if APP_ENV == "production":
    if not API_KEY or API_KEY in {"change-me", "replace-with-a-long-random-secret", "default"}:
        raise RuntimeError(
            "Ứng dụng từ chối khởi chạy ở môi trường Production vì chưa cấu hình khóa FACE_ATTENDANCE_API_KEY bảo mật."
        )

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Đường dẫn lưu trữ dữ liệu ứng dụng & SQLite database
BASE_DIR = Path(__file__).resolve().parents[2]
custom_db = os.getenv("DATABASE_PATH", "").strip()
if custom_db:
    DB_PATH = Path(custom_db).expanduser().resolve()
    DATA_DIR = DB_PATH.parent
else:
    DATA_DIR = (
        Path(os.getenv("FACE_ATTENDANCE_DATA_DIR", str(BASE_DIR / "face_attendance_data")))
        .expanduser()
        .resolve()
    )
    DB_PATH = DATA_DIR / "face_attendance.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    """Đọc biến môi trường số thực (float) với kiểm tra giới hạn an toàn.

    Args:
        name: Tên biến môi trường.
        default: Giá trị mặc định nếu không khai báo.
        minimum: Giá trị nhỏ nhất cho phép.
        maximum: Giá trị lớn nhất cho phép.

    Returns:
        float: Giá trị số thực đã được xác thực.
    """
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} phải là số, nhận được {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} phải nằm trong khoảng {minimum}-{maximum}")
    return value


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Đọc biến môi trường số nguyên (int) với kiểm tra giới hạn an toàn.

    Args:
        name: Tên biến môi trường.
        default: Giá trị mặc định nếu không khai báo.
        minimum: Giá trị nhỏ nhất cho phép.
        maximum: Giá trị lớn nhất cho phép.

    Returns:
        int: Giá trị số nguyên đã được xác thực.
    """
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} phải là số nguyên, nhận được {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} phải nằm trong khoảng {minimum}-{maximum}")
    return value


# Chính sách nhận diện là nguồn sự thật duy nhất.
# Hai hằng số cũ được giữ lại để không làm hỏng client cũ; code mới không tự
# đọc threshold từ environment ở từng module nữa.
RECOGNITION_POLICY: RecognitionPolicy = DEFAULT_RECOGNITION_POLICY
FACE_TOLERANCE = RECOGNITION_POLICY.distance_threshold
MIN_IDENTITY_MARGIN = RECOGNITION_POLICY.identity_margin

if APP_ENV == "production" and not RECOGNITION_POLICY.calibrated:
    raise RuntimeError(
        "Production yêu cầu recognition policy đã được hiệu chuẩn trên private validation dataset."
    )

# Cấu hình xử lý camera WebRTC & Xác nhận đa khung hình
# Bỏ qua một số khung hình để cân bằng độ trễ và mức sử dụng CPU.
PROCESS_EVERY_N_FRAMES = _env_int("PROCESS_EVERY_N_FRAMES", 3, 1, 60)
# Tương thích ngược với UI/test cũ; runtime dùng thêm stable_duration_ms.
CONFIRMATION_FRAMES = RECOGNITION_POLICY.minimum_observations

# Cấu hình kiểm tra chất lượng ảnh đầu vào (Quality Control)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # Giới hạn kích thước file 8MB
MIN_FACE_SIZE_PX = 100  # Độ phân giải khuôn mặt tối thiểu (px)
MIN_BLUR_SCORE = 45.0  # Ngưỡng biến thiên Laplacian tối thiểu (chống mờ)
MIN_BRIGHTNESS = 40.0  # Ngưỡng độ sáng trung bình tối thiểu
MAX_BRIGHTNESS = 220.0  # Ngưỡng độ sáng trung bình tối đa

# Cấu hình máy trạng thái kiểm tra liveness chớp mắt (Eye Aspect Ratio - EAR)
BLINK_EAR_CLOSED = 0.19  # EAR khi nhắm mắt
BLINK_EAR_OPEN = 0.23  # EAR khi mở mắt
BLINK_VERIFICATION_SECONDS = RECOGNITION_POLICY.liveness_ttl_seconds
ATTEMPT_COOLDOWN_SECONDS = 8.0  # Cooldown giữa các lần thử ghi nhận điểm danh (giây)

# Chính sách lưu trữ dữ liệu sinh trắc học & Mã hóa PIN
# Thời hạn lưu embedding trước khi tác vụ dọn dữ liệu xóa bản ghi.
BIOMETRIC_RETENTION_DAYS = _env_int("BIOMETRIC_RETENTION_DAYS", 365, 1, 3650)
PIN_ITERATIONS = 240_000  # Số vòng lặp PBKDF2-HMAC-SHA256 băm PIN

# Regex kiểm tra định dạng dữ liệu đầu vào
STUDENT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,30}$")
COURSE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{2,30}$")
