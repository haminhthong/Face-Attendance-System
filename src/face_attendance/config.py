"""Cấu hình hệ thống điểm danh sinh viên bằng khuôn mặt.

Chứa các hằng số, đường dẫn lưu trữ, cấu hình ngưỡng nhận diện,
kiểm tra chất lượng ảnh và tham số liveness chớp mắt.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    """Nạp biến môi trường từ file .env nếu có."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(BASE_DIR / ".env", override=False)


_load_project_env()

# Thông tin ứng dụng
APP_TITLE = "Hệ thống Điểm danh Sinh viên bằng Khuôn mặt"
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
API_KEY = os.getenv("FACE_ATTENDANCE_API_KEY", "").strip()
ADMIN_PIN = os.getenv("ADMIN_PIN", "123456").strip()

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Đường dẫn lưu trữ SQLite database
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

# Cấu hình logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    LOG_LEVEL = "INFO"
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")


@dataclass(frozen=True)
class RecognitionConfig:
    """Cấu hình tham số nhận diện và ngưỡng phân biệt danh tính."""

    # Ngưỡng khoảng cách L2 Euclidean tối đa để chấp nhận match
    distance_threshold: float = 0.50
    # Ngưỡng margin tối thiểu giữa Top-1 và Top-2 để chống match mơ hồ
    identity_margin: float = 0.05
    # Số lượng template gần nhất dùng để tính khoảng cách trung bình (Top-K Mean)
    top_k: int = 2

    # Số frame liên tiếp cùng một danh tính để xác nhận
    min_confirmations: int = 3
    # Thời gian tracking ổn định tối thiểu (ms)
    stable_duration_ms: int = 500

    # Ngưỡng tỉ lệ mắt (Eye Aspect Ratio - EAR)
    eye_closed_threshold: float = 0.19
    eye_open_threshold: float = 0.23


DEFAULT_CONFIG = RecognitionConfig()

# Giữ các hằng số tiện ích cho module khác
FACE_DISTANCE_THRESHOLD = DEFAULT_CONFIG.distance_threshold
IDENTITY_MARGIN = DEFAULT_CONFIG.identity_margin
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))
CONFIRMATION_FRAMES = DEFAULT_CONFIG.min_confirmations
STABLE_DURATION_MS = DEFAULT_CONFIG.stable_duration_ms

# Kiểm soát chất lượng ảnh đầu vào (Quality gates)
MAX_UPLOAD_SIZE_MB = 8
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MIN_FACE_SIZE_PX = 100
MIN_BLUR_SCORE = 40.0  # Laplacian variance
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0
MIN_ENROLLMENT_IMAGES = 5

# Tham số kiểm tra chớp mắt
BLINK_EAR_CLOSED = DEFAULT_CONFIG.eye_closed_threshold
BLINK_EAR_OPEN = DEFAULT_CONFIG.eye_open_threshold
ATTEMPT_COOLDOWN_SECONDS = 5.0

# Regex kiểm tra định dạng
STUDENT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,30}$")
COURSE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{2,30}$")
PIN_ITERATIONS = 240_000
