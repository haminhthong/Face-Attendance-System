"""Module kiểm tra tương tác chớp mắt cơ bản (Basic Blink Challenge).

Sử dụng tỉ lệ khung mắt (Eye Aspect Ratio - EAR) dựa trên 6 điểm mốc của mỗi mắt
và máy trạng thái 4 bước: eyes open -> eyes closed -> eyes open -> verified.
Lưu ý: Đây là cơ chế challenge-response heuristic để tránh ảnh tĩnh đơn giản,
không thay thế cho các giải pháp Face PAD / anti-spoofing chuyên dụng.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def eye_aspect_ratio(landmarks_points: list[tuple[int, int]]) -> float | None:
    """Tính Eye Aspect Ratio (EAR) từ 6 tọa độ mốc của một mắt.

    Công thức:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        landmarks_points: Danh sách 6 điểm (x, y) của vùng mắt.

    Returns:
        Giá trị EAR hoặc None nếu dữ liệu không hợp lệ.
    """
    if len(landmarks_points) != 6:
        return None
    pts = np.asarray(landmarks_points, dtype=np.float64)
    horizontal = float(np.linalg.norm(pts[0] - pts[3]))
    if horizontal <= 1e-6:
        return None
    vertical_1 = float(np.linalg.norm(pts[1] - pts[5]))
    vertical_2 = float(np.linalg.norm(pts[2] - pts[4]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# Alias tiếng Việt
ti_le_mat = eye_aspect_ratio


@dataclass
class BlinkDetector:
    """Máy trạng thái theo dõi và xác nhận chu trình chớp mắt của từng khuôn mặt.

    Trạng thái:
    - `can_mo`: Chờ mắt mở (EAR >= eye_open) -> chuyển sang `can_nham`.
    - `can_nham`: Chờ nhắm mắt (EAR <= eye_closed) -> chuyển sang `can_mo_lai`.
    - `can_mo_lai`: Chờ mở mắt lại để hoàn tất 1 chu trình chớp.
    - `da_xac_minh`: Đã xác minh thành công.
    """

    eye_closed_threshold: float = 0.19
    eye_open_threshold: float = 0.23
    ttl_seconds: float = 10.0
    states: dict[int, str] = field(default_factory=dict)
    verified_at: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.eye_closed_threshold < self.eye_open_threshold < 1:
            raise ValueError("Ngưỡng nhắm mắt phải nhỏ hơn ngưỡng mở mắt.")
        if self.ttl_seconds <= 0:
            raise ValueError("Thời hạn xác minh phải lớn hơn 0 giây.")

    def update(self, student_id: int, ear: float | None) -> bool:
        """Cập nhật tỉ lệ EAR của sinh viên và kiểm tra trạng thái chớp mắt."""
        if ear is None or not np.isfinite(ear) or ear < 0:
            return False

        now = time.monotonic()
        state = self.states.get(student_id, "can_mo")

        # Kiểm tra TTL xác minh
        if state == "da_xac_minh":
            if now - self.verified_at.get(student_id, 0) <= self.ttl_seconds:
                return True
            state = "can_mo"

        # Chuyển đổi trạng thái FSM
        if state == "can_mo" and ear >= self.eye_open_threshold:
            state = "can_nham"
        elif state == "can_nham" and ear <= self.eye_closed_threshold:
            state = "can_mo_lai"
        elif state == "can_mo_lai" and ear >= self.eye_open_threshold:
            state = "da_xac_minh"
            self.verified_at[student_id] = now

        self.states[student_id] = state
        return state == "da_xac_minh"

    def reset(self, student_id: int) -> None:
        """Xóa trạng thái theo dõi khi khuôn mặt rời khỏi khung hình hoặc đổi người."""
        self.states.pop(student_id, None)
        self.verified_at.pop(student_id, None)


# Alias tương thích
class BoKiemTraChopMat:
    """Wrapper tương thích mã cũ."""

    def __init__(
        self,
        nguong_nham: float = 0.19,
        nguong_mo: float = 0.23,
        thoi_han_giay: float = 10.0,
    ) -> None:
        self._detector = BlinkDetector(
            eye_closed_threshold=nguong_nham,
            eye_open_threshold=nguong_mo,
            ttl_seconds=thoi_han_giay,
        )

    def cap_nhat(self, student_id: int, ti_le: float | None) -> bool:
        return self._detector.update(student_id, ti_le)

    def dat_lai(self, student_id: int) -> None:
        self._detector.reset(student_id)
