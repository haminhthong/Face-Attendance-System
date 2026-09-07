"""Module kiểm tra tương tác chớp mắt cơ bản (Basic Blink Challenge).

Sử dụng tỉ lệ đặc trưng của mắt (Eye Aspect Ratio - EAR) dựa trên 6 mốc khuôn mặt
và máy trạng thái 4 bước. Cơ chế này không chống được replay video/deepfake/PAD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def ti_le_mat(points: list[tuple[int, int]]) -> float | None:
    """Tính Tỉ Lệ Đặc Trưng Mắt (Eye Aspect Ratio - EAR) từ 6 tọa độ mốc của mắt.

    Công thức:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Sáu điểm mốc gồm hai góc mắt và bốn điểm ở viền trên/dưới.

    Args:
        points (list[tuple[int, int]]): Danh sách 6 tọa độ (x, y) của một mắt.

    Returns:
        Giá trị EAR hoặc None nếu dữ liệu không hợp lệ.
    """
    if len(points) != 6:
        return None
    values = np.asarray(points, dtype=np.float64)
    # Khoảng cách ngang: ||p1 - p4||
    ngang = float(np.linalg.norm(values[0] - values[3]))
    if ngang <= 1e-6:
        return None
    # Khoảng cách dọc 1: ||p2 - p6||, Khoảng cách dọc 2: ||p3 - p5||
    doc_1 = float(np.linalg.norm(values[1] - values[5]))
    doc_2 = float(np.linalg.norm(values[2] - values[4]))
    return (doc_1 + doc_2) / (2.0 * ngang)


@dataclass
class BoKiemTraChopMat:
    """Máy trạng thái theo dõi và xác nhận chu trình chớp mắt của từng sinh viên.

    Attributes:
        nguong_nham (float): Ngưỡng EAR coi là nhắm mắt (ví dụ: <= 0.19).
        nguong_mo (float): Ngưỡng EAR coi là mở mắt (ví dụ: >= 0.23).
        thoi_han_giay: Thời gian xác minh còn hiệu lực.
        trang_thai (dict[int, str]): Lưu trữ trạng thái hiện tại của từng student_id.
        da_xac_minh_luc: Thời điểm xác minh thành công theo đồng hồ monotonic.
    """

    nguong_nham: float
    nguong_mo: float
    thoi_han_giay: float
    trang_thai: dict[int, str] = field(default_factory=dict)
    da_xac_minh_luc: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các ngưỡng cấu hình."""
        if not 0 < self.nguong_nham < self.nguong_mo < 1:
            raise ValueError("Ngưỡng nhắm mắt phải nhỏ hơn ngưỡng mở mắt.")
        if self.thoi_han_giay <= 0:
            raise ValueError("Thời hạn xác minh phải lớn hơn 0 giây.")

    def cap_nhat(self, student_id: int, ti_le: float | None) -> bool:
        """Cập nhật trạng thái chớp mắt của sinh viên dựa trên tỉ lệ EAR mới nhất.

        Luồng chuyển trạng thái:
        1. `can_mo`: Chờ mắt mở (EAR >= nguong_mo) -> chuyển sang `can_nham`.
        2. `can_nham`: Chờ nhắm mắt (EAR <= nguong_nham) -> chuyển sang `can_mo_lai`.
        3. `can_mo_lai`: Chờ mở mắt để hoàn tất xác minh.
        4. `da_xac_minh`: Đã xác minh thành công. Reset về `can_mo` khi hết thời hạn (TTL).

        Args:
            student_id (int): ID sinh viên.
            ti_le (float | None): Tỉ lệ EAR trung bình của 2 mắt.

        Returns:
            bool: True nếu sinh viên đã vượt qua kiểm tra liveness và còn hiệu lực.
        """
        if ti_le is None or not np.isfinite(ti_le) or ti_le < 0:
            return False
        luc_nay = time.monotonic()
        trang_thai = self.trang_thai.get(student_id, "can_mo")

        # Kiểm tra nếu trạng thái da_xac_minh còn hiệu lực TTL hay không
        if trang_thai == "da_xac_minh":
            if luc_nay - self.da_xac_minh_luc.get(student_id, 0) <= self.thoi_han_giay:
                return True
            trang_thai = "can_mo"  # Hết hiệu lực -> làm mới chu trình

        # Máy trạng thái nhận diện chớp mắt
        if trang_thai == "can_mo" and ti_le >= self.nguong_mo:
            trang_thai = "can_nham"
        elif trang_thai == "can_nham" and ti_le <= self.nguong_nham:
            trang_thai = "can_mo_lai"
        elif trang_thai == "can_mo_lai" and ti_le >= self.nguong_mo:
            trang_thai = "da_xac_minh"
            self.da_xac_minh_luc[student_id] = luc_nay

        self.trang_thai[student_id] = trang_thai
        return trang_thai == "da_xac_minh"

    def dat_lai(self, student_id: int) -> None:
        """Xóa trạng thái khi khuôn mặt rời khỏi khung hình."""
        self.trang_thai.pop(student_id, None)
        self.da_xac_minh_luc.pop(student_id, None)
