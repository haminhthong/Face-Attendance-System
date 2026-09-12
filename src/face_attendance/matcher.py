"""Module thuật toán so khớp khuôn mặt tập mở (Open-Set Face Recognition).

Sử dụng khoảng cách Euclidean L2 trên vector 128 chiều trích xuất từ dlib ResNet.
Áp dụng chiến lược Top-K Mean để gom khoảng cách của các template tham chiếu
cho mỗi sinh viên và từ chối người lạ (Unknown Rejection) bằng hai điều kiện:
1. Khoảng cách tốt nhất (Top-1) phải <= distance_threshold.
2. Chênh lệch (Margin) giữa Top-2 và Top-1 phải >= identity_margin để loại bỏ mơ hồ danh tính.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class FaceTemplateProtocol(Protocol):
    """Protocol đại diện cho mẫu khuôn mặt tham chiếu."""

    student_id: int
    embedding: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    """Kết quả so khớp khuôn mặt.

    Attributes:
        template: Mẫu tham chiếu tốt nhất của Top-1 (None nếu bị từ chối / Unknown).
        distance: Khoảng cách đại diện của Top-1 candidate.
        margin: Chênh lệch khoảng cách giữa Top-2 và Top-1 (second_distance - distance).
        second_distance: Khoảng cách đại diện của Top-2 candidate.
        best_student_id: ID sinh viên Top-1.
        second_student_id: ID sinh viên Top-2.
    """

    template: Any | None
    distance: float
    margin: float
    second_distance: float = float("inf")
    best_student_id: int | None = None
    second_student_id: int | None = None

    @property
    def is_matched(self) -> bool:
        """Kiểm tra khuôn mặt có được chấp nhận danh tính hay không."""
        return self.template is not None


def match_face(
    embedding: np.ndarray,
    templates: Sequence[FaceTemplateProtocol],
    distance_threshold: float = 0.50,
    identity_margin: float = 0.05,
    top_k: int = 2,
) -> MatchResult:
    """So khớp vector khuôn mặt đầu vào với gallery mẫu sử dụng Open-Set Matching.

    Args:
        embedding: Vector khuôn mặt đầu vào (128 chiều, float).
        templates: Danh sách mẫu khuôn mặt đã đăng ký của các sinh viên.
        distance_threshold: Ngưỡng L2 tối đa để chấp nhận match (mặc định 0.50).
        identity_margin: Ngưỡng chênh lệch tối thiểu giữa Top-1 và Top-2 (mặc định 0.05).
        top_k: Số mẫu gần nhất được dùng để tính khoảng cách trung bình cho mỗi sinh viên.

    Returns:
        MatchResult: Chứa thông tin ứng viên khớp nhất hoặc Unknown nếu bị từ chối.
    """
    if not 0 <= distance_threshold <= 2:
        raise ValueError("Ngưỡng khoảng cách phải nằm trong khoảng 0-2.")
    if not 0 <= identity_margin <= 2:
        raise ValueError("Ngưỡng phân biệt phải nằm trong khoảng 0-2.")
    if not templates:
        return MatchResult(None, float("inf"), float("inf"))

    query_vec = np.asarray(embedding, dtype=np.float64)
    if query_vec.shape != (128,) or not np.isfinite(query_vec).all():
        raise ValueError("Embedding đầu vào phải có 128 giá trị hữu hạn.")

    # 1. Gom các mẫu theo từng sinh viên và tính khoảng cách L2
    by_student: dict[int, list[tuple[float, FaceTemplateProtocol]]] = {}
    for tpl in templates:
        tpl_vec = np.asarray(tpl.embedding, dtype=np.float64)
        if tpl_vec.shape != (128,) or not np.isfinite(tpl_vec).all():
            raise ValueError("Embedding mẫu phải có 128 giá trị hữu hạn.")
        dist = float(np.linalg.norm(tpl_vec - query_vec))
        by_student.setdefault(tpl.student_id, []).append((dist, tpl))

    # 2. Tính khoảng cách đại diện Top-K Mean cho từng sinh viên
    candidates: list[tuple[float, FaceTemplateProtocol, int]] = []
    for student_id, sample_list in by_student.items():
        sample_list.sort(key=lambda x: x[0])
        best_tpl = sample_list[0][1]
        k = max(1, min(top_k, len(sample_list)))
        rep_distance = float(np.mean([d for d, _ in sample_list[:k]]))
        candidates.append((rep_distance, best_tpl, student_id))

    # 3. Sắp xếp các danh tính theo khoảng cách tăng dần
    candidates.sort(key=lambda x: x[0])
    best_distance, best_tpl, best_id = candidates[0]

    if len(candidates) > 1:
        second_distance, _, second_id = candidates[1]
    else:
        # Trường hợp chỉ có 1 sinh viên trong hệ thống
        second_distance = best_distance + identity_margin
        second_id = None

    margin = second_distance - best_distance

    # 4. Open-set recognition gate:
    # - Khoảng cách tốt nhất phải <= distance_threshold
    # - Margin giữa #1 và #2 phải >= identity_margin
    if best_distance <= distance_threshold and margin >= identity_margin:
        matched_tpl = best_tpl
    else:
        matched_tpl = None

    return MatchResult(
        template=matched_tpl,
        distance=best_distance,
        margin=margin,
        second_distance=second_distance,
        best_student_id=best_id,
        second_student_id=second_id,
    )
