"""Module thuật toán so khớp khuôn mặt tập mở (Open-Set Face Recognition).

Thực hiện tính khoảng cách Euclidean L2 giữa vector khuôn mặt đầu vào (128D) và các mẫu tham chiếu,
gom nhóm theo từng sinh viên và áp dụng cơ chế từ chối người lạ (Unknown Rejection) bằng hai ngưỡng:
1. Ngưỡng khoảng cách tối đa từ RecognitionPolicy.
2. Ngưỡng chênh lệch giữa Top-1 và Top-2 từ RecognitionPolicy nhằm loại bỏ sự mơ hồ danh tính.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np


class MauKhuonMat(Protocol):
    """Protocol đại diện cho mẫu khuôn mặt tham chiếu."""

    student_id: int
    embedding: np.ndarray


class AggregationStrategy(str, Enum):
    """Chiến lược gom cụm khoảng cách danh tính khi sinh viên có nhiều ảnh mẫu."""

    MIN_DISTANCE = "min_distance"
    CENTROID = "centroid"
    TOP_K_MEAN = "top_k_mean"


@dataclass(frozen=True)
class KetQuaSoKhop:
    """Kết quả so khớp khuôn mặt.

    Attributes:
        mau (MauKhuonMat | None): Mẫu khuôn mặt khớp nhất (None nếu bị từ chối).
        khoang_cach (float): Khoảng cách đại diện tốt nhất của Top-1 candidate.
        do_phan_biet (float): Chênh lệch khoảng cách giữa Top-1 và Top-2 (Margin).
        khoang_cach_thu_hai (float): Khoảng cách đại diện của Top-2 candidate.
        student_id_tot_nhat (int | None): ID sinh viên xếp hạng 1.
        student_id_thu_hai (int | None): ID sinh viên xếp hạng 2.
    """

    mau: MauKhuonMat | None
    khoang_cach: float
    do_phan_biet: float
    khoang_cach_thu_hai: float = float("inf")
    student_id_tot_nhat: int | None = None
    student_id_thu_hai: int | None = None


def tim_danh_tinh_tot_nhat(
    embedding: np.ndarray,
    danh_sach_mau: Sequence[MauKhuonMat],
    nguong_khoang_cach: float,
    nguong_phan_biet: float,
    strategy: AggregationStrategy | str = AggregationStrategy.TOP_K_MEAN,
    top_k: int = 2,
) -> KetQuaSoKhop:
    """Xác định danh tính phù hợp nhất từ vector đầu vào sử dụng thuật toán Open-Set Matching.

    Hỗ trợ 3 chiến lược gom cụm mẫu (Identity Aggregation Strategies):
    1. TOP_K_MEAN: Trung bình khoảng cách của Top-K mẫu gần nhất của sinh viên.
    2. MIN_DISTANCE/CENTROID: Chỉ giữ cho benchmark và nghiên cứu, không dùng runtime.

    Args:
        embedding (np.ndarray): Vector khuôn mặt đầu vào (128 chiều).
        danh_sach_mau (Sequence[MauKhuonMat]): Danh sách tất cả ảnh mẫu tham chiếu.
        nguong_khoang_cach (float): Khoảng cách L2 tối đa chấp nhận (FACE_TOLERANCE).
        nguong_phan_biet (float): Chênh lệch tối thiểu giữa hai ứng viên đầu (Margin).
        strategy: Chiến lược gom mẫu ('min_distance', 'centroid', 'top_k_mean').
        top_k: Số mẫu gần nhất dùng khi chọn strategy='top_k_mean'.

    Returns:
        KetQuaSoKhop: Đối tượng chứa thông tin danh tính khớp nhất hoặc None nếu bị từ chối.

    Raises:
        ValueError: Nếu ngưỡng hoặc dữ liệu vector đầu vào/mẫu không hợp lệ.
    """
    if not 0 <= nguong_khoang_cach <= 2:
        raise ValueError("Ngưỡng khoảng cách phải nằm trong khoảng 0-2.")
    if not 0 <= nguong_phan_biet <= 2:
        raise ValueError("Ngưỡng phân biệt phải nằm trong khoảng 0-2.")
    if not danh_sach_mau:
        return KetQuaSoKhop(None, float("inf"), float("inf"))

    strat = strategy if isinstance(strategy, AggregationStrategy) else AggregationStrategy(strategy)
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("Embedding đầu vào phải có 128 giá trị hữu hạn.")

    # Gom các mẫu theo từng sinh viên
    theo_sinh_vien: dict[int, list[tuple[float, MauKhuonMat]]] = {}
    for mau in danh_sach_mau:
        vector_mau = np.asarray(mau.embedding, dtype=np.float64)
        if vector_mau.shape != (128,) or not np.isfinite(vector_mau).all():
            raise ValueError("Embedding mẫu phải có 128 giá trị hữu hạn.")
        khoang_cach = float(np.linalg.norm(vector_mau - vector))
        theo_sinh_vien.setdefault(mau.student_id, []).append((khoang_cach, mau))

    # Tính khoảng cách đại diện cho từng sinh viên theo strategy
    ung_vien: list[tuple[float, MauKhuonMat, int]] = []
    for student_id, mau_list in theo_sinh_vien.items():
        # Sắp xếp các mẫu của sinh viên này theo khoảng cách tăng dần
        mau_list.sort(key=lambda item: item[0])
        best_single_mau = mau_list[0][1]

        if strat == AggregationStrategy.MIN_DISTANCE:
            rep_distance = mau_list[0][0]
        elif strat == AggregationStrategy.CENTROID:
            all_embs = np.array([m.embedding for _, m in mau_list], dtype=np.float64)
            centroid = np.mean(all_embs, axis=0)
            # Centroid phải dùng cùng normalization contract với query.
            centroid_norm = np.linalg.norm(centroid)
            query_norm = np.linalg.norm(vector)
            if centroid_norm <= 1e-9 or query_norm <= 1e-9:
                rep_distance = float(np.linalg.norm(centroid - vector))
            else:
                rep_distance = float(
                    np.linalg.norm(centroid / centroid_norm - vector / query_norm)
                )
        elif strat == AggregationStrategy.TOP_K_MEAN:
            k = max(1, min(top_k, len(mau_list)))
            rep_distance = float(np.mean([d for d, _ in mau_list[:k]]))
        else:
            rep_distance = mau_list[0][0]

        ung_vien.append((rep_distance, best_single_mau, student_id))

    # Sắp xếp các danh tính ứng viên theo khoảng cách tăng dần
    ung_vien.sort(key=lambda item: item[0])
    khoang_cach_tot_nhat, mau_tot_nhat, id_tot_nhat = ung_vien[0]

    if len(ung_vien) > 1:
        khoang_cach_thu_hai, _, id_thu_hai = ung_vien[1]
    else:
        khoang_cach_thu_hai = float("inf")
        id_thu_hai = None

    do_phan_biet = khoang_cach_thu_hai - khoang_cach_tot_nhat

    # Kiểm tra điều kiện từ chối người lạ / mơ hồ danh tính
    # 1. Khoảng cách tốt nhất phải <= nguong_khoang_cach
    # 2. Độ chênh lệch giữa ứng viên #1 và ứng viên #2 phải >= nguong_phan_biet
    if (
        khoang_cach_tot_nhat > nguong_khoang_cach
        or do_phan_biet < nguong_phan_biet
    ):
        ket_qua_mau = None
    else:
        ket_qua_mau = mau_tot_nhat

    return KetQuaSoKhop(
        mau=ket_qua_mau,
        khoang_cach=khoang_cach_tot_nhat,
        do_phan_biet=do_phan_biet,
        khoang_cach_thu_hai=khoang_cach_thu_hai,
        student_id_tot_nhat=id_tot_nhat,
        student_id_thu_hai=id_thu_hai,
    )
