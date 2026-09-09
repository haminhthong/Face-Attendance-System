"""Unit tests cho các chiến lược gom cụm khoảng cách danh tính (Identity Aggregation Strategies)."""

from dataclasses import dataclass

import numpy as np
import pytest

from face_attendance.matcher import AggregationStrategy, tim_danh_tinh_tot_nhat


@dataclass(frozen=True)
class MauKhuonMatMock:
    student_id: int
    embedding: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def test_min_distance_aggregation() -> None:
    query = normalize(np.ones(128))

    # Student 1: mẫu gần (dist ~ 0.2) và mẫu xa (dist ~ 0.6)
    # L2 distance giữa 2 normalized vectors: d = sqrt(2 - 2 * cos)
    # Ta có thể tạo vector trực tiếp
    emb1_close = normalize(np.ones(128) + np.random.RandomState(42).normal(0, 0.05, 128))
    emb1_far = normalize(np.ones(128) + np.random.RandomState(43).normal(0, 0.5, 128))

    # Student 2: mẫu khoảng cách trung bình
    emb2 = normalize(np.ones(128) + np.random.RandomState(44).normal(0, 0.3, 128))

    d1_close = float(np.linalg.norm(emb1_close - query))
    d1_far = float(np.linalg.norm(emb1_far - query))
    min_d1 = min(d1_close, d1_far)
    d2 = float(np.linalg.norm(emb2 - query))

    samples = [
        MauKhuonMatMock(1, emb1_far),
        MauKhuonMatMock(1, emb1_close),
        MauKhuonMatMock(2, emb2),
    ]

    res = tim_danh_tinh_tot_nhat(
        query,
        samples,
        nguong_khoang_cach=0.8,
        nguong_phan_biet=0.01,
        strategy=AggregationStrategy.MIN_DISTANCE,
    )

    assert res.student_id_tot_nhat == 1
    assert pytest.approx(res.khoang_cach, abs=1e-5) == min_d1
    assert pytest.approx(res.khoang_cach_thu_hai, abs=1e-5) == d2


def test_centroid_aggregation() -> None:
    query = np.zeros(128)
    query[0] = 1.0  # Vector đơn vị trên chiều 0

    # Student 1: hai vector
    v1 = np.zeros(128)
    v1[0] = 0.8
    v1[1] = 0.6
    v1 = normalize(v1)

    v2 = np.zeros(128)
    v2[0] = 0.8
    v2[1] = -0.6
    v2 = normalize(v2)

    # Centroid của v1 và v2: [0.8, 0, ...], normalized là [1.0, 0, ...]
    centroid = normalize((v1 + v2) / 2.0)
    expected_dist = float(np.linalg.norm(centroid - query))

    samples = [
        MauKhuonMatMock(1, v1),
        MauKhuonMatMock(1, v2),
    ]

    res = tim_danh_tinh_tot_nhat(
        query,
        samples,
        nguong_khoang_cach=0.5,
        nguong_phan_biet=0.0,
        strategy=AggregationStrategy.CENTROID,
    )

    assert res.student_id_tot_nhat == 1
    assert pytest.approx(res.khoang_cach, abs=1e-5) == expected_dist


def test_top_k_mean_aggregation() -> None:
    query = np.zeros(128)
    query[0] = 1.0

    # Tạo 3 mẫu cho Student 1 với khoảng cách đã biết
    v1 = np.zeros(128)
    v1[0] = 0.95
    v2 = np.zeros(128)
    v2[0] = 0.85
    v3 = np.zeros(128)
    v3[0] = 0.50

    d1 = float(np.linalg.norm(v1 - query))
    d2 = float(np.linalg.norm(v2 - query))
    _d3 = float(np.linalg.norm(v3 - query))

    samples = [
        MauKhuonMatMock(1, v1),
        MauKhuonMatMock(1, v2),
        MauKhuonMatMock(1, v3),
    ]

    expected_top2_mean = (d1 + d2) / 2.0

    res = tim_danh_tinh_tot_nhat(
        query,
        samples,
        nguong_khoang_cach=1.0,
        nguong_phan_biet=0.0,
        strategy=AggregationStrategy.TOP_K_MEAN,
        top_k=2,
    )

    assert res.student_id_tot_nhat == 1
    assert pytest.approx(res.khoang_cach, abs=1e-5) == expected_top2_mean


def test_strategy_string_input() -> None:
    query = np.zeros(128)
    samples = [MauKhuonMatMock(1, np.full(128, 0.01))]

    res = tim_danh_tinh_tot_nhat(
        query,
        samples,
        0.5,
        0.05,
        strategy="min_distance",
    )
    assert res.mau is not None
