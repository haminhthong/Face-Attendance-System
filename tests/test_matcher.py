from dataclasses import dataclass

import numpy as np
import pytest

from face_attendance.matcher import tim_danh_tinh_tot_nhat


@dataclass(frozen=True)
class Mau:
    student_id: int
    embedding: np.ndarray


def test_matcher_accepts_clear_identity() -> None:
    query = np.zeros(128)
    samples = [Mau(1, np.zeros(128)), Mau(2, np.ones(128))]
    result = tim_danh_tinh_tot_nhat(query, samples, 0.5, 0.05)
    assert result.mau is samples[0]


def test_matcher_rejects_ambiguous_identity() -> None:
    query = np.zeros(128)
    samples = [Mau(1, np.full(128, 0.01)), Mau(2, np.full(128, 0.011))]
    result = tim_danh_tinh_tot_nhat(query, samples, 0.5, 0.05)
    assert result.mau is None


def test_matcher_rejects_invalid_template_size() -> None:
    with pytest.raises(ValueError, match="Embedding mẫu"):
        tim_danh_tinh_tot_nhat(np.zeros(128), [Mau(1, np.zeros(127))], 0.5, 0.05)

    with pytest.raises(ValueError, match="Embedding mẫu"):
        tim_danh_tinh_tot_nhat(np.zeros(128), [Mau(1, np.zeros(129))], 0.5, 0.05)


def test_matcher_rejects_invalid_query_size() -> None:
    with pytest.raises(ValueError, match="Embedding đầu vào"):
        tim_danh_tinh_tot_nhat(np.zeros(127), [Mau(1, np.zeros(128))], 0.5, 0.05)


def test_matcher_rejects_nan_or_inf() -> None:
    query_nan = np.zeros(128)
    query_nan[0] = np.nan
    with pytest.raises(ValueError, match="Embedding đầu vào"):
        tim_danh_tinh_tot_nhat(query_nan, [Mau(1, np.zeros(128))], 0.5, 0.05)

    query_inf = np.zeros(128)
    query_inf[0] = np.inf
    with pytest.raises(ValueError, match="Embedding đầu vào"):
        tim_danh_tinh_tot_nhat(query_inf, [Mau(1, np.zeros(128))], 0.5, 0.05)


def test_matcher_empty_database() -> None:
    result = tim_danh_tinh_tot_nhat(np.zeros(128), [], 0.5, 0.05)
    assert result.mau is None
    assert result.khoang_cach == float("inf")
    assert result.do_phan_biet == float("inf")


def test_matcher_boundary_thresholds() -> None:
    query = np.zeros(128)
    # Khoảng cách L2 giữa zeros(128) và vec là sqrt(128 * 0.001953125) = sqrt(0.25) = 0.50
    val = np.sqrt(0.25 / 128)
    sample_exact = Mau(1, np.full(128, val))

    # Khoảng cách đúng bằng threshold 0.50 -> được chấp nhận
    res_exact = tim_danh_tinh_tot_nhat(query, [sample_exact], 0.50, 0.0)
    assert res_exact.mau is sample_exact
    assert pytest.approx(res_exact.khoang_cach, abs=1e-5) == 0.50

    # Khoảng cách lớn hơn 0.50 -> bị từ chối
    val_larger = np.sqrt(0.26 / 128)
    sample_larger = Mau(1, np.full(128, val_larger))
    res_larger = tim_danh_tinh_tot_nhat(query, [sample_larger], 0.50, 0.0)
    assert res_larger.mau is None


def test_matcher_single_identity_serializes_finite_margin() -> None:
    result = tim_danh_tinh_tot_nhat(np.zeros(128), [Mau(1, np.zeros(128))], 0.5, 0.05)

    assert result.mau is not None
    assert np.isfinite(result.khoang_cach_thu_hai)
    assert result.do_phan_biet == 0.05


def test_top_k_mean_averages_closest_k_samples() -> None:
    query = np.zeros(128)
    # Student 1 có 3 samples với khoảng cách 0.2, 0.4, 0.8
    v1 = np.full(128, np.sqrt(0.04 / 128))  # dist = 0.2
    v2 = np.full(128, np.sqrt(0.16 / 128))  # dist = 0.4
    v3 = np.full(128, np.sqrt(0.64 / 128))  # dist = 0.8
    samples = [Mau(1, v3), Mau(1, v1), Mau(1, v2)]

    res = tim_danh_tinh_tot_nhat(
        query, samples, distance_threshold=0.5, identity_margin=0.0, top_k=2
    )
    # Top 2 samples gần nhất là v1 (0.2) và v2 (0.4) -> trung bình = 0.3
    assert res.mau is not None
    assert pytest.approx(res.distance, abs=1e-4) == 0.30
