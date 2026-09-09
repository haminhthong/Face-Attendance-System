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
    # L2 distance between zeros(128) and vec is sqrt(128 * 0.001953125) = sqrt(0.25) = 0.50
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
