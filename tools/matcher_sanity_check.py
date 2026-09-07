"""Điểm vào chính thức cho synthetic matcher sanity check.

Không tool nào trong file này được phép tạo ``recognition_policy.json``.
Policy runtime chỉ được tạo bởi evaluate_biometrics.py sau khi có private
validation dataset hợp lệ.
"""

from __future__ import annotations  # noqa: I001

from evaluate_matching import run_evaluation


if __name__ == "__main__":
    run_evaluation()
