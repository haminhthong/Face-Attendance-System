"""Module định nghĩa các kiểu liệt kê (Enums) domain cho hệ thống điểm danh."""

from __future__ import annotations

from enum import Enum


class AttendanceStatus(str, Enum):
    """Trạng thái điểm danh sinh viên."""

    PRESENT = "present"
    LATE = "late"
    ABSENT = "absent"


class AttendanceDecision(str, Enum):
    """Quyết định của hệ thống nhận diện AI."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Lý do từ chối điểm danh."""

    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOW_IMAGE_QUALITY = "low_image_quality"
    UNKNOWN_FACE = "unknown_face"
    AMBIGUOUS_MATCH = "ambiguous_match"
    LIVENESS_FAILED = "liveness_failed"
    OUTSIDE_ATTENDANCE_WINDOW = "outside_attendance_window"
    ALREADY_ATTENDED = "already_attended"
    NOT_IN_ROSTER = "not_in_roster"
    NO_CONSENT = "no_consent"
    UNSTABLE_TRACKING = "unstable_tracking"


class ConfidenceLevel(str, Enum):
    """Mức độ tin cậy định tính của khoảng cách nhận diện."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MatchQuality(str, Enum):
    """Phân loại dải khoảng cách so khớp (Distance band/Match quality)."""

    STRONG_MATCH = "strong_match"
    BORDERLINE_MATCH = "borderline_match"
    WEAK_MATCH = "weak_match"


def get_match_quality(distance: float | None, tolerance: float = 0.50) -> MatchQuality:
    """Xác định dải chất lượng so khớp (tránh gây hiểu lầm là xác suất calibrated)."""
    if distance is None or distance > tolerance:
        return MatchQuality.WEAK_MATCH
    if distance <= tolerance * 0.75:
        return MatchQuality.STRONG_MATCH
    return MatchQuality.BORDERLINE_MATCH


def get_confidence_level(distance: float | None, tolerance: float = 0.50) -> ConfidenceLevel:
    """Xác định mức độ tin cậy dựa trên khoảng cách Euclidean L2.

    Args:
        distance: Khoảng cách Euclidean L2.
        tolerance: Ngưỡng khoảng cách tối đa chấp nhận.

    Returns:
        ConfidenceLevel: Mức độ tin cậy 'high', 'medium', hoặc 'low'.
    """
    if distance is None or distance > tolerance:
        return ConfidenceLevel.LOW
    if distance <= tolerance * 0.75:  # Ví dụ <= 0.375 với tolerance 0.50
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.MEDIUM

