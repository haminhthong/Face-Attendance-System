"""Unit tests cho domain package."""

from face_attendance.domain import (
    AttendanceDecision,
    AttendanceStatus,
    ConfidenceLevel,
    RejectionReason,
    build_accepted_result,
    build_rejected_result,
)


def test_build_accepted_result():
    result = build_accepted_result(
        student_id="SV001",
        status="present",
        distance=0.35,
        liveness_passed=True,
        recognized_at="2026-09-01T08:15:00+07:00",
        tolerance=0.50,
    )
    assert result.decision == AttendanceDecision.ACCEPTED
    assert result.confidence_level == ConfidenceLevel.HIGH
    assert result.rejection_reason is None
    assert result.to_dict()["decision"] == "accepted"


def test_build_rejected_result():
    result = build_rejected_result(
        reason=RejectionReason.UNKNOWN_FACE,
        recognized_at="2026-09-01T08:15:00+07:00",
        student_id=None,
    )
    assert result.decision == AttendanceDecision.REJECTED
    assert result.rejection_reason == RejectionReason.UNKNOWN_FACE
    assert result.status == AttendanceStatus.ABSENT
    assert result.to_dict()["rejection_reason"] == "unknown_face"
