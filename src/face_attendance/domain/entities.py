"""Module định nghĩa các đối tượng dữ liệu (Entities / Schemas) domain cho hệ thống điểm danh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .enums import (
    AttendanceDecision,
    AttendanceStatus,
    ConfidenceLevel,
    MatchQuality,
    RejectionReason,
    get_confidence_level,
    get_match_quality,
)


@dataclass(frozen=True)
class RecognitionDecision:
    """Đối tượng truyền tải quyết định nhận diện từ Vision Pipeline sang Application Service.

    Mang đầy đủ bằng chứng kiểm định (distance, margin, liveness, confirmation frames).
    """

    student_id: int
    student_code: str
    full_name: str
    distance: float
    second_distance: float
    margin: float
    liveness_passed: bool
    confirmation_frames: int
    policy_version: str = "face-policy-v1"
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi sang dict định dạng JSON thân thiện."""
        return asdict(self)


@dataclass(frozen=True)
class AttendanceResult:
    """Cấu trúc kết quả điểm danh chuẩn hóa của hệ thống.

    Attributes:
        student_id: Mã sinh viên (hoặc ID sinh viên dạng chuỗi/số).
        status: Trạng thái điểm danh ('present', 'late', 'absent').
        distance: Khoảng cách Euclidean L2 giữa vector nhận diện và mẫu (None nếu không khớp).
        confidence_level: Mức độ tin cậy định tính ('high', 'medium', 'low').
        liveness_passed: Kết quả kiểm tra tương tác/chớp mắt.
        recognized_at: Thời điểm nhận diện theo chuẩn ISO 8601 UTC/VN.
        decision: Quyết định điểm danh ('accepted' hoặc 'rejected').
        rejection_reason: Lý do từ chối nếu decision == 'rejected'.
        margin: Độ chênh lệch giữa Top-1 và Top-2 (nếu có).
        match_quality: Phân loại dải khoảng cách ('strong_match', 'borderline_match', 'weak_match').
        source: Nguồn gốc điểm danh ('face_webrtc', 'manual', etc.).
        policy_version: Phiên bản chính sách nhận diện được áp dụng.
    """

    student_id: str | None
    status: AttendanceStatus | str
    distance: float | None
    confidence_level: ConfidenceLevel | str
    liveness_passed: bool
    recognized_at: str
    decision: AttendanceDecision | str
    rejection_reason: RejectionReason | str | None = None
    margin: float | None = None
    match_quality: MatchQuality | str | None = None
    source: str = "face_webrtc"
    policy_version: str = "face-policy-v1"

    @property
    def is_accepted(self) -> bool:
        """Kiểm tra xem kết quả có phải được chấp nhận (ACCEPTED) hay không."""
        val = self.decision.value if hasattr(self.decision, "value") else str(self.decision)
        return val.lower() == "accepted"

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi sang dict định dạng JSON thân thiện."""
        data = asdict(self)
        if isinstance(self.status, Enum_or_str):
            data["status"] = str(self.status.value if hasattr(self.status, "value") else self.status)
        if isinstance(self.decision, Enum_or_str):
            data["decision"] = str(self.decision.value if hasattr(self.decision, "value") else self.decision)
        if isinstance(self.confidence_level, Enum_or_str):
            data["confidence_level"] = str(
                self.confidence_level.value if hasattr(self.confidence_level, "value") else self.confidence_level
            )
        if isinstance(self.match_quality, Enum_or_str):
            data["match_quality"] = str(
                self.match_quality.value if hasattr(self.match_quality, "value") else self.match_quality
            )
        if self.rejection_reason is not None and hasattr(self.rejection_reason, "value"):
            data["rejection_reason"] = self.rejection_reason.value
        return data


Enum_or_str = (AttendanceStatus, AttendanceDecision, ConfidenceLevel, MatchQuality, RejectionReason)


def build_accepted_result(
    student_id: str,
    status: AttendanceStatus | str,
    distance: float,
    liveness_passed: bool,
    recognized_at: str,
    tolerance: float = 0.50,
    margin: float | None = None,
    source: str = "face_webrtc",
    policy_version: str = "face-policy-v1",
) -> AttendanceResult:
    """Tạo kết quả chấp nhận điểm danh chuẩn hóa.

    Args:
        student_id: Mã sinh viên.
        status: Trạng thái điểm danh ('present' hoặc 'late').
        distance: Khoảng cách Euclidean L2.
        liveness_passed: Trạng thái liveness.
        recognized_at: Chuỗi ISO 8601 thời gian nhận diện.
        tolerance: Ngưỡng tối đa để phân loại confidence level.
        margin: Độ chênh lệch giữa Top-1 và Top-2.
        source: Nguồn gốc điểm danh.
        policy_version: Phiên bản nhận diện.

    Returns:
        AttendanceResult: Đối tượng kết quả điểm danh được chấp nhận.
    """
    conf = get_confidence_level(distance, tolerance)
    quality = get_match_quality(distance, tolerance)
    return AttendanceResult(
        student_id=student_id,
        status=status if isinstance(status, AttendanceStatus) else AttendanceStatus(status),
        distance=round(distance, 4),
        confidence_level=conf,
        liveness_passed=liveness_passed,
        recognized_at=recognized_at,
        decision=AttendanceDecision.ACCEPTED,
        rejection_reason=None,
        margin=round(margin, 4) if margin is not None else None,
        match_quality=quality,
        source=source,
        policy_version=policy_version,
    )


def build_rejected_result(
    reason: RejectionReason | str,
    recognized_at: str,
    student_id: str | None = None,
    distance: float | None = None,
    liveness_passed: bool = False,
    tolerance: float = 0.50,
    margin: float | None = None,
    source: str = "face_webrtc",
    policy_version: str = "face-policy-v1",
) -> AttendanceResult:
    """Tạo kết quả từ chối điểm danh chuẩn hóa với lý do rõ ràng.

    Args:
        reason: Lý do từ chối.
        recognized_at: Chuỗi ISO 8601 thời gian.
        student_id: Mã sinh viên (nếu xác định được nhưng bị từ chối).
        distance: Khoảng cách Euclidean L2 nếu có.
        liveness_passed: Kết quả liveness.
        tolerance: Ngưỡng khoảng cách.
        margin: Chênh lệch margin nếu có.
        source: Nguồn gốc.
        policy_version: Phiên bản nhận diện.

    Returns:
        AttendanceResult: Đối tượng kết quả từ chối.
    """
    reason_enum = reason if isinstance(reason, RejectionReason) else RejectionReason(reason)
    conf = get_confidence_level(distance, tolerance) if distance is not None else ConfidenceLevel.LOW
    quality = get_match_quality(distance, tolerance) if distance is not None else MatchQuality.WEAK_MATCH
    return AttendanceResult(
        student_id=student_id,
        status=AttendanceStatus.ABSENT,
        distance=round(distance, 4) if distance is not None else None,
        confidence_level=conf,
        liveness_passed=liveness_passed,
        recognized_at=recognized_at,
        decision=AttendanceDecision.REJECTED,
        rejection_reason=reason_enum,
        margin=round(margin, 4) if margin is not None else None,
        match_quality=quality,
        source=source,
        policy_version=policy_version,
    )

