"""Domain package cho hệ thống điểm danh khuôn mặt."""

from .entities import (
    AttendanceResult,
    RecognitionDecision,
    build_accepted_result,
    build_rejected_result,
)
from .enums import (
    AttendanceDecision,
    AttendanceStatus,
    ConfidenceLevel,
    MatchQuality,
    RejectionReason,
    get_confidence_level,
    get_match_quality,
)
from .exceptions import (
    AttendanceError,
    BiometricConsentMissingError,
    DuplicateAttendanceError,
    FaceNotFoundError,
    InvalidImageError,
    MultipleFacesError,
    SessionClosedError,
    StudentNotInRosterError,
    UnknownFaceError,
)

__all__ = [
    "AttendanceResult",
    "RecognitionDecision",
    "build_accepted_result",
    "build_rejected_result",
    "AttendanceStatus",
    "AttendanceDecision",
    "RejectionReason",
    "ConfidenceLevel",
    "MatchQuality",
    "get_confidence_level",
    "get_match_quality",
    "AttendanceError",
    "StudentNotInRosterError",
    "DuplicateAttendanceError",
    "SessionClosedError",
    "BiometricConsentMissingError",
    "InvalidImageError",
    "FaceNotFoundError",
    "MultipleFacesError",
    "UnknownFaceError",
]
