"""Application layer package cho hệ thống điểm danh."""

from .attendance_service import (
    record_biometric_attendance,
    record_manual_attendance,
)
from .enrollment_service import process_student_enrollment

__all__ = [
    "record_biometric_attendance",
    "record_manual_attendance",
    "process_student_enrollment",
]
