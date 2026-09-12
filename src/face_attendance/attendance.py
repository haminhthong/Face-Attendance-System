"""Module xử lý nghiệp vụ điểm danh sinh viên (Attendance Business Service).

Chịu trách nhiệm:
1. Xác thực điều kiện buổi học (buổi học đang mở, sinh viên nằm trong snapshot danh sách buổi học).
2. Chống điểm danh trùng lặp trong cùng một buổi học.
3. Phân loại trạng thái 'Có mặt' (present) hoặc 'Đi trễ' (late) dựa trên thời gian bắt đầu buổi học.
4. Cho phép giảng viên điều chỉnh trạng thái điểm danh thủ công (Manual Override) kèm lý do.
"""

from __future__ import annotations

from dataclasses import dataclass

from .database import (
    get_connection,
    get_session,
    manual_attendance_correction,
    mark_attendance,
)


class AttendanceError(Exception):
    """Lỗi nghiệp vụ điểm danh chung."""

    pass


class SessionClosedError(AttendanceError):
    """Buổi học đã đóng hoặc chưa mở."""

    pass


class StudentNotInRosterError(AttendanceError):
    """Sinh viên không thuộc danh sách buổi học."""

    pass


class DuplicateAttendanceError(AttendanceError):
    """Sinh viên đã được điểm danh trong buổi học này."""

    pass


@dataclass(frozen=True)
class AttendanceRecordResult:
    """Kết quả ghi nhận điểm danh."""

    session_id: int
    student_id: int
    status: str
    distance: float
    margin: float
    is_duplicate: bool = False
    message: str = ""

    @property
    def is_accepted(self) -> bool:
        """Kiểm tra điểm danh thành công hay không."""
        return not self.is_duplicate and self.status in {"present", "late"}


def record_biometric_attendance(
    session_id: int,
    student_id: int,
    distance: float,
    margin: float = 0.0,
    **kwargs,
) -> AttendanceRecordResult:
    """Ghi nhận điểm danh sinh trắc học từ luồng camera.

    Args:
        session_id: ID của buổi học.
        student_id: ID của sinh viên được nhận diện.
        distance: Khoảng cách L2 Euclidean từ vector khuôn mặt đến template.
        margin: Chênh lệch khoảng cách giữa Top-1 và Top-2.

    Returns:
        AttendanceRecordResult: Kết quả ghi nhận điểm danh.

    Raises:
        SessionClosedError: Nếu buổi học chưa mở hoặc đã kết thúc.
        StudentNotInRosterError: Nếu sinh viên không thuộc danh sách buổi học.
        DuplicateAttendanceError: Nếu sinh viên đã được ghi nhận trước đó.
    """
    session = get_session(session_id)
    if not session:
        raise AttendanceError(f"Không tìm thấy buổi học #{session_id}")
    if session["status"] != "open":
        raise SessionClosedError(
            f"Buổi học #{session_id} hiện đang ở trạng thái '{session['status']}'."
        )

    with get_connection() as conn:
        in_roster = conn.execute(
            "SELECT 1 FROM session_enrollments WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if not in_roster:
            raise StudentNotInRosterError("Sinh viên không thuộc danh sách lớp của buổi học này.")

    result_code, status = mark_attendance(
        session_id=session_id,
        student_id=student_id,
        distance=distance,
        margin=margin,
        source="face_webrtc",
    )

    if result_code == "already":
        raise DuplicateAttendanceError(f"Sinh viên đã được điểm danh trước đó ({status}).")

    return AttendanceRecordResult(
        session_id=session_id,
        student_id=student_id,
        status=status,
        distance=distance,
        margin=margin,
        message="Điểm danh thành công.",
    )


def record_manual_attendance(
    session_id: int,
    student_id: int,
    status: str,
    lecturer_id: str,
    reason: str,
) -> AttendanceRecordResult:
    """Giảng viên can thiệp hoặc sửa đổi điểm danh thủ công."""
    session = get_session(session_id)
    if not session:
        raise AttendanceError(f"Không tìm thấy buổi học #{session_id}")

    manual_attendance_correction(
        session_id=session_id,
        student_id=student_id,
        status=status,
        lecturer_name=lecturer_id,
        reason=reason,
    )

    return AttendanceRecordResult(
        session_id=session_id,
        student_id=student_id,
        status=status,
        distance=0.0,
        margin=0.0,
        message=f"Đã điều chỉnh thủ công bởi {lecturer_id}.",
    )
