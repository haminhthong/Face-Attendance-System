"""Unit tests cho các quy tắc nghiệp vụ điểm danh sinh viên."""

from datetime import timedelta

import pytest

from face_attendance import config, database
from face_attendance.attendance import (
    DuplicateAttendanceError,
    SessionClosedError,
    StudentNotInRosterError,
    record_biometric_attendance,
    record_manual_attendance,
)
from face_attendance.utils import utc_now


@pytest.fixture
def setup_session(tmp_path, monkeypatch):
    test_db = tmp_path / "test_attendance.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    database.init_database()

    st1 = database.upsert_student("SV001", "Nguyen Van An", "23DTH01")
    st2 = database.upsert_student("SV002", "Tran Thi Bich", "23DTH01")
    s1_id, s2_id = int(st1["id"]), int(st2["id"])

    course = database.create_course("CS101", "Lap trinh Python", "GV Nguyen")
    c_id = int(course["id"])
    database.set_course_roster(c_id, [s1_id, s2_id])

    now = utc_now()
    session = database.create_attendance_session(
        c_id,
        "Buoi 1",
        now - timedelta(minutes=5),
        now + timedelta(minutes=60),
        late_after_minutes=15,
    )
    sess_id = int(session["id"])
    database.change_session_status(sess_id, "open")

    return sess_id, s1_id, s2_id


def test_biometric_attendance_success(setup_session) -> None:
    session_id, s1_id, _ = setup_session
    res = record_biometric_attendance(session_id, s1_id, distance=0.35, margin=0.15)
    assert res.is_accepted is True
    assert res.status == "present"
    assert res.student_id == s1_id


def test_biometric_attendance_duplicate_raises_error(setup_session) -> None:
    session_id, s1_id, _ = setup_session
    record_biometric_attendance(session_id, s1_id, distance=0.35, margin=0.15)

    with pytest.raises(DuplicateAttendanceError, match="đã được điểm danh"):
        record_biometric_attendance(session_id, s1_id, distance=0.36, margin=0.14)


def test_biometric_attendance_session_closed(setup_session) -> None:
    session_id, s1_id, _ = setup_session
    database.change_session_status(session_id, "closed")

    with pytest.raises(SessionClosedError, match="trạng thái 'closed'"):
        record_biometric_attendance(session_id, s1_id, distance=0.35, margin=0.15)


def test_biometric_attendance_student_not_in_roster(setup_session) -> None:
    session_id, _, _ = setup_session
    st_outsider = database.upsert_student("SV999", "Người Ngoài Lớp", "K23")
    outsider_id = int(st_outsider["id"])

    with pytest.raises(StudentNotInRosterError, match="không thuộc danh sách lớp"):
        record_biometric_attendance(session_id, outsider_id, distance=0.35, margin=0.15)


def test_biometric_attendance_late_status(tmp_path, monkeypatch) -> None:
    test_db = tmp_path / "test_late.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    database.init_database()

    st1 = database.upsert_student("SV001", "Nguyen Van An", "23DTH01")
    s1_id = int(st1["id"])
    course = database.create_course("CS101", "Lap trinh", "GV")
    c_id = int(course["id"])
    database.set_course_roster(c_id, [s1_id])

    # Buổi học đã bắt đầu từ 30 phút trước, giới hạn trễ là 15 phút -> Late
    now = utc_now()
    session = database.create_attendance_session(
        c_id,
        "Buoi 1",
        now - timedelta(minutes=30),
        now + timedelta(minutes=60),
        late_after_minutes=15,
    )
    sess_id = int(session["id"])
    database.change_session_status(sess_id, "open")

    res = record_biometric_attendance(sess_id, s1_id, distance=0.38, margin=0.12)
    assert res.status == "late"


def test_manual_attendance_correction(setup_session) -> None:
    session_id, s1_id, _ = setup_session
    # Giảng viên điều chỉnh trạng thái sinh viên sang Đi trễ kèm lý do
    res = record_manual_attendance(
        session_id=session_id,
        student_id=s1_id,
        status="late",
        lecturer_id="TS. Tran Van Binh",
        reason="Camera chập chờn, đã kiểm tra trực tiếp qua thẻ SV",
    )
    assert res.status == "late"

    report = database.attendance_report(session_id)
    s1_row = report[report["MSSV"] == "SV001"].iloc[0]
    assert s1_row["Trạng thái"] == "Đi trễ"
    assert "TS. Tran Van Binh" in s1_row["Ghi chú"]
