"""Kiểm thử luồng RecognitionDecision và điều chỉnh điểm danh thủ công."""

from datetime import timedelta

import pytest

from face_attendance import config, database
from face_attendance.application.attendance_service import (
    record_biometric_attendance,
    record_manual_attendance,
)
from face_attendance.domain import (
    DuplicateAttendanceError,
    RecognitionDecision,
    RejectionReason,
    SessionClosedError,
)
from face_attendance.utils import utc_iso, utc_now


@pytest.fixture
def setup_attendance_env(tmp_path, monkeypatch):
    test_db = tmp_path / "test_flow.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    student = database.upsert_student(
        "SV001",
        "Nguyễn Văn An",
        "K23",
        consent_given=True,
        consent_policy_version="face-policy-v1",
    )
    student_id = int(student["id"])
    database.create_course("CS101", "Nhập môn lập trình", "Trần Văn Bình")
    course_id = int(database.list_courses()[0]["id"])
    database.set_course_roster(course_id, [student_id])

    now = utc_now()
    database.create_attendance_session(
        course_id,
        "Buổi 1",
        now - timedelta(minutes=5),
        now + timedelta(minutes=30),
        15,
    )
    session_id = int(database.list_sessions()[0]["id"])
    database.change_session_status(session_id, "open")
    return student_id, session_id


def test_biometric_attendance_records_all_evidence(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    res = record_biometric_attendance(session_id, decision)
    assert res.is_accepted is True
    assert res.status == "present"
    assert res.student_id == str(student_id)
    assert res.margin == 0.23

    # Kiểm tra bằng chứng lưu trữ trong cơ sở dữ liệu
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()

        assert row is not None
        assert row["source"] == "face_webrtc"
        assert pytest.approx(row["recognition_distance"], abs=1e-4) == 0.32
        assert pytest.approx(row["identity_margin"], abs=1e-4) == 0.23
        assert row["liveness_policy"] == "ear_blink_v1"
        assert row["recognition_policy_version"] == "face-policy-v1"
        assert row["confirmation_frames"] == 5


def test_biometric_attendance_duplicate_raises_error(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    record_biometric_attendance(session_id, decision)
    with pytest.raises(DuplicateAttendanceError):
        record_biometric_attendance(session_id, decision)


def test_biometric_attendance_liveness_rejected(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=False,
        confirmation_frames=2,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    res = record_biometric_attendance(session_id, decision)
    assert res.is_accepted is False
    assert res.rejection_reason == RejectionReason.LIVENESS_FAILED


def test_biometric_attendance_session_closed(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env
    database.change_session_status(session_id, "closed")

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    with pytest.raises(SessionClosedError):
        record_biometric_attendance(session_id, decision)


def test_biometric_attendance_consent_revoked(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env
    database.revoke_student_consent(student_id)

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    res = record_biometric_attendance(session_id, decision)
    assert res.is_accepted is False
    assert res.rejection_reason == RejectionReason.NO_CONSENT


def test_biometric_attendance_requires_granted_consent(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env
    with database.get_connection() as connection:
        connection.execute(
            """
            UPDATE students
            SET consent_status = 'pending', consent_at_utc = NULL
            WHERE id = ?
            """,
            (student_id,),
        )

    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )

    res = record_biometric_attendance(session_id, decision)
    assert res.is_accepted is False
    assert res.rejection_reason == RejectionReason.NO_CONSENT


def test_biometric_attendance_rejects_policy_mismatch(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env
    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        distance_threshold=0.60,
        timestamp_utc=utc_iso(),
    )

    res = record_biometric_attendance(session_id, decision)
    assert res.is_accepted is False
    assert res.rejection_reason == RejectionReason.POLICY_MISMATCH


def test_manual_attendance_correction_audit(setup_attendance_env) -> None:
    student_id, session_id = setup_attendance_env

    # 1. Điểm danh sinh viên bằng sinh trắc trước
    decision = RecognitionDecision(
        student_id=student_id,
        student_code="SV001",
        full_name="Nguyễn Văn An",
        distance=0.32,
        second_distance=0.55,
        margin=0.23,
        liveness_passed=True,
        confirmation_frames=5,
        policy_version="face-policy-v1",
        timestamp_utc=utc_iso(),
    )
    record_biometric_attendance(session_id, decision)

    # 2. Giảng viên can thiệp thủ công có audit: đổi trạng thái thành vắng.
    res_corr = record_manual_attendance(
        session_id=session_id,
        student_id=student_id,
        status="absent",
        lecturer_id="GV_HOANG",
        reason="Sinh viên vi phạm quy chế thi",
    )
    assert res_corr.is_accepted is True
    assert res_corr.status == "absent"
    assert res_corr.source == "manual"

    # 3. Kiểm tra nhật ký kiểm toán lưu trong DB
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()

        assert row["attendance_status"] == "absent"
        assert row["original_status"] == "present"
        assert row["final_status"] == "absent"
        assert row["corrected_by"] == "GV_HOANG"
        assert row["correction_reason"] == "Sinh viên vi phạm quy chế thi"
        assert row["corrected_at_utc"] is not None
