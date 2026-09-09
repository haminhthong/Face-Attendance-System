from datetime import timedelta

import numpy as np

from face_attendance import config, database
from face_attendance.utils import utc_iso, utc_now


def tao_du_lieu_buoi_hoc(tmp_path, monkeypatch) -> tuple[int, int]:
    """Tạo dữ liệu tối thiểu cho các test nghiệp vụ điểm danh."""
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    student = database.upsert_student("SV001", "Nguyễn Văn An", "K23")
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
    return student_id, session_id


def test_attendance_is_created_once(tmp_path, monkeypatch) -> None:
    student_id, session_id = tao_du_lieu_buoi_hoc(tmp_path, monkeypatch)
    database.change_session_status(session_id, "open")

    result, _ = database.mark_attendance(session_id, student_id, 0.4)
    duplicate_result, _ = database.mark_attendance(session_id, student_id, 0.4)

    assert result == "created"
    assert duplicate_result == "already"


def test_session_roster_is_a_snapshot(tmp_path, monkeypatch) -> None:
    _, session_id = tao_du_lieu_buoi_hoc(tmp_path, monkeypatch)
    course_id = int(database.list_courses()[0]["id"])
    database.set_course_roster(course_id, [])

    report = database.attendance_report(session_id)

    assert report["MSSV"].tolist() == ["SV001"]
    assert report["Trạng thái"].tolist() == ["Vắng"]


def test_expired_embedding_is_removed(tmp_path, monkeypatch) -> None:
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()
    student = database.upsert_student("SV001", "Nguyễn Văn An", "K23")
    student_id = int(student["id"])
    database.save_embedding(
        student_id,
        np.zeros(128),
        "image-hash",
        100.0,
        120.0,
        150,
        150,
    )
    with database.get_connection() as connection:
        connection.execute(
            "UPDATE face_embeddings SET created_at_utc = ?",
            (utc_iso(utc_now() - timedelta(days=10)),),
        )

    deleted = database.purge_expired_biometrics(retention_days=5)

    assert deleted == 1
    with database.get_connection() as connection:
        row = connection.execute(
            "SELECT active, consent_status FROM students WHERE id = ?", (student_id,)
        ).fetchone()
    assert row["active"] == 1
    assert row["consent_status"] == "pending"


def test_attendance_report_handles_absent_students(tmp_path, monkeypatch) -> None:
    """Kiểm tra báo cáo điểm danh xử lý chính xác cả sinh viên có mặt lẫn sinh viên vắng mặt mà không gây lỗi kiểu dữ liệu."""
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()
    st1 = database.upsert_student("SV001", "Nguyễn Văn An", "K23")
    st2 = database.upsert_student("SV002", "Trần Thị Bích", "K23")
    st1_id, st2_id = int(st1["id"]), int(st2["id"])

    database.create_course("CS101", "Nhập môn lập trình", "Giảng viên A")
    course_id = int(database.list_courses()[0]["id"])
    database.set_course_roster(course_id, [st1_id, st2_id])

    now = utc_now()
    database.create_attendance_session(
        course_id, "Buổi 1", now - timedelta(minutes=5), now + timedelta(minutes=30), 15
    )
    session_id = int(database.list_sessions()[0]["id"])
    database.change_session_status(session_id, "open")

    # Chỉ điểm danh SV1, SV2 vắng mặt
    database.mark_attendance(session_id, st1_id, 0.35)

    report = database.attendance_report(session_id)
    assert len(report) == 2
    statuses = dict(zip(report["MSSV"], report["Trạng thái"], strict=True))
    assert statuses["SV001"] == "Có mặt"
    assert statuses["SV002"] == "Vắng"
