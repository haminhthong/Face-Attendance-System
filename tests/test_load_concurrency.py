"""Load test và concurrency benchmark kiểm thử khả năng đáp ứng nhiều request đồng thời."""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np

from face_attendance import config, database
from face_attendance.utils import utc_now


def setup_benchmark_db(tmp_path, monkeypatch, num_students: int = 50) -> tuple[int, list[int]]:
    """Khởi tạo cơ sở dữ liệu với num_students sinh viên và 1 buổi học đang mở."""
    test_db = tmp_path / "load_test.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    student_ids: list[int] = []
    for i in range(1, num_students + 1):
        st = database.upsert_student(f"SV{i:03d}", f"Sinh Vien {i}", "Lop 12A")
        s_id = int(st["id"])
        database.save_embedding(s_id, np.zeros(128), f"hash_{i}", 100.0, 100.0, 150, 150)
        student_ids.append(s_id)

    database.create_course("CS101", "Lap Trinh", "Giang Vien A")
    course_id = int(database.list_courses()[0]["id"])
    database.set_course_roster(course_id, student_ids)

    now = utc_now()
    database.create_attendance_session(
        course_id, "Buoi Load Test", now - timedelta(minutes=5), now + timedelta(minutes=60), 15
    )
    session_id = int(database.list_sessions()[0]["id"])
    database.change_session_status(session_id, "open")

    return session_id, student_ids


def test_concurrent_attendance_requests_thread_safety(tmp_path, monkeypatch) -> None:
    """Kiểm tra 50 requests đồng thời ghi nhận điểm danh cho các sinh viên khác nhau."""
    session_id, student_ids = setup_benchmark_db(tmp_path, monkeypatch, num_students=50)

    def submit_attendance(s_id: int):
        return database.mark_attendance(session_id, s_id, distance=0.38)

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(submit_attendance, student_ids))
    elapsed = time.perf_counter() - start_time

    created_count = sum(1 for res, _ in results if res == "created")
    assert created_count == 50
    avg_latency_ms = (elapsed / 50.0) * 1000.0
    print(
        f"\n[LOAD TEST 50 CONCURRENT REQUESTS] Elapsed: {elapsed:.2f}s, Avg Latency: {avg_latency_ms:.2f}ms/request"
    )


def test_concurrent_duplicate_attendance_lock(tmp_path, monkeypatch) -> None:
    """Kiểm tra 10 requests đồng thời cho CÙNG 1 sinh viên: duy nhất 1 request 'created', 9 request 'already'."""
    session_id, student_ids = setup_benchmark_db(tmp_path, monkeypatch, num_students=5)
    target_student_id = student_ids[0]

    def submit_same_student(_):
        return database.mark_attendance(session_id, target_student_id, distance=0.35)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(submit_same_student, range(10)))

    status_codes = [res for res, _ in results]
    created_count = status_codes.count("created")
    already_count = status_codes.count("already")

    assert created_count == 1
    assert already_count == 9
