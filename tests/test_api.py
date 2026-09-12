"""Unit tests cho FastAPI endpoints."""

from datetime import timedelta

import numpy as np
from fastapi.testclient import TestClient

from face_attendance import api, config, database
from face_attendance.utils import utc_now

app = api.app


def test_health() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_recognize_endpoint_success(tmp_path, monkeypatch) -> None:
    test_db = tmp_path / "test_api_rec.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    st = database.upsert_student("SV001", "Nguyen Van An", "23DTH01")
    s_id = int(st["id"])
    database.save_embedding(s_id, np.zeros(128))

    query_vec = [0.0] * 128
    with TestClient(app) as client:
        res = client.post("/recognize", json={"embedding": query_vec})

    assert res.status_code == 200
    data = res.json()
    assert data["matched"] is True
    assert data["student_code"] == "SV001"
    assert data["distance"] == 0.0
    assert data["status"] == "matched"


def test_recognize_endpoint_unknown(tmp_path, monkeypatch) -> None:
    test_db = tmp_path / "test_api_unknown.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    st = database.upsert_student("SV001", "Nguyen Van An", "23DTH01")
    s_id = int(st["id"])
    database.save_embedding(s_id, np.zeros(128))

    # Vector xa (dist > 0.50) -> Unknown
    query_vec = [1.0] * 128
    with TestClient(app) as client:
        res = client.post("/recognize", json={"embedding": query_vec})

    assert res.status_code == 200
    data = res.json()
    assert data["matched"] is False
    assert data["status"] == "unknown"


def test_recognize_endpoint_invalid_dim() -> None:
    with TestClient(app) as client:
        res = client.post("/recognize", json={"embedding": [0.0] * 100})
    assert res.status_code == 422


def test_session_attendance_report_endpoint(tmp_path, monkeypatch) -> None:
    test_db = tmp_path / "test_api_report.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_database()

    st = database.upsert_student("SV001", "Nguyen Van An", "23DTH01")
    s_id = int(st["id"])
    course = database.create_course("CS101", "Lap trinh", "GV A")
    c_id = int(course["id"])
    database.set_course_roster(c_id, [s_id])

    now = utc_now()
    session = database.create_attendance_session(
        c_id, "Buoi 1", now - timedelta(minutes=5), now + timedelta(minutes=30), 15
    )
    sess_id = int(session["id"])
    database.change_session_status(sess_id, "open")
    database.mark_attendance(sess_id, s_id, 0.35)

    with TestClient(app) as client:
        res = client.get(f"/sessions/{sess_id}/attendance")

    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["MSSV"] == "SV001"
    assert data[0]["Trạng thái"] == "Có mặt"
