from datetime import timedelta

import numpy as np
from fastapi.testclient import TestClient

from face_attendance import api, config, database
from face_attendance.utils import utc_now

app = api.app


def biometric_payload(session_id: int, student_id: int) -> dict[str, object]:
    policy = config.RECOGNITION_POLICY
    return {
        "session_id": session_id,
        "student_id": student_id,
        "student_code": "SV001",
        "full_name": "Nguyen Van A",
        "distance": 0.35,
        "second_distance": 0.60,
        "margin": 0.25,
        "liveness_passed": True,
        "confirmation_frames": policy.minimum_observations,
        "policy_version": policy.policy_version,
        "distance_threshold": policy.distance_threshold,
        "margin_threshold": policy.identity_margin,
        "aggregation_strategy": policy.aggregation_strategy,
        "embedding_model": policy.embedding_model,
        "embedding_model_version": policy.embedding_model_version,
        "stable_duration_ms": policy.stable_duration_ms,
        "liveness_policy": policy.liveness_policy,
        "recognition_policy_hash": policy.policy_hash,
    }


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_attendance_write_is_disabled_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(api, "API_KEY", "")
    with TestClient(app) as client:
        response = client.get("/policy")
    assert response.status_code == 503


def test_sessions_requires_valid_api_key(monkeypatch) -> None:
    monkeypatch.setattr(api, "API_KEY", "secret-key")
    with TestClient(app) as client:
        unauthorized = client.get("/sessions")
        invalid_key = client.get("/sessions", headers={"X-API-Key": "wrong-key"})
        authorized = client.get("/sessions", headers={"X-API-Key": "secret-key"})
    assert unauthorized.status_code == 401
    assert invalid_key.status_code == 401
    assert authorized.status_code == 200


def test_attendance_payload_validation(monkeypatch) -> None:
    monkeypatch.setattr(api, "API_KEY", "secret-key")
    with TestClient(app) as client:
        # Thiếu toàn bộ bằng chứng bắt buộc của pipeline biometric.
        res1 = client.post(
            "/attendance/biometric",
            headers={"X-API-Key": "secret-key"},
            json={"student_id": 1},
        )
        # Sai kiểu dữ liệu student_id.
        res2 = client.post(
            "/attendance/biometric",
            headers={"X-API-Key": "secret-key"},
            json={**biometric_payload(1, 1), "student_id": "not-an-int"},
        )
        res3 = client.post(
            "/attendance/biometric",
            headers={"X-API-Key": "secret-key"},
            json={**biometric_payload(1, 1), "margin": 0.01},
        )
    assert res1.status_code == 422
    assert res2.status_code == 422
    assert res3.status_code == 422


def test_attendance_success_structured_response(monkeypatch, tmp_path) -> None:
    test_db = tmp_path / "test_api.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    monkeypatch.setattr(api, "API_KEY", "secret-key")
    database.init_database()
    student = database.upsert_student(
        "SV001",
        "Nguyen Van A",
        "12A1",
        consent_given=True,
        consent_policy_version="biometric-consent-v1",
    )
    student_id = int(student["id"])
    database.save_embedding(student_id, np.zeros(128), "hash_img", 100.0, 100.0, 150, 150)
    database.create_course("MAT101", "Toan Co So", "Giang Vien A")
    course_id = int(database.list_courses()[0]["id"])
    database.set_course_roster(course_id, [student_id])

    now = utc_now()
    database.create_attendance_session(
        course_id, "Buoi 1", now - timedelta(minutes=5), now + timedelta(minutes=30), 15
    )
    session_id = int(database.list_sessions()[0]["id"])
    database.change_session_status(session_id, "open")

    with TestClient(app) as client:
        res = client.post(
            "/attendance/biometric",
            headers={"X-API-Key": "secret-key"},
            json=biometric_payload(session_id, student_id),
        )
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] == "accepted"
    assert data["student_id"] == str(student_id)
    assert data["distance"] == 0.35
    assert data["confidence_level"] == "high"
    assert data["liveness_passed"] is True


def test_unhandled_exception_does_not_leak_stacktrace(monkeypatch) -> None:
    monkeypatch.setattr(api, "API_KEY", "secret-key")

    def mock_broken_service(*args, **kwargs):
        raise RuntimeError("Internal DB crashed on line 123 in /var/internal/db.py")

    monkeypatch.setattr(api, "record_biometric_attendance", mock_broken_service)

    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.post(
            "/attendance/biometric",
            headers={"X-API-Key": "secret-key"},
            json=biometric_payload(1, 1),
        )
    assert res.status_code == 500
    body = res.json()
    assert "detail" in body
    assert "/var/internal/db.py" not in body["detail"]
    assert "Internal DB crashed" not in body["detail"]
