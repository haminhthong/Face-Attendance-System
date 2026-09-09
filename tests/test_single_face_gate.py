"""Unit tests cho cơ chế Single-Face Gate và Temporal Stability trong RecognitionEngine."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Đảm bảo face_recognition có trong sys.modules để mock trên môi trường không có dlib binary
if "face_recognition" not in sys.modules:
    sys.modules["face_recognition"] = MagicMock()

import face_attendance.recognition
from face_attendance.recognition import FaceTemplate, RecognitionEngine


@pytest.fixture(autouse=True)
def process_every_frame(monkeypatch):
    """Đặt PROCESS_EVERY_N_FRAMES = 1 để kiểm tra frame-by-frame mà không bị skip."""
    monkeypatch.setattr(face_attendance.recognition, "PROCESS_EVERY_N_FRAMES", 1)


@pytest.fixture
def mock_engine() -> RecognitionEngine:
    with patch("face_attendance.recognition.load_templates") as mock_load:
        # Mock 2 sinh viên trong session
        mock_load.return_value = [
            FaceTemplate(
                student_id=1,
                student_code="SV001",
                full_name="Nguyễn Văn A",
                embedding=np.zeros(128),
            ),
            FaceTemplate(
                student_id=2,
                student_code="SV002",
                full_name="Trần Văn B",
                embedding=np.ones(128),
            ),
        ]
        mock_svc = MagicMock()
        engine = RecognitionEngine(session_id=1, require_blink=False, attendance_service=mock_svc)
        # Unit test này tập trung vào single-face/temporal gate; liveness đã có
        # test riêng nên mock bằng chứng blink để không tạo attendance thật.
        engine.update_blink = MagicMock(return_value=True)
        return engine


def test_single_face_gate_zero_faces(mock_engine: RecognitionEngine) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("face_recognition.face_locations", return_value=[]):
        mock_engine.process(frame)

    event_type, msg, _ = mock_engine.snapshot()
    assert event_type == "info"
    assert "Đang chờ khuôn mặt" in msg
    assert mock_engine.current_tracking_student_id is None
    assert len(mock_engine.confirm_counts) == 0


def test_single_face_gate_multiple_faces(mock_engine: RecognitionEngine) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 2 khuôn mặt phát hiện
    fake_locations = [(10, 50, 60, 10), (100, 150, 160, 100)]
    with patch("face_recognition.face_locations", return_value=fake_locations):
        mock_engine.process(frame)

    event_type, msg, _ = mock_engine.snapshot()
    assert event_type == "warning"
    assert "Phát hiện 2 khuôn mặt" in msg
    assert mock_engine.current_tracking_student_id is None
    assert any("NHIỀU KHUÔN MẶT (2)" in ann[5] for ann in mock_engine.last_annotations)


def test_single_face_gate_single_face_proceeds(mock_engine: RecognitionEngine) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_locations = [(10, 50, 60, 10)]
    fake_encoding = [np.zeros(128)]

    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=fake_encoding),
    ):
        mock_engine.process(frame)

    assert mock_engine.current_tracking_student_id == 1
    assert mock_engine.confirm_counts[1] == 1


def test_candidate_switching_resets_confirmation(mock_engine: RecognitionEngine) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_locations = [(10, 50, 60, 10)]

    # Frame đối tượng SV001 (zeros)
    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=[np.zeros(128)]),
    ):
        mock_engine.process(frame)
        assert mock_engine.current_tracking_student_id == 1
        assert mock_engine.confirm_counts[1] == 1

        mock_engine.process(frame)
        assert mock_engine.confirm_counts[1] == 2

    # Đột ngột đổi sang SV002 (ones)
    with (
        patch("face_recognition.face_locations", return_value=fake_locations),
        patch("face_recognition.face_encodings", return_value=[np.ones(128)]),
    ):
        mock_engine.process(frame)
        # ID đổi sang 2, confirmation count của SV002 bắt đầu lại từ 1
        assert mock_engine.current_tracking_student_id == 2
        assert mock_engine.confirm_counts[2] == 1
