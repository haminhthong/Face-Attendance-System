"""Module xử lý nhận diện khuôn mặt thời gian thực và WebRTC Video Engine.

Chịu trách nhiệm:
1. Tải và quản lý bộ nhớ đệm mẫu vector khuôn mặt (FaceTemplate) theo buổi học.
2. Single-Face Gate: Chặn nhận diện nếu phát hiện nhiều hơn 1 khuôn mặt trong khung hình.
3. Open-Set Recognition: So khớp khoảng cách L2 và Top-1 vs Top-2 Margin.
4. Basic Blink Challenge: Kiểm tra cử động chớp mắt bằng tỉ lệ EAR.
5. Temporal Confirmation: Xác nhận ổn định danh tính qua nhiều khung hình liên tiếp.
6. Streamlit WebRTC VideoProcessor cho luồng camera trình duyệt.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:

    class VideoProcessorBase:  # type: ignore[no-redef]
        """Fallback khi streamlit_webrtc không có sẵn."""

        pass


from .attendance import DuplicateAttendanceError, record_biometric_attendance
from .config import (
    ATTEMPT_COOLDOWN_SECONDS,
    DEFAULT_CONFIG,
    PROCESS_EVERY_N_FRAMES,
    RecognitionConfig,
)
from .database import get_connection

# Re-export các hàm enrollment để giữ tính tiện dụng
from .enrollment import (
    EnrollmentSample as EnrollmentResult,  # noqa: F401
)
from .enrollment import (
    decode_and_validate_face,  # noqa: F401
    enroll_student,  # noqa: F401
)
from .liveness import BlinkDetector, eye_aspect_ratio
from .matcher import MatchResult, match_face
from .utils import utc_iso

LOGGER = logging.getLogger(__name__)


# Alias tương thích
def enroll_student_images(*args: Any, **kwargs: Any):
    """Hàm wrapper cho enroll_student."""
    return enroll_student(*args, **kwargs)


@dataclass(frozen=True)
class FaceTemplate:
    """Mẫu vector khuôn mặt tham chiếu của sinh viên."""

    student_id: int
    student_code: str
    full_name: str
    embedding: np.ndarray


def load_templates(
    session_id: int | None = None,
    config: RecognitionConfig | None = None,
) -> list[FaceTemplate]:
    """Tải toàn bộ vector khuôn mặt 128D của sinh viên có trong danh sách buổi học."""
    query = """
    SELECT s.id AS student_id, s.student_code, s.full_name, fe.embedding
    FROM face_embeddings fe
    JOIN students s ON s.id = fe.student_id
    """
    conditions = ["s.active = 1"]
    params: list[Any] = []

    if session_id is not None:
        query += " JOIN session_enrollments se ON se.student_id = s.id"
        conditions.append("se.session_id = ?")
        params.append(session_id)

    query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.student_code, fe.id"

    templates: list[FaceTemplate] = []
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float64).copy()
        if emb.shape == (128,):
            templates.append(
                FaceTemplate(
                    student_id=int(row["student_id"]),
                    student_code=str(row["student_code"]),
                    full_name=str(row["full_name"]),
                    embedding=emb,
                )
            )
    return templates


class RecognitionEngine:
    """Bộ máy điều phối và xử lý nhận diện khuôn mặt thời gian thực."""

    def __init__(
        self,
        session_id: int,
        require_blink: bool = True,
        attendance_service: Any = None,
        config: RecognitionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.session_id = session_id
        self.config = config or DEFAULT_CONFIG
        self.require_blink = require_blink
        self.attendance_service = attendance_service or record_biometric_attendance
        self.templates = load_templates(session_id, self.config)

        self.frame_number = 0
        self.confirm_counts: dict[int, int] = {}
        self.tracking_started_at: float | None = None
        self.current_tracking_student_id: int | None = None

        self.blink_detector = BlinkDetector(
            eye_closed_threshold=self.config.eye_closed_threshold,
            eye_open_threshold=self.config.eye_open_threshold,
        )
        self.last_attempt: dict[int, float] = {}
        self.last_event = "Đang chờ khuôn mặt..."
        self.last_event_type = "info"
        self.last_event_at = utc_iso()
        self.lock = threading.Lock()
        self.last_annotations: list[tuple[int, int, int, int, tuple[int, int, int], str]] = []

    def snapshot(self) -> tuple[str, str, str]:
        """Trích xuất trạng thái sự kiện mới nhất cho UI (thread-safe)."""
        with self.lock:
            return self.last_event_type, self.last_event, self.last_event_at

    def set_event(self, event_type: str, message: str) -> None:
        """Cập nhật thông báo sự kiện cho UI (thread-safe)."""
        with self.lock:
            self.last_event_type = event_type
            self.last_event = message
            self.last_event_at = utc_iso()

    def best_identity(
        self, encoding: np.ndarray
    ) -> tuple[FaceTemplate | None, float, float, float]:
        """Tìm danh tính khớp nhất sử dụng thuật toán Open-Set Matcher."""
        if not self.templates:
            return None, float("inf"), float("inf"), float("inf")
        res: MatchResult = match_face(
            encoding,
            self.templates,
            distance_threshold=self.config.distance_threshold,
            identity_margin=self.config.identity_margin,
            top_k=self.config.top_k,
        )
        return res.template, res.distance, res.second_distance, res.margin

    def update_blink(self, student_id: int, landmarks: dict[str, Any] | None) -> bool:
        """Kiểm tra và cập nhật trạng thái chớp mắt của sinh viên."""
        if not landmarks:
            return False
        left = eye_aspect_ratio(landmarks.get("left_eye", []))
        right = eye_aspect_ratio(landmarks.get("right_eye", []))
        if left is None or right is None:
            return False
        avg_ear = (left + right) / 2.0
        return self.blink_detector.update(student_id, avg_ear)

    def draw_annotations(self, image_bgr: np.ndarray) -> np.ndarray:
        """Vẽ bounding box và nhãn tên/trạng thái lên hình ảnh."""
        for top, right, bottom, left, color, label in self.last_annotations:
            cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
            label_y = max(25, top - 10)
            cv2.putText(
                image_bgr,
                label,
                (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return image_bgr

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """Quy trình xử lý từng khung hình video từ camera."""
        self.frame_number += 1
        if self.frame_number % PROCESS_EVERY_N_FRAMES != 0:
            return self.draw_annotations(image_bgr)

        import face_recognition

        # Giảm kích thước ảnh 0.25x để tăng tốc độ phát hiện mặt
        small = cv2.resize(image_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model="hog")
        num_faces = len(locations)

        # 1. Không phát hiện khuôn mặt nào
        if num_faces == 0:
            self.current_tracking_student_id = None
            self.confirm_counts.clear()
            self.tracking_started_at = None
            self.last_annotations = []
            self.set_event("info", "Đang chờ khuôn mặt...")
            return self.draw_annotations(image_bgr)

        # 2. Phát hiện nhiều hơn 1 khuôn mặt: Single-Face Gate từ chối
        if num_faces > 1:
            self.current_tracking_student_id = None
            self.confirm_counts.clear()
            self.tracking_started_at = None
            warning_annotations = []
            for top, right, bottom, left in locations:
                warning_annotations.append(
                    (
                        top * 4,
                        right * 4,
                        bottom * 4,
                        left * 4,
                        (0, 140, 255),  # Màu cam
                        f"NHIỀU KHUÔN MẶT ({num_faces})",
                    )
                )
            self.last_annotations = warning_annotations
            self.set_event(
                "warning",
                f"Phát hiện {num_faces} khuôn mặt. Điểm danh yêu cầu duy nhất 1 người trước camera.",
            )
            return self.draw_annotations(image_bgr)

        # 3. Duy nhất 1 khuôn mặt: Tiếp tục nhận diện
        location = locations[0]
        encodings = face_recognition.face_encodings(rgb_small, [location], model="small")
        landmarks_list = face_recognition.face_landmarks(rgb_small, [location], model="small")

        if not encodings:
            self.set_event("warning", "Không thể trích xuất đặc trưng khuôn mặt.")
            return self.draw_annotations(image_bgr)

        encoding = encodings[0]
        landmarks = landmarks_list[0] if landmarks_list else {}
        template, distance, _second_distance, margin = self.best_identity(encoding)

        if template:
            # Ổn định danh tính: nếu đổi người, reset lại toàn bộ đếm xác nhận
            if self.current_tracking_student_id != template.student_id:
                self.current_tracking_student_id = template.student_id
                self.confirm_counts[template.student_id] = 0
                self.tracking_started_at = time.monotonic()
                self.blink_detector.reset(template.student_id)

            live = self.update_blink(template.student_id, landmarks) if self.require_blink else True
            if live:
                self.confirm_counts[template.student_id] = (
                    self.confirm_counts.get(template.student_id, 0) + 1
                )
            else:
                self.confirm_counts[template.student_id] = 0
        else:
            self.current_tracking_student_id = None
            self.confirm_counts.clear()
            self.tracking_started_at = None
            live = False

        top, right, bottom, left = location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        new_annotations: list[tuple[int, int, int, int, tuple[int, int, int], str]] = []

        if template is None:
            color = (0, 0, 255)  # Đỏ: Người lạ
            label = "KHÔNG XÁC ĐỊNH"
            self.set_event("warning", "Khuôn mặt chưa được đăng ký trong danh sách buổi học.")
        else:
            count = self.confirm_counts.get(template.student_id, 0)
            stable_ms = int(
                max(0.0, time.monotonic() - (self.tracking_started_at or time.monotonic())) * 1000
            )

            if not live:
                color = (0, 215, 255)  # Vàng: Chờ chớp mắt
                label = f"{template.student_code} - CHỚP MẮT"
                self.set_event("warning", f"{template.student_code}: Hãy chớp mắt một lần.")
            elif (
                count < self.config.min_confirmations or stable_ms < self.config.stable_duration_ms
            ):
                color = (0, 215, 255)  # Vàng: Đang giữ ổn định
                label = f"{template.student_code} - GIỮ YÊN {count}/{self.config.min_confirmations}"
            else:
                color = (0, 255, 0)  # Xanh: Đã xác thực thành công
                label = f"{template.student_code} - KHỚP {distance:.3f} (Δ={margin:.3f})"
                now_mono = time.monotonic()
                last = self.last_attempt.get(template.student_id, 0.0)

                # Ghi nhận điểm danh nếu đã qua thời gian cooldown
                if now_mono - last >= ATTEMPT_COOLDOWN_SECONDS:
                    self.last_attempt[template.student_id] = now_mono
                    self.blink_detector.reset(template.student_id)
                    self.confirm_counts[template.student_id] = 0
                    self.tracking_started_at = None
                    self.current_tracking_student_id = None

                    try:
                        res = self.attendance_service(
                            session_id=self.session_id,
                            student_id=template.student_id,
                            distance=distance,
                            margin=margin,
                        )
                        status_label = "CÓ MẶT" if res.status == "present" else "ĐI TRỄ"
                        self.set_event("success", f"{template.student_code} - {status_label}")
                    except DuplicateAttendanceError:
                        self.set_event("info", f"{template.student_code}: Đã điểm danh trước đó.")
                    except Exception as exc:
                        self.set_event("error", str(exc))

        new_annotations.append((top, right, bottom, left, color, label))
        self.last_annotations = new_annotations
        return self.draw_annotations(image_bgr)


class AttendanceVideoProcessor(VideoProcessorBase):
    """Processor cho Streamlit WebRTC nhận luồng khung hình từ webcam."""

    def __init__(self, engine: RecognitionEngine) -> None:
        self.engine = engine

    def recv(self, frame: Any) -> Any:
        import av

        image = frame.to_ndarray(format="bgr24")
        try:
            output = self.engine.process(image)
        except Exception:
            LOGGER.exception("Không thể xử lý khung hình camera")
            self.engine.set_event("error", "Lỗi khi xử lý hình ảnh camera.")
            output = image
        return av.VideoFrame.from_ndarray(output, format="bgr24")
