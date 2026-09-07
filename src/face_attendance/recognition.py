"""Module xử lý nhận diện khuôn mặt, đăng ký mẫu tham chiếu và WebRTC Video Engine.

Chịu trách nhiệm:
1. Đăng ký và kiểm tra độ mờ, độ sáng, kích thước và số khuôn mặt.
2. Tải và quản lý bộ cache vector mẫu (FaceTemplate) theo buổi học.
3. Nhận diện thời gian thực, chớp mắt và xác nhận qua nhiều khung hình.
4. Streamlit WebRTC VideoProcessor cho luồng camera trình duyệt.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    class VideoProcessorBase:  # type: ignore[no-redef]
        """Fallback class when streamlit_webrtc is not installed."""
        pass

from .config import (
    ATTEMPT_COOLDOWN_SECONDS,
    BLINK_EAR_CLOSED,
    BLINK_EAR_OPEN,
    BLINK_VERIFICATION_SECONDS,
    MAX_BRIGHTNESS,
    MAX_UPLOAD_BYTES,
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MIN_FACE_SIZE_PX,
    PROCESS_EVERY_N_FRAMES,
)
from .database import (
    get_connection,
    get_student_by_code,
    get_student_embeddings,
    save_embedding,
    upsert_student,
)
from .domain import BiometricConsentMissingError, RecognitionDecision
from .liveness import BoKiemTraChopMat, ti_le_mat
from .matcher import tim_danh_tinh_tot_nhat
from .policy import DEFAULT_RECOGNITION_POLICY, RecognitionPolicy
from .utils import utc_iso

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentResult:
    """Kết quả giải mã và xác thực ảnh đăng ký sinh viên.

    Attributes:
        embedding (np.ndarray): Vector khuôn mặt 128 chiều.
        image_hash (str): Hash SHA-256 của file ảnh gốc.
        blur_score (float): Điểm độ sắc nét (Laplacian variance).
        brightness (float): Độ sáng trung bình (0-255).
        face_width (int): Chiều rộng vùng mặt (px).
        face_height (int): Chiều cao vùng mặt (px).
        phash (str | None): Perceptual hash phục vụ phát hiện ảnh gần trùng.
    """

    embedding: np.ndarray
    image_hash: str
    blur_score: float
    brightness: float
    face_width: int
    face_height: int
    phash: str | None = None


def _check_identity_consistency(
    results: list[EnrollmentResult],
    policy: RecognitionPolicy,
    context: str,
) -> None:
    """Chặn việc trộn ảnh của nhiều người vào cùng một hồ sơ."""
    for index, current in enumerate(results):
        for previous in results[:index]:
            distance = float(np.linalg.norm(current.embedding - previous.embedding))
            if distance > policy.distance_threshold:
                raise ValueError(
                    f"{context}: các ảnh hợp lệ không nhất quán danh tính "
                    f"(khoảng cách {distance:.3f} > {policy.distance_threshold:.3f}). "
                    "Toàn bộ đợt đăng ký đã bị từ chối."
                )


def decode_and_validate_face(image_bytes: bytes) -> EnrollmentResult:
    """Giải mã file ảnh, kiểm tra tiêu chuẩn chất lượng và trích xuất vector khuôn mặt 128D.

    Các bước kiểm tra:
    1. Giới hạn dung lượng file <= 8 MB.
    2. Đọc định dạng ảnh (OpenCV imdecode).
    3. Điểm sắc nét (Laplacian Variance >= MIN_BLUR_SCORE).
    4. Độ sáng trung bình (MIN_BRIGHTNESS <= Mean <= MAX_BRIGHTNESS).
    5. Phát hiện duy nhất 1 khuôn mặt trong ảnh.
    6. Kích thước vùng mặt tối thiểu (>= 100x100px).
    7. Trích xuất vector đặc trưng 128D bằng dlib resnet model.
    8. Tính Perceptual Hash (pHash) để kiểm soát ảnh gần trùng.

    Args:
        image_bytes (bytes): Dữ liệu nhị phân của file ảnh.

    Returns:
        EnrollmentResult: Kết quả trích xuất hợp lệ.

    Raises:
        ValueError: Nếu ảnh vi phạm bất kỳ tiêu chuẩn chất lượng nào.
    """
    if not image_bytes:
        raise ValueError("File ảnh đang trống.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 8 MB.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Nội dung file không phải ảnh hợp lệ.")

    # 1. Kiểm tra độ mờ bằng biến thiên Laplacian
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if blur_score < MIN_BLUR_SCORE:
        raise ValueError(
            f"Ảnh quá mờ (blur={blur_score:.1f}, yêu cầu ≥ {MIN_BLUR_SCORE:.0f})."
        )

    # 2. Kiểm tra độ sáng trung bình
    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        raise ValueError(
            f"Ánh sáng chưa phù hợp (brightness={brightness:.1f}, "
            f"yêu cầu {MIN_BRIGHTNESS:.0f}-{MAX_BRIGHTNESS:.0f})."
        )

    # 3. Phát hiện số lượng khuôn mặt
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError("Thư viện face_recognition/dlib chưa được cài đặt.") from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image_rgb, number_of_times_to_upsample=1)
    if len(locations) != 1:
        raise ValueError(
            f"Mỗi ảnh phải có đúng 1 khuôn mặt; hệ thống tìm thấy {len(locations)}."
        )

    # 4. Kiểm tra kích thước khuôn mặt
    top, right, bottom, left = locations[0]
    face_width = right - left
    face_height = bottom - top
    if face_width < MIN_FACE_SIZE_PX or face_height < MIN_FACE_SIZE_PX:
        raise ValueError(
            f"Khuôn mặt quá nhỏ ({face_width}×{face_height}px); hãy đứng gần camera hơn."
        )

    # 5. Trích xuất 128-dimensional face encoding
    encodings = face_recognition.face_encodings(
        image_rgb, known_face_locations=locations, num_jitters=2, model="small"
    )
    if len(encodings) != 1:
        raise ValueError("Không thể tạo vector khuôn mặt từ ảnh này.")

    # 6. Tính toán pHash kiểm soát gần trùng
    phash_str: str | None = None
    try:
        import io

        import imagehash
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes))
        phash_str = str(imagehash.phash(pil_img))
    except Exception:
        phash_str = None

    return EnrollmentResult(
        embedding=np.asarray(encodings[0], dtype=np.float64),
        image_hash=hashlib.sha256(image_bytes).hexdigest(),
        blur_score=blur_score,
        brightness=brightness,
        face_width=face_width,
        face_height=face_height,
        phash=phash_str,
    )


def enroll_student_images(
    student_code: str,
    full_name: str,
    class_name: str,
    image_sources: Iterable[Any],
    consent_given: bool | None = None,
    consent_policy_version: str | None = None,
    policy: RecognitionPolicy | None = None,
) -> tuple[int, list[str]]:
    """Đăng ký sinh viên và lưu các mẫu vector khuôn mặt vào cơ sở dữ liệu.

    Args:
        student_code (str): Mã sinh viên.
        full_name (str): Họ tên sinh viên.
        class_name (str): Lớp sinh hoạt.
        image_sources: Danh sách ảnh tải lên hoặc ảnh chụp từ Streamlit.

    Returns:
        tuple[int, list[str]]: (Số ảnh đã lưu thành công, Danh sách cảnh báo/lỗi nếu có).

    ``consent_given=None`` chỉ được giữ cho các tool seed/compatibility cũ.
    Luồng người dùng và application service luôn truyền ``True`` hoặc ``False``;
    không gọi trực tiếp hàm thấp hơn để bỏ qua consent.
    """
    selected_policy = policy or DEFAULT_RECOGNITION_POLICY
    if consent_given is False:
        raise BiometricConsentMissingError(
            "Cần có consent rõ ràng trước khi giải mã hoặc xử lý ảnh khuôn mặt."
        )

    sources = list(image_sources)
    if not sources:
        raise ValueError("Hãy tải ảnh lên hoặc chụp ít nhất một ảnh.")
    if consent_given is True and len(sources) < 5:
        raise ValueError("Đăng ký chính thức yêu cầu đúng tối thiểu 5 ảnh hợp lệ.")

    # Chỉ tạo/cập nhật sinh viên sau khi đã có ít nhất một ảnh hợp lệ.
    validated: list[tuple[str, EnrollmentResult]] = []
    errors: list[str] = []
    for index, source in enumerate(sources, start=1):
        source_name = getattr(source, "name", f"Ảnh {index}")
        try:
            image_bytes = source.getvalue()
            validated.append((source_name, decode_and_validate_face(image_bytes)))
        except (AttributeError, OSError, ValueError) as exc:
            errors.append(f"{source_name}: {exc}")

    if not validated:
        raise ValueError("Không có ảnh hợp lệ. " + " | ".join(errors))

    # Kiểm tra near-duplicate giữa các ảnh trong cùng đợt tải lên
    valid_unique: list[tuple[str, EnrollmentResult]] = []
    for name_i, res_i in validated:
        is_near_dup = False
        for name_prev, res_prev in valid_unique:
            if res_i.image_hash == res_prev.image_hash:
                is_near_dup = True
                errors.append(f"{name_i}: Bỏ qua vì trùng byte với {name_prev}.")
                break
            if res_i.phash and res_prev.phash:
                try:
                    import imagehash

                    h_curr = imagehash.hex_to_hash(res_i.phash)
                    h_prev = imagehash.hex_to_hash(res_prev.phash)
                    if (h_curr - h_prev) <= 2:
                        is_near_dup = True
                        errors.append(
                            f"{name_i}: Quá giống {name_prev} (near-duplicate). "
                            "Hãy cung cấp góc mặt hoặc ánh sáng khác nhau để tăng độ bao phủ."
                        )
                        break
                except Exception:
                    pass
        if not is_near_dup:
            valid_unique.append((name_i, res_i))

    if not valid_unique:
        raise ValueError("Tất cả ảnh tải lên đều bị trùng lặp hoặc near-duplicate. " + " | ".join(errors))

    if consent_given is True and len(valid_unique) < 5:
        raise ValueError(
            "Đợt đăng ký phải còn tối thiểu 5 ảnh khác nhau sau kiểm tra chất lượng và trùng lặp."
        )

    # Kiểm tra consistency trước khi tạo/cập nhật student để không tạo profile
    # dở dang khi batch ảnh chứa nhiều danh tính.
    _check_identity_consistency(
        [result for _, result in valid_unique],
        selected_policy,
        "Enrollment mới",
    )

    if consent_given is True:
        existing_student = get_student_by_code(student_code)
        if existing_student is not None:
            with get_connection() as connection:
                incompatible = connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM face_embeddings
                    WHERE student_id = ?
                      AND revoked_at_utc IS NULL
                      AND (
                          embedding_dim != ?
                          OR embedding_model != ?
                          OR embedding_model_version != ?
                      )
                    """,
                    (
                        int(existing_student["id"]),
                        selected_policy.embedding_dimension,
                        selected_policy.embedding_model,
                        selected_policy.embedding_model_version,
                    ),
                ).fetchone()["total"]
            if incompatible:
                raise ValueError(
                    "Profile biometric hiện tại dùng model/version khác; "
                    "hãy thu hồi và đăng ký lại thay vì trộn hai vector space."
                )
            existing_embeddings = get_student_embeddings(int(existing_student["id"]), selected_policy)
            for _, result in valid_unique:
                if existing_embeddings:
                    nearest = min(
                        float(np.linalg.norm(result.embedding - old_embedding))
                        for old_embedding in existing_embeddings
                    )
                    if nearest > selected_policy.distance_threshold:
                        raise ValueError(
                            "Ảnh mới không nhất quán với profile biometric hiện tại "
                            f"(khoảng cách gần nhất {nearest:.3f} > "
                            f"{selected_policy.distance_threshold:.3f})."
                        )

    student = upsert_student(
        student_code,
        full_name,
        class_name,
        consent_given=consent_given,
        consent_policy_version=consent_policy_version or selected_policy.policy_version,
    )
    saved = 0
    duplicates = 0
    for _, result in valid_unique:
        if save_embedding(
            int(student["id"]),
            result.embedding,
            result.image_hash,
            result.blur_score,
            result.brightness,
            result.face_width,
            result.face_height,
            policy=selected_policy,
            phash=result.phash,
        ):
            saved += 1
        else:
            duplicates += 1

    messages = errors
    if duplicates:
        messages.append(f"Bỏ qua {duplicates} ảnh trùng đã đăng ký trước đó trong DB.")
    return saved, messages


@dataclass(frozen=True)
class FaceTemplate:
    """Mẫu khuôn mặt tham chiếu được nạp vào RAM cho quá trình so khớp realtime."""

    student_id: int
    student_code: str
    full_name: str
    embedding: np.ndarray


def load_templates(
    session_id: int | None = None,
    policy: RecognitionPolicy | None = None,
) -> list[FaceTemplate]:
    """Tải danh sách các mẫu khuôn mặt (FaceTemplate) active từ cơ sở dữ liệu.

    Tùy chọn lọc theo `session_id` để chỉ tải các sinh viên thuộc danh sách môn học của buổi đó.

    Args:
        session_id (int | None): ID buổi học hoặc None nếu nạp toàn bộ.

    Returns:
        list[FaceTemplate]: Danh sách mẫu tham chiếu.
    """
    selected_policy = policy or DEFAULT_RECOGNITION_POLICY
    mismatch_query = """
        SELECT COUNT(*) AS total
        FROM face_embeddings fe
        JOIN students s ON s.id = fe.student_id
    """
    mismatch_conditions = [
        "s.active = 1",
        "s.consent_status = 'granted'",
        "fe.revoked_at_utc IS NULL",
        "(fe.embedding_dim != ? OR fe.embedding_model != ? OR fe.embedding_model_version != ?)",
    ]
    mismatch_params: tuple[Any, ...] = (
        selected_policy.embedding_dimension,
        selected_policy.embedding_model,
        selected_policy.embedding_model_version,
    )
    if session_id is not None:
        mismatch_query += " JOIN session_enrollments se ON se.student_id = s.id"
        mismatch_conditions.append("se.session_id = ?")
        mismatch_params += (session_id,)
    mismatch_query += " WHERE " + " AND ".join(mismatch_conditions)
    with get_connection() as connection:
        mismatch_count = int(connection.execute(mismatch_query, mismatch_params).fetchone()["total"])
    if mismatch_count:
        raise RuntimeError(
            "Gallery chứa embedding không tương thích với recognition policy; "
            "cần re-enrollment thay vì trộn khác model/version."
        )

    query = """
    SELECT s.id AS student_id, s.student_code, s.full_name, fe.embedding
    FROM face_embeddings fe
    JOIN students s ON s.id = fe.student_id
    """
    conditions = [
        "s.active = 1",
        "s.consent_status = 'granted'",
        "fe.revoked_at_utc IS NULL",
        "fe.embedding_dim = ?",
        "fe.embedding_model = ?",
        "fe.embedding_model_version = ?",
    ]
    params: tuple[Any, ...] = (
        selected_policy.embedding_dimension,
        selected_policy.embedding_model,
        selected_policy.embedding_model_version,
    )
    if session_id is not None:
        query += """
        JOIN session_enrollments se ON se.student_id = s.id
        """
        conditions.append("se.session_id = ?")
        params += (session_id,)
    query += " WHERE " + " AND ".join(conditions)
    query += """
    ORDER BY s.student_code, fe.id
    """
    templates: list[FaceTemplate] = []
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float64).copy()
        if embedding.shape == (selected_policy.embedding_dimension,):
            templates.append(
                FaceTemplate(
                    student_id=int(row["student_id"]),
                    student_code=str(row["student_code"]),
                    full_name=str(row["full_name"]),
                    embedding=embedding,
                )
            )
    return templates


class RecognitionEngine:
    """Bộ máy xử lý và điều phối nhận diện khuôn mặt realtime qua luồng camera WebRTC.

    Tích hợp:
    - Single-face Policy Gate: Nghiêm ngặt từ chối nhận diện khi có > 1 khuôn mặt trong khung hình.
    - Skip-frame processing (thu nhỏ ảnh 0.25x) để duy trì tốc độ FPS cao.
    - So khớp Open-Set với từ chối người lạ (Matcher).
    - Máy trạng thái liveness chớp mắt (BoKiemTraChopMat).
    - Đếm xác nhận liên tiếp nhiều khung hình cùng danh tính ổn định (Confirmation frames).
    - Clean Architecture: Chỉ sinh RecognitionDecision DTO, chuyển giao lưu trữ cho AttendanceService.
    - Khóa Threading Lock cho đồng bộ sự kiện sang UI Streamlit.
    """

    def __init__(
        self,
        session_id: int,
        require_blink: bool,
        attendance_service: Any = None,
        policy: RecognitionPolicy | None = None,
    ) -> None:
        self.session_id = session_id
        self.policy = policy or DEFAULT_RECOGNITION_POLICY
        if attendance_service is None:
            from .application.attendance_service import record_biometric_attendance

            self.attendance_service = record_biometric_attendance
        else:
            self.attendance_service = attendance_service
        self.require_blink = require_blink
        self.templates = load_templates(session_id, self.policy)
        self.frame_number = 0
        self.confirm_counts: dict[int, int] = {}
        self.tracking_started_at: float | None = None
        self.current_tracking_student_id: int | None = None
        self.blink_checker = BoKiemTraChopMat(
            BLINK_EAR_CLOSED, BLINK_EAR_OPEN, BLINK_VERIFICATION_SECONDS
        )
        self.last_attempt: dict[int, float] = {}
        self.last_event = "Đang chờ khuôn mặt..."
        self.last_event_type = "info"
        self.last_event_at = utc_iso()
        self.lock = threading.Lock()
        self.last_annotations: list[tuple[int, int, int, int, tuple[int, int, int], str]] = []

    def snapshot(self) -> tuple[str, str, str]:
        """Trích xuất ảnh chụp trạng thái sự kiện mới nhất cho UI (thread-safe)."""
        with self.lock:
            return self.last_event_type, self.last_event, self.last_event_at

    def set_event(self, event_type: str, message: str) -> None:
        """Cập nhật thông báo sự kiện điểm danh cho UI (thread-safe)."""
        with self.lock:
            self.last_event_type = event_type
            self.last_event = message
            self.last_event_at = utc_iso()

    def best_identity(
        self, encoding: np.ndarray
    ) -> tuple[FaceTemplate | None, float, float, float]:
        """Tìm danh tính khớp nhất sử dụng module matcher.

        Returns:
            (template_tot_nhat, khoang_cach_top1, khoang_cach_top2, margin)
        """
        if not self.templates:
            return None, float("inf"), float("inf"), float("inf")
        result = tim_danh_tinh_tot_nhat(
            encoding,
            self.templates,
            self.policy.distance_threshold,
            self.policy.identity_margin,
            strategy=self.policy.aggregation_strategy,
            top_k=self.policy.top_k,
        )
        return result.mau, result.khoang_cach, result.khoang_cach_thu_hai, result.do_phan_biet

    def update_blink(self, student_id: int, landmarks: dict[str, Any] | None) -> bool:
        """Cập nhật tỉ lệ mắt và máy trạng thái chớp mắt."""
        if not self.require_blink:
            return True
        if not landmarks:
            return False
        left = ti_le_mat(landmarks.get("left_eye", []))
        right = ti_le_mat(landmarks.get("right_eye", []))
        if left is None or right is None:
            return False
        return self.blink_checker.cap_nhat(student_id, (left + right) / 2.0)

    def draw_annotations(self, image_bgr: np.ndarray) -> np.ndarray:
        """Vẽ các ô bounding box và nhãn tên/trạng thái lên hình ảnh."""
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
        """Xử lý chính cho mỗi khung hình (Frame processing pipeline).

        Áp dụng Single-Face Gate: Chỉ xử lý điểm danh khi phát hiện đúng 1 khuôn mặt.
        """
        self.frame_number += 1
        # Chỉ xử lý các frame cách nhau PROCESS_EVERY_N_FRAMES để tăng tốc độ xử lý
        if self.frame_number % PROCESS_EVERY_N_FRAMES != 0:
            return self.draw_annotations(image_bgr)

        # Thu nhỏ khung hình 0.25x để phát hiện khuôn mặt nhanh hơn
        import face_recognition

        small = cv2.resize(image_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model="hog")

        num_faces = len(locations)

        # 1. Không tìm thấy khuôn mặt
        if num_faces == 0:
            self.current_tracking_student_id = None
            self.confirm_counts.clear()
            self.tracking_started_at = None
            self.last_annotations = []
            self.set_event("info", "Đang chờ khuôn mặt...")
            return self.draw_annotations(image_bgr)

        # 2. Phát hiện nhiều hơn 1 khuôn mặt: Kích hoạt Single-Face Gate từ chối
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
                        (0, 140, 255),  # Màu cam: Cảnh báo
                        f"NHIỀU KHUÔN MẶT ({num_faces})",
                    )
                )
            self.last_annotations = warning_annotations
            self.set_event(
                "warning",
                f"Phát hiện {num_faces} khuôn mặt. Chính sách yêu cầu duy nhất 1 người trước camera.",
            )
            return self.draw_annotations(image_bgr)

        # 3. Duy nhất 1 khuôn mặt hợp lệ: Tiếp tục pipeline
        location = locations[0]
        encodings = face_recognition.face_encodings(rgb_small, [location], model="small")
        landmarks_list = (
            face_recognition.face_landmarks(rgb_small, [location], model="small")
            if self.require_blink
            else [{}]
        )
        if not encodings:
            self.set_event("warning", "Không thể trích xuất đặc trưng khuôn mặt.")
            return self.draw_annotations(image_bgr)

        encoding = encodings[0]
        landmarks = landmarks_list[0] if landmarks_list else {}
        template, distance, second_distance, margin = self.best_identity(encoding)

        if template:
            # Theo dõi tính ổn định danh tính (tránh nhảy liên tục giữa SV001 và SV002)
            if self.current_tracking_student_id != template.student_id:
                self.current_tracking_student_id = template.student_id
                self.confirm_counts[template.student_id] = 0
                self.tracking_started_at = time.monotonic()
                self.blink_checker.dat_lai(template.student_id)

            live = self.update_blink(template.student_id, landmarks)
            if live:
                self.confirm_counts[template.student_id] = (
                    self.confirm_counts.get(template.student_id, 0) + 1
                )
            else:
                # Liveness chưa hợp lệ thì không cộng dồn temporal evidence.
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
                max(0.0, time.monotonic() - (self.tracking_started_at or time.monotonic()))
                * 1000
            )
            if not live:
                color = (0, 215, 255)  # Vàng: Cần chớp mắt
                label = f"{template.student_code} - CHỚP MẮT"
                self.set_event("warning", f"{template.student_code}: Hãy chớp mắt một lần.")
            elif count < self.policy.minimum_observations or stable_ms < self.policy.stable_duration_ms:
                color = (0, 215, 255)  # Vàng: Đang xác nhận giữ yên
                label = (
                    f"{template.student_code} - GIỮ YÊN "
                    f"{count}/{self.policy.minimum_observations}, {stable_ms}ms"
                )
            else:
                color = (0, 255, 0)  # Xanh lá: Khớp thành công
                label = f"{template.student_code} - KHỚP {distance:.3f} (Δ={margin:.3f})"
                now_mono = time.monotonic()
                last = self.last_attempt.get(template.student_id, 0.0)
                # Ghi điểm danh qua Clean Architecture Decision Service
                if now_mono - last >= ATTEMPT_COOLDOWN_SECONDS:
                    decision = RecognitionDecision(
                        student_id=template.student_id,
                        student_code=template.student_code,
                        full_name=template.full_name,
                        distance=distance,
                        second_distance=second_distance,
                        margin=margin,
                        liveness_passed=True,
                        confirmation_frames=count,
                        policy_version=self.policy.policy_version,
                        timestamp_utc=utc_iso(),
                        distance_threshold=self.policy.distance_threshold,
                        margin_threshold=self.policy.identity_margin,
                        aggregation_strategy=self.policy.aggregation_strategy,
                        embedding_model=self.policy.embedding_model,
                        embedding_model_version=self.policy.embedding_model_version,
                        stable_duration_ms=stable_ms,
                        liveness_policy=self.policy.liveness_policy,
                        recognition_policy_hash=self.policy.policy_hash,
                    )
                    self.last_attempt[template.student_id] = now_mono
                    # Mỗi lần gửi decision là một attendance attempt mới. Không
                    # cho phép lần sau dùng lại bằng chứng blink của attempt này.
                    self.blink_checker.dat_lai(template.student_id)
                    self.confirm_counts[template.student_id] = 0
                    self.tracking_started_at = None
                    try:
                        res = self.attendance_service(self.session_id, decision)
                        status_label = "CÓ MẶT" if res.status == "present" else "ĐI TRỄ"
                        self.set_event("success", f"{template.student_code} - {status_label}")
                    except Exception as exc:
                        msg = str(exc)
                        evt_type = "info" if "đã điểm danh" in msg.lower() else "error"
                        self.set_event(evt_type, msg)

        new_annotations.append((top, right, bottom, left, color, label))
        self.last_annotations = new_annotations
        return self.draw_annotations(image_bgr)



class AttendanceVideoProcessor(VideoProcessorBase):
    """Processor tương thích với streamlit_webrtc để xử lý từng VideoFrame từ webcam trình duyệt."""

    def __init__(self, engine: RecognitionEngine) -> None:
        self.engine = engine

    def recv(self, frame: Any) -> Any:
        """Hàm callback nhận khung hình video từ WebRTC streamer."""
        import av

        image = frame.to_ndarray(format="bgr24")
        try:
            output = self.engine.process(image)
        except Exception:
            LOGGER.exception("Không thể xử lý khung hình từ camera")
            self.engine.set_event("error", "Không thể xử lý hình ảnh từ camera.")
            output = image
        return av.VideoFrame.from_ndarray(output, format="bgr24")
