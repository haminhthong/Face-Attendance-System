"""Service xử lý nghiệp vụ điểm danh sinh viên (Attendance Application Service)."""

from __future__ import annotations

from ..database import log_recognition_attempt, manual_attendance_correction, mark_attendance
from ..domain import (
    AttendanceResult,
    DuplicateAttendanceError,
    RecognitionDecision,
    RejectionReason,
    SessionClosedError,
    StudentNotInRosterError,
    build_accepted_result,
    build_rejected_result,
)
from ..policy import DEFAULT_RECOGNITION_POLICY
from ..utils import utc_iso


def record_biometric_attendance(
    session_id: int,
    decision: RecognitionDecision,
    tolerance: float | None = None,
) -> AttendanceResult:
    """Xử lý quyết định nhận diện từ Vision Pipeline và thực hiện giao dịch ghi nhận điểm danh.

    Chỉ chấp nhận khi RecognitionDecision đã vượt qua kiểm tra liveness và số khung hình xác thực.
    Lưu trữ đầy đủ bằng chứng nhận diện (distance, margin, liveness, policy version) phục vụ kiểm toán.
    """
    now_str = utc_iso()
    # Không lấy lại threshold từ config khác. Evidence trong decision là policy
    # đã được matcher dùng và phải là nguồn duy nhất cho transaction này.
    effective_tolerance = decision.distance_threshold
    if not decision.liveness_passed:
        log_recognition_attempt(
            session_id,
            "rejected",
            decision.policy_version,
            rejection_reason=RejectionReason.LIVENESS_FAILED.value,
            candidate_student_id=decision.student_id,
            distance=decision.distance,
            margin=decision.margin,
            stable_duration_ms=decision.stable_duration_ms,
        )
        return build_rejected_result(
            reason=RejectionReason.LIVENESS_FAILED,
            recognized_at=now_str,
            student_id=str(decision.student_id),
            distance=decision.distance,
            liveness_passed=False,
            tolerance=effective_tolerance,
            margin=decision.margin,
            source="face_webrtc",
            policy_version=decision.policy_version,
            distance_threshold=effective_tolerance,
            margin_threshold=decision.margin_threshold,
            aggregation_strategy=decision.aggregation_strategy,
            embedding_model_version=decision.embedding_model_version,
            stable_duration_ms=decision.stable_duration_ms,
            recognition_policy_hash=decision.recognition_policy_hash,
        )

    if decision.margin < decision.margin_threshold:
        log_recognition_attempt(
            session_id,
            "rejected",
            decision.policy_version,
            rejection_reason=RejectionReason.AMBIGUOUS_MATCH.value,
            candidate_student_id=decision.student_id,
            distance=decision.distance,
            margin=decision.margin,
            liveness_passed=True,
            stable_duration_ms=decision.stable_duration_ms,
        )
        return build_rejected_result(
            reason=RejectionReason.AMBIGUOUS_MATCH,
            recognized_at=now_str,
            student_id=str(decision.student_id),
            distance=decision.distance,
            liveness_passed=True,
            tolerance=effective_tolerance,
            margin=decision.margin,
            source="face_webrtc",
            policy_version=decision.policy_version,
            distance_threshold=effective_tolerance,
            margin_threshold=decision.margin_threshold,
            aggregation_strategy=decision.aggregation_strategy,
            embedding_model_version=decision.embedding_model_version,
            stable_duration_ms=decision.stable_duration_ms,
            recognition_policy_hash=decision.recognition_policy_hash,
        )

    if decision.distance > effective_tolerance:
        log_recognition_attempt(
            session_id,
            "rejected",
            decision.policy_version,
            rejection_reason=RejectionReason.UNKNOWN_FACE.value,
            candidate_student_id=decision.student_id,
            distance=decision.distance,
            margin=decision.margin,
            liveness_passed=True,
            stable_duration_ms=decision.stable_duration_ms,
        )
        return build_rejected_result(
            reason=RejectionReason.UNKNOWN_FACE,
            recognized_at=now_str,
            student_id=str(decision.student_id),
            distance=decision.distance,
            liveness_passed=True,
            tolerance=effective_tolerance,
            margin=decision.margin,
            source="face_webrtc",
            policy_version=decision.policy_version,
            distance_threshold=effective_tolerance,
            margin_threshold=decision.margin_threshold,
            aggregation_strategy=decision.aggregation_strategy,
            embedding_model_version=decision.embedding_model_version,
            stable_duration_ms=decision.stable_duration_ms,
            recognition_policy_hash=decision.recognition_policy_hash,
        )

    result_code, message = mark_attendance(
        session_id=session_id,
        student_id=decision.student_id,
        distance=decision.distance,
        identity_margin=decision.margin,
        margin_threshold=decision.margin_threshold,
        liveness_policy=decision.liveness_policy,
        policy_version=decision.policy_version,
        confirmation_frames=decision.confirmation_frames,
        stable_duration_ms=decision.stable_duration_ms,
        aggregation_strategy=decision.aggregation_strategy,
        embedding_model_version=decision.embedding_model_version,
        recognition_policy_hash=decision.recognition_policy_hash,
        source="face_webrtc",
        tolerance=effective_tolerance,
    )

    if result_code == "created":
        log_recognition_attempt(
            session_id,
            "accepted",
            decision.policy_version,
            candidate_student_id=decision.student_id,
            distance=decision.distance,
            margin=decision.margin,
            liveness_passed=True,
            stable_duration_ms=decision.stable_duration_ms,
        )
        status = "late" if "ĐI TRỄ" in message else "present"
        return build_accepted_result(
            student_id=str(decision.student_id),
            status=status,
            distance=decision.distance,
            liveness_passed=True,
            recognized_at=now_str,
            tolerance=effective_tolerance,
            margin=decision.margin,
            source="face_webrtc",
            policy_version=decision.policy_version,
            distance_threshold=effective_tolerance,
            margin_threshold=decision.margin_threshold,
            aggregation_strategy=decision.aggregation_strategy,
            embedding_model_version=decision.embedding_model_version,
            stable_duration_ms=decision.stable_duration_ms,
            recognition_policy_hash=decision.recognition_policy_hash,
        )
    elif result_code == "already":
        raise DuplicateAttendanceError(message)
    elif result_code in {"closed", "outside"}:
        raise SessionClosedError(message)
    elif result_code in {"not_in_roster", "inactive"}:
        raise StudentNotInRosterError(message)
    else:
        return build_rejected_result(
            reason=RejectionReason.UNKNOWN_FACE,
            recognized_at=now_str,
            student_id=str(decision.student_id),
            distance=decision.distance,
            liveness_passed=True,
            tolerance=effective_tolerance,
            margin=decision.margin,
            source="face_webrtc",
            policy_version=decision.policy_version,
            distance_threshold=effective_tolerance,
            margin_threshold=decision.margin_threshold,
            aggregation_strategy=decision.aggregation_strategy,
            embedding_model_version=decision.embedding_model_version,
            stable_duration_ms=decision.stable_duration_ms,
            recognition_policy_hash=decision.recognition_policy_hash,
        )


def record_manual_attendance(
    session_id: int,
    student_id: int,
    status: str,
    lecturer_id: str,
    reason: str,
) -> AttendanceResult:
    """Xử lý điểm danh thủ công hoặc sửa trạng thái bởi giảng viên (First-class manual correction)."""
    now_str = utc_iso()
    success, message = manual_attendance_correction(
        session_id=session_id,
        student_id=student_id,
        new_status=status,
        lecturer_id=lecturer_id,
        reason=reason,
    )
    if not success:
        raise ValueError(message)

    return build_accepted_result(
        student_id=str(student_id),
        status=status,
        distance=0.0,
        liveness_passed=False,  # Điểm danh thủ công không giả lập liveness AI
        recognized_at=now_str,
        tolerance=DEFAULT_RECOGNITION_POLICY.distance_threshold,
        margin=None,
        source="manual",
        policy_version="manual",
    )


def process_attendance_record(
    session_id: int,
    student_id: int,
    distance: float,
    tolerance: float = DEFAULT_RECOGNITION_POLICY.distance_threshold,
    liveness_passed: bool = True,
    margin: float = DEFAULT_RECOGNITION_POLICY.identity_margin,
    source: str = "face_webrtc",
) -> AttendanceResult:
    """Xử lý yêu cầu điểm danh cho sinh viên và trả về kết quả cấu trúc chuẩn hóa.

    Args:
        session_id: ID buổi học.
        student_id: ID sinh viên trong DB.
        distance: Khoảng cách Euclidean L2 nhận diện được.
        tolerance: Ngưỡng khoảng cách tối đa chấp nhận.
        liveness_passed: Trạng thái liveness đã qua kiểm tra.
        margin: Độ phân biệt margin.
        source: Nguồn điểm danh.

    Returns:
        AttendanceResult: Đối tượng kết quả điểm danh chuẩn hóa.

    Raises:
        DuplicateAttendanceError: Nếu sinh viên đã được điểm danh trước đó.
        SessionClosedError: Nếu buổi học đã đóng hoặc ngoài giờ.
        StudentNotInRosterError: Nếu sinh viên không thuộc danh sách lớp của buổi học.
    """
    now_str = utc_iso()
    if margin < DEFAULT_RECOGNITION_POLICY.identity_margin:
        return build_rejected_result(
            reason=RejectionReason.AMBIGUOUS_MATCH,
            recognized_at=now_str,
            student_id=str(student_id),
            distance=distance,
            liveness_passed=liveness_passed,
            tolerance=tolerance,
            margin=margin,
            source=source,
            policy_version=DEFAULT_RECOGNITION_POLICY.policy_version,
            distance_threshold=tolerance,
            margin_threshold=DEFAULT_RECOGNITION_POLICY.identity_margin,
            aggregation_strategy=DEFAULT_RECOGNITION_POLICY.aggregation_strategy,
            embedding_model_version=DEFAULT_RECOGNITION_POLICY.embedding_model_version,
            stable_duration_ms=0,
            recognition_policy_hash=DEFAULT_RECOGNITION_POLICY.policy_hash,
        )
    result_code, message = mark_attendance(
        session_id=session_id,
        student_id=student_id,
        distance=distance,
        identity_margin=margin,
        source=source,
        tolerance=tolerance,
    )

    if result_code == "created":
        status = "late" if "ĐI TRỄ" in message else "present"
        return build_accepted_result(
            student_id=str(student_id),
            status=status,
            distance=distance,
            liveness_passed=liveness_passed,
            recognized_at=now_str,
            tolerance=tolerance,
            margin=margin,
            source=source,
        )
    elif result_code == "already":
        raise DuplicateAttendanceError(message)
    elif result_code in {"closed", "outside"}:
        raise SessionClosedError(message)
    elif result_code in {"not_in_roster", "inactive"}:
        raise StudentNotInRosterError(message)
    else:
        return build_rejected_result(
            reason=RejectionReason.UNKNOWN_FACE,
            recognized_at=now_str,
            student_id=str(student_id),
            distance=distance,
            tolerance=tolerance,
            margin=margin,
            source=source,
        )
