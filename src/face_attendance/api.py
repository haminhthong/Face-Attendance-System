"""Module RESTful API sử dụng FastAPI cho hệ thống điểm danh khuôn mặt.

Cung cấp các endpoint tích hợp dịch vụ:
- Health check kiểm tra trạng thái hoạt động (/health).
- Danh sách buổi học (/sessions).
- Báo cáo kết quả điểm danh (/sessions/{session_id}/attendance).
- Ghi nhận biometric/manual attendance với application service.

Bảo mật bằng Header X-API-Key với cơ chế so sánh hằng số thời gian secrets.compare_digest chống Timing Attack.
"""

from __future__ import annotations

import logging
import math
import secrets
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .application import (
    record_biometric_attendance,
    record_manual_attendance,
)
from .config import API_KEY, RECOGNITION_POLICY
from .database import (
    attendance_report,
    init_database,
    list_sessions,
    purge_expired_biometrics,
)
from .domain import (
    AttendanceError,
    DuplicateAttendanceError,
    RecognitionDecision,
    SessionClosedError,
    StudentNotInRosterError,
)

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Khởi tạo cơ sở dữ liệu và dọn dẹp vector khuôn mặt quá hạn khi ứng dụng khởi chạy."""
    init_database()
    purge_expired_biometrics()
    yield


app = FastAPI(
    title="Face Attendance API",
    version=__version__,
    description="RESTful API nghiệp vụ cho hệ thống điểm danh sinh viên bằng khuôn mặt.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Bắt và ẩn các ngoại lệ nội bộ không mong muốn, tránh lộ stack trace/đường dẫn nội bộ."""
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    LOGGER.exception("Lỗi hệ thống không xử lý được tại endpoint %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau."},
    )


class YeuCauDiemDanhBiometric(BaseModel):
    """Schema cho quyết định nhận diện sinh trắc học có đầy đủ bằng chứng kiểm định."""

    session_id: int = Field(gt=0, description="ID của buổi học đang mở điểm danh")
    student_id: int = Field(gt=0, description="ID sinh viên được nhận diện")
    student_code: str = Field(min_length=3, max_length=30, description="Mã sinh viên")
    full_name: str = Field(default="", max_length=120, description="Họ tên sinh viên")
    distance: float = Field(ge=0.0, le=2.0, description="Khoảng cách khuôn mặt Euclidean L2")
    second_distance: float = Field(ge=0.0, description="Khoảng cách ứng viên Top-2")
    margin: float = Field(ge=0.0, description="Chênh lệch giữa Top-1 và Top-2 (Margin)")
    liveness_passed: bool = Field(description="Bằng chứng kiểm tra liveness thành công")
    confirmation_frames: int = Field(ge=1, description="Số khung hình nhận diện liên tiếp hợp lệ")
    policy_version: str = Field(min_length=1, max_length=64, description="Phiên bản chính sách")
    distance_threshold: float = Field(ge=0.0, le=2.0)
    margin_threshold: float = Field(ge=0.0, le=2.0)
    aggregation_strategy: str = Field(min_length=1, max_length=64)
    embedding_model: str = Field(min_length=1, max_length=120)
    embedding_model_version: str = Field(min_length=1, max_length=64)
    stable_duration_ms: int = Field(ge=0)
    liveness_policy: str = Field(min_length=1, max_length=64)
    recognition_policy_hash: str = Field(min_length=64, max_length=64)


class YeuCauDiemDanhManual(BaseModel):
    """Schema cho giảng viên sửa hoặc ghi nhận điểm danh thủ công (Manual Correction)."""

    session_id: int = Field(gt=0, description="ID của buổi học")
    student_id: int = Field(gt=0, description="ID sinh viên")
    status: Literal["present", "late", "absent"] = Field(
        description="Trạng thái điểm danh ('present', 'late', 'absent')"
    )
    lecturer_id: str = Field(max_length=120, description="Mã hoặc tên giảng viên thực hiện")
    reason: str = Field(max_length=500, description="Lý do điều chỉnh hoặc điểm danh thay thế")


def xac_thuc_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency xác thực API Key từ Header 'X-API-Key'.

    Sử dụng secrets.compare_digest để chống tấn công Timing Attack.

    Raises:
        HTTPException(503): Nếu chưa cấu hình biến môi trường FACE_ATTENDANCE_API_KEY.
        HTTPException(401): Nếu API key không khớp hoặc bị thiếu.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API chưa được bật vì chưa cấu hình khóa truy cập (FACE_ATTENDANCE_API_KEY).",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Khóa API không hợp lệ.")


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ API")
def health() -> dict[str, str]:
    """Endpoint công khai dùng cho Load Balancer hoặc Docker health check."""
    return {"status": "ok"}


@app.get(
    "/policy",
    dependencies=[Depends(xac_thuc_api_key)],
    summary="Lấy phiên bản policy đang chạy",
)
def policy_metadata() -> dict[str, object]:
    """Expose metadata để client và audit tool không phải tự đoán threshold."""
    return {
        "policy_version": RECOGNITION_POLICY.policy_version,
        "policy_hash": RECOGNITION_POLICY.policy_hash,
        "embedding_model": RECOGNITION_POLICY.embedding_model,
        "embedding_model_version": RECOGNITION_POLICY.embedding_model_version,
        "embedding_dimension": RECOGNITION_POLICY.embedding_dimension,
        "aggregation_strategy": RECOGNITION_POLICY.aggregation_strategy,
        "distance_threshold": RECOGNITION_POLICY.distance_threshold,
        "identity_margin": RECOGNITION_POLICY.identity_margin,
        "calibrated": RECOGNITION_POLICY.calibrated,
    }


@app.get(
    "/sessions",
    dependencies=[Depends(xac_thuc_api_key)],
    summary="Lấy danh sách các buổi học",
)
def danh_sach_buoi_hoc() -> list[dict[str, object]]:
    """Trả về danh sách tất cả các buổi học kèm thông tin môn học và trạng thái."""
    return [dict(row) for row in list_sessions()]


@app.get(
    "/sessions/{session_id}/attendance",
    dependencies=[Depends(xac_thuc_api_key)],
    summary="Trích xuất báo cáo điểm danh của một buổi học",
)
def bao_cao_buoi_hoc(session_id: int) -> list[dict[str, object]]:
    """Trả về sinh viên, trạng thái điểm danh và khoảng cách nhận diện."""
    report = attendance_report(session_id)
    return report.astype(object).where(report.notna(), None).to_dict(orient="records")


@app.post(
    "/attendance/biometric",
    summary="Ghi nhận điểm danh sinh trắc học có kèm bằng chứng quyết định (Recognition Decision)",
    dependencies=[Depends(xac_thuc_api_key)],
)
def attendance_biometric(request: YeuCauDiemDanhBiometric) -> dict[str, Any]:
    """Nhận RecognitionDecision đầy đủ từ vision pipeline và thực hiện ghi nhận."""
    if request.second_distance < request.distance or not math.isclose(
        request.margin,
        request.second_distance - request.distance,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise HTTPException(
            status_code=422,
            detail="Evidence không nhất quán: margin phải bằng Top-2 trừ Top-1.",
        )
    policy_matches = RECOGNITION_POLICY.matches_evidence(
        policy_version=request.policy_version,
        distance_threshold=request.distance_threshold,
        margin_threshold=request.margin_threshold,
        aggregation_strategy=request.aggregation_strategy,
        embedding_model=request.embedding_model,
        embedding_model_version=request.embedding_model_version,
        confirmation_frames=request.confirmation_frames,
        stable_duration_ms=request.stable_duration_ms,
        liveness_policy=request.liveness_policy,
        recognition_policy_hash=request.recognition_policy_hash,
    )
    if not policy_matches:
        raise HTTPException(
            status_code=409,
            detail="Recognition policy của client không khớp policy đang chạy trên server.",
        )
    decision = RecognitionDecision(
        student_id=request.student_id,
        student_code=request.student_code,
        full_name=request.full_name,
        distance=request.distance,
        second_distance=request.second_distance,
        margin=request.margin,
        liveness_passed=request.liveness_passed,
        confirmation_frames=request.confirmation_frames,
        policy_version=request.policy_version,
        distance_threshold=request.distance_threshold,
        margin_threshold=request.margin_threshold,
        aggregation_strategy=request.aggregation_strategy,
        embedding_model=request.embedding_model,
        embedding_model_version=request.embedding_model_version,
        stable_duration_ms=request.stable_duration_ms,
        liveness_policy=request.liveness_policy,
        recognition_policy_hash=request.recognition_policy_hash,
    )
    try:
        res = record_biometric_attendance(
            session_id=request.session_id,
            decision=decision,
        )
        return res.to_dict()
    except DuplicateAttendanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except SessionClosedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except StudentNotInRosterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AttendanceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post(
    "/attendance/manual",
    summary="Giảng viên điều chỉnh hoặc điểm danh thủ công (Manual Correction)",
    dependencies=[Depends(xac_thuc_api_key)],
)
def attendance_manual(request: YeuCauDiemDanhManual) -> dict[str, Any]:
    """Cho phép giảng viên can thiệp hoặc sửa đổi điểm danh kèm lý do và lưu audit log."""
    try:
        res = record_manual_attendance(
            session_id=request.session_id,
            student_id=request.student_id,
            status=request.status,
            lecturer_id=request.lecturer_id,
            reason=request.reason,
        )
        return res.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        LOGGER.exception("Lỗi khi điều chỉnh điểm danh thủ công")
        raise HTTPException(
            status_code=500, detail="Không thể điều chỉnh điểm danh do lỗi hệ thống."
        ) from None
