"""Module RESTful API sử dụng FastAPI cho hệ thống điểm danh khuôn mặt.

Cung cấp 3 endpoint thực tế phục vụ AI Engineer serving:
1. GET /health: Kiểm tra trạng thái dịch vụ (Health check).
2. POST /recognize: Nhận diện khuôn mặt từ vector đặc trưng (128D) hoặc ảnh.
3. GET /sessions/{session_id}/attendance: Lấy báo cáo điểm danh của một buổi học.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import API_KEY, DEFAULT_CONFIG
from .database import attendance_report, init_database
from .matcher import match_face
from .recognition import load_templates

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Khởi tạo cơ sở dữ liệu khi khởi động ứng dụng API."""
    init_database()
    yield


app = FastAPI(
    title="Face Attendance API",
    version=__version__,
    description="RESTful API phục vụ nhận diện và quản trị điểm danh sinh viên bằng khuôn mặt.",
    lifespan=lifespan,
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Xác thực API Key nếu hệ thống cấu hình FACE_ATTENDANCE_API_KEY."""
    if not API_KEY:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Khóa API không hợp lệ hoặc bị thiếu.")


class RecognizeRequest(BaseModel):
    """Schema yêu cầu nhận diện khuôn mặt."""

    embedding: list[float] = Field(..., description="Vector đặc trưng khuôn mặt 128 chiều")
    session_id: int | None = Field(
        default=None, description="ID buổi học để lọc theo danh sách lớp"
    )


class RecognizeResponse(BaseModel):
    """Schema kết quả nhận diện khuôn mặt."""

    matched: bool
    student_id: int | None = None
    student_code: str | None = None
    full_name: str | None = None
    distance: float
    margin: float
    status: str


@app.get("/health", summary="Kiểm tra trạng thái hoạt động")
def health() -> dict[str, str]:
    """Endpoint kiểm tra sức khỏe của dịch vụ API."""
    return {"status": "ok"}


@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Nhận diện khuôn mặt từ vector đặc trưng 128D",
)
def recognize(request: RecognizeRequest) -> RecognizeResponse:
    """So khớp vector khuôn mặt đầu vào với mẫu sinh viên đã đăng ký.

    Áp dụng thuật toán Open-Set Recognition:
    - Khoảng cách L2 <= 0.50
    - Margin (Top-2 - Top-1) >= 0.05
    """
    if len(request.embedding) != 128:
        raise HTTPException(
            status_code=422,
            detail=f"Vector khuôn mặt phải có đúng 128 chiều (nhận được {len(request.embedding)}).",
        )

    templates = load_templates(request.session_id)
    if not templates:
        return RecognizeResponse(
            matched=False,
            distance=float("inf"),
            margin=float("inf"),
            status="no_templates",
        )

    import numpy as np

    vector = np.array(request.embedding, dtype=np.float64)
    result = match_face(
        vector,
        templates,
        distance_threshold=DEFAULT_CONFIG.distance_threshold,
        identity_margin=DEFAULT_CONFIG.identity_margin,
        top_k=DEFAULT_CONFIG.top_k,
    )

    if result.is_matched and result.template:
        return RecognizeResponse(
            matched=True,
            student_id=result.template.student_id,
            student_code=result.template.student_code,
            full_name=result.template.full_name,
            distance=round(result.distance, 4),
            margin=round(result.margin, 4),
            status="matched",
        )

    return RecognizeResponse(
        matched=False,
        distance=round(result.distance, 4),
        margin=round(result.margin, 4),
        status="unknown",
    )


@app.get(
    "/sessions/{session_id}/attendance",
    dependencies=[Depends(verify_api_key)],
    summary="Lấy báo cáo điểm danh của một buổi học",
)
def get_session_attendance(session_id: int) -> list[dict[str, Any]]:
    """Trả về danh sách sinh viên và trạng thái điểm danh trong buổi học."""
    report = attendance_report(session_id)
    return report.astype(object).where(report.notna(), None).to_dict(orient="records")
