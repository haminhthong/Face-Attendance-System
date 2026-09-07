"""File điểm chạy chính của ứng dụng điểm danh sinh viên bằng khuôn mặt (Streamlit Dashboard).

Khởi tạo cơ sở dữ liệu SQLite, thiết lập giao diện Streamlit wide layout,
và điều hướng giữa trang Điểm danh trực tiếp và Khu vực quản trị.
"""

import streamlit as st

from face_attendance.config import (
    APP_TITLE,
    CONFIRMATION_FRAMES,
    FACE_TOLERANCE,
    PROCESS_EVERY_N_FRAMES,
)
from face_attendance.database import init_database
from face_attendance.ui import render_admin_page, render_attendance_page

# Cấu hình trang Streamlit Dashboard
st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")


def main() -> None:
    """Hàm chính khởi chạy giao diện điểm danh và sidebar điều hướng."""
    init_database()
    st.title("🎓 " + APP_TITLE)
    st.sidebar.header("Điều hướng")
    page = st.sidebar.radio("Chức năng", ["Điểm danh", "Quản trị"])
    st.sidebar.divider()
    st.sidebar.caption(
        f"Ngưỡng nhận diện: {FACE_TOLERANCE:.2f} · "
        f"Xác nhận: {CONFIRMATION_FRAMES} khung hình · "
        f"Xử lý mỗi {PROCESS_EVERY_N_FRAMES} khung hình"
    )
    if page == "Điểm danh":
        render_attendance_page()
    else:
        render_admin_page()


if __name__ == "__main__":
    main()
