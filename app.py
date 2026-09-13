"""Điểm chạy chính của ứng dụng điểm danh sinh viên bằng khuôn mặt (Streamlit Dashboard).

Khởi tạo cơ sở dữ liệu SQLite, thiết lập giao diện Design System hiện đại,
và cung cấp thanh điều hướng thông minh giữa các phân hệ chức năng.
"""

import sys
from pathlib import Path

import streamlit as st

# Tự động nạp thư mục 'src' vào sys.path để ứng dụng luôn chạy độc lập mượt mà
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_attendance.config import (
    APP_TITLE,
    CONFIRMATION_FRAMES,
    FACE_DISTANCE_THRESHOLD,
    PROCESS_EVERY_N_FRAMES,
)
from face_attendance.database import init_database, list_sessions
from face_attendance.styles import apply_custom_styles
from face_attendance.ui import (
    render_admin_auth,
    render_attendance_page,
    render_course_session_management,
    render_overview_page,
    render_reports,
    render_student_management,
    render_system_settings,
)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Hàm chính khởi chạy giao diện Dashboard và điều hướng phân hệ."""
    init_database()
    apply_custom_styles()

    # Sidebar Thương hiệu và Trạng thái
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-title">
                    <span>🎓</span> AI Attendance
                </div>
                <div class="sidebar-subtitle">Hệ sinh thái Điểm danh Sinh viên v1.2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Phân hệ chức năng",
            [
                "📊 Bảng điều khiển",
                "📸 Điểm danh trực tiếp",
                "👥 Quản lý Sinh viên",
                "📚 Môn học & Buổi học",
                "📋 Báo cáo & Điều chỉnh",
                "⚙️ Cài đặt Hệ thống",
            ],
            index=0,
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.divider()

        # Tiện ích trạng thái buổi học
        open_sessions = list_sessions("open")
        open_count = len(open_sessions)

        if open_count > 0:
            st.markdown(
                f"""
                <div class="sidebar-status-box">
                    <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #065f46;">
                        <span class="live-dot"></span> Đang có {open_count} buổi học mở
                    </div>
                    <div style="color: #64748b; font-size: 0.78rem; margin-top: 0.2rem;">Sẵn sàng nhận diện camera</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-status-box">
                    <div style="color: #92400e; font-weight: 600;">Chưa có buổi học nào mở</div>
                    <div style="color: #64748b; font-size: 0.78rem; margin-top: 0.2rem;">Mở buổi học trong Quản lý buổi học</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quản trị viên authentication toggle
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        is_admin = st.session_state.get("admin_authenticated", False)
        if is_admin:
            st.markdown(
                """
                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.82rem; color: #065f46; display: flex; justify-content: space-between; align-items: center;">
                    <span>🛡️ Quản trị viên (Active)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Đăng xuất Quản trị", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.rerun()
        else:
            st.caption("Khu vực cấu hình học vụ yêu cầu mã PIN.")

        st.caption(
            f"Ngưỡng L2: ≤{FACE_DISTANCE_THRESHOLD:.2f} · "
            f"Xác nhận: {CONFIRMATION_FRAMES} frames · "
            f"Mỗi {PROCESS_EVERY_N_FRAMES} frames"
        )

    # Điều hướng trang
    if page == "📊 Bảng điều khiển":
        render_overview_page()
    elif page == "📸 Điểm danh trực tiếp":
        render_attendance_page()
    else:
        # Các trang quản trị yêu cầu nhập PIN
        if not render_admin_auth():
            return

        if page == "👥 Quản lý Sinh viên":
            render_student_management()
        elif page == "📚 Môn học & Buổi học":
            render_course_session_management()
        elif page == "📋 Báo cáo & Điều chỉnh":
            render_reports()
        elif page == "⚙️ Cài đặt Hệ thống":
            render_system_settings()


if __name__ == "__main__":
    main()
