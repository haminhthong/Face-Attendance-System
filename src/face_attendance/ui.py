"""Module giao diện người dùng (User Interface) xây dựng bằng Streamlit.

Cung cấp hai khu vực chính:
1. Điểm danh trực tiếp bằng video WebRTC camera.
2. Quản trị sinh viên, môn học, buổi học, báo cáo và điều chỉnh điểm danh thủ công.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from datetime import time as dt_time
from typing import Any

import streamlit as st

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    WebRtcMode = None  # type: ignore[assignment, misc]
    webrtc_streamer = None  # type: ignore[assignment]


from .attendance import record_manual_attendance
from .config import ADMIN_PIN, DEFAULT_CONFIG
from .database import (
    attendance_report,
    change_session_status,
    create_attendance_session,
    create_course,
    get_connection,
    get_course_roster,
    list_courses,
    list_sessions,
    remove_student_biometrics,
    set_course_roster,
    student_table,
)
from .enrollment import enroll_student
from .recognition import (
    AttendanceVideoProcessor,
    RecognitionEngine,
    load_templates,
)
from .utils import display_datetime, local_datetime


def session_label(row: sqlite3.Row) -> str:
    """Tạo chuỗi nhãn hiển thị thân thiện cho buổi học."""
    return (
        f"#{row['id']} · {row['course_code']} · {row['session_name']} · "
        f"{display_datetime(row['start_at_utc'])}"
    )


def render_admin_auth() -> bool:
    """Xác thực PIN quản trị đơn giản."""
    if st.session_state.get("admin_authenticated", False):
        return True

    st.subheader("Đăng nhập Quản trị viên")
    with st.form("admin_login"):
        pin = st.text_input("PIN quản trị", type="password", placeholder="Nhập PIN")
        login = st.form_submit_button("Đăng nhập", type="primary")

    if login:
        if pin == ADMIN_PIN:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("PIN không đúng. Vui lòng thử lại.")
    return False


def render_attendance_page() -> None:
    """Trang điểm danh trực tiếp qua webcam trình duyệt (WebRTC)."""
    st.header("Điểm danh trực tiếp")
    open_sessions = list_sessions("open")
    if not open_sessions:
        st.warning(
            "Hiện chưa có buổi học nào được mở. Giảng viên cần mở buổi học trong phần Quản trị."
        )
        return

    session_options = {session_label(row): row for row in open_sessions}
    selected_label = st.selectbox("Chọn buổi học đang mở", list(session_options))
    selected = session_options[selected_label]
    templates = load_templates(int(selected["id"]))
    unique_students = len({item.student_id for item in templates})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sinh viên trong danh sách", unique_students)
    col_b.metric("Ảnh mẫu tham chiếu", len(templates))
    col_c.metric("Ngưỡng khoảng cách", f"≤ {DEFAULT_CONFIG.distance_threshold:.2f}")

    if not templates:
        st.error(
            "Chưa có mẫu khuôn mặt cho buổi học này. Hãy đăng ký sinh viên trong khu vực quản trị."
        )
        return

    st.caption(
        "Hướng dẫn: Nhìn thẳng camera, giữ ổn định và chớp mắt một lần để hoàn tất xác nhận điểm danh."
    )

    if webrtc_streamer is None or WebRtcMode is None:
        st.warning("Thư viện `streamlit-webrtc` chưa sẵn sàng trên môi trường này.")
        return

    engine = RecognitionEngine(int(selected["id"]), require_blink=True)
    context = webrtc_streamer(
        key=f"attendance-{selected['id']}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: AttendanceVideoProcessor(engine),
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )

    @st.fragment(run_every=1)
    def live_status_panel() -> None:
        processor = context.video_processor
        active_engine = processor.engine if processor is not None else engine
        event_type, message, event_at = active_engine.snapshot()
        full_message = f"{message} · {display_datetime(event_at)}"
        if event_type == "success":
            st.success(full_message)
        elif event_type == "warning":
            st.warning(full_message)
        elif event_type == "error":
            st.error(full_message)
        else:
            st.info(full_message)

    live_status_panel()


def render_student_management() -> None:
    """Quản lý sinh viên và đăng ký khuôn mặt."""
    st.subheader("Đăng ký khuôn mặt sinh viên")
    st.info(
        "Cần tối thiểu 5 ảnh/người ở góc nhìn và ánh sáng khác nhau. "
        "Hệ thống trích xuất vector đặc trưng 128D và không lưu trữ file ảnh gốc."
    )
    student_code = st.text_input("Mã sinh viên", placeholder="23DH113428")
    full_name = st.text_input("Họ và tên", placeholder="Hà Minh Thông")
    class_name = st.text_input("Lớp sinh hoạt", placeholder="23DTH01")
    uploaded = st.file_uploader(
        "Tải nhiều ảnh tham chiếu (tối thiểu 5 ảnh)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    captured = st.camera_input("Hoặc chụp ảnh từ camera", resolution="720p")
    consent = st.checkbox("Sinh viên đã đồng ý việc xử lý dữ liệu khuôn mặt phục vụ điểm danh")

    if st.button("Đăng ký sinh viên", type="primary"):
        if not consent:
            st.error("Cần có sự đồng ý của sinh viên trước khi đăng ký.")
        else:
            sources: list[Any] = list(uploaded or [])
            if captured is not None:
                sources.append(captured)
            try:
                saved, warnings = enroll_student(
                    student_code=student_code,
                    full_name=full_name,
                    class_name=class_name,
                    image_sources=sources,
                    consent_given=consent,
                )
                st.success(f"Đã lưu thành công {saved} ảnh mẫu khuôn mặt.")
                for msg in warnings:
                    st.warning(msg)
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Danh sách sinh viên")
    students = student_table()
    if not students.empty:
        st.dataframe(
            students.drop(columns=["id"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        option_map = {
            f"{row['MSSV']} - {row['Họ tên']}": int(row["id"]) for _, row in students.iterrows()
        }
        selected_label = st.selectbox(
            "Chọn sinh viên cần thu hồi dữ liệu khuôn mặt", list(option_map)
        )
        confirm_delete = st.checkbox("Xác nhận xóa toàn bộ vector khuôn mặt của sinh viên này")
        if st.button("Thu hồi dữ liệu khuôn mặt", disabled=not confirm_delete):
            remove_student_biometrics(option_map[selected_label])
            st.success("Đã thu hồi dữ liệu khuôn mặt thành công.")
            st.rerun()
    else:
        st.info("Chưa có sinh viên nào trong hệ thống.")


def render_course_session_management() -> None:
    """Quản lý môn học, danh sách lớp và tạo buổi học."""
    st.subheader("Môn học")
    with st.form("create_course"):
        c1, c2, c3 = st.columns(3)
        course_code = c1.text_input("Mã môn", placeholder="CS101")
        course_name = c2.text_input("Tên môn", placeholder="Nhập môn Trí tuệ Nhân tạo")
        lecturer = c3.text_input("Giảng viên", placeholder="TS. Trần Văn Bình")
        create_course_btn = st.form_submit_button("Thêm môn học")

    if create_course_btn:
        try:
            create_course(course_code, course_name, lecturer)
            st.success("Đã tạo môn học thành công.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Mã môn học đã tồn tại.")
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))

    courses = list_courses()
    if not courses:
        st.info("Hãy tạo môn học trước khi thiết lập danh sách và buổi học.")
        return

    st.divider()
    st.subheader("Danh sách sinh viên theo môn học")
    roster_course_map = {
        f"{row['course_code']} - {row['course_name']}": int(row["id"]) for row in courses
    }
    roster_course_label = st.selectbox(
        "Chọn môn học để xếp danh sách", list(roster_course_map), key="roster_course"
    )
    roster_course_id = roster_course_map[roster_course_label]
    students = student_table()
    active_students = students[students["Hoạt động"] == "Có"] if not students.empty else students
    student_option_map = {
        f"{row['MSSV']} - {row['Họ tên']} - {row['Lớp']}": int(row["id"])
        for _, row in active_students.iterrows()
    }
    current_roster = get_course_roster(roster_course_id)
    default_roster = [label for label, s_id in student_option_map.items() if s_id in current_roster]
    roster_labels = st.multiselect(
        "Sinh viên thuộc môn học",
        list(student_option_map),
        default=default_roster,
    )
    if st.button("Lưu danh sách môn học"):
        try:
            saved_count = set_course_roster(
                roster_course_id,
                [student_option_map[label] for label in roster_labels],
            )
            st.success(f"Đã lưu {saved_count} sinh viên vào môn học.")
            st.rerun()
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Tạo buổi học")
    course_map = {f"{row['course_code']} - {row['course_name']}": int(row["id"]) for row in courses}
    with st.form("create_session"):
        sel_course = st.selectbox("Môn học", list(course_map))
        session_name = st.text_input("Tên buổi học", placeholder="Buổi 01: Giới thiệu")
        cs1, cs2 = st.columns(2)
        start_day = cs1.date_input("Ngày bắt đầu", value=date.today())
        start_time = cs2.time_input("Giờ bắt đầu", value=dt_time(7, 30))
        ce1, ce2 = st.columns(2)
        end_day = ce1.date_input("Ngày kết thúc", value=date.today())
        end_time = ce2.time_input("Giờ kết thúc", value=dt_time(11, 0))
        late_min = st.number_input("Tính đi trễ sau (phút)", min_value=0, max_value=180, value=15)
        create_session_btn = st.form_submit_button("Tạo buổi học")

    if create_session_btn:
        try:
            create_attendance_session(
                course_id=course_map[sel_course],
                session_name=session_name,
                start_at=local_datetime(start_day, start_time),
                end_at=local_datetime(end_day, end_time),
                late_after_minutes=int(late_min),
            )
            st.success("Đã tạo buổi học và snapshot danh sách sinh viên.")
            st.rerun()
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))

    sessions = list_sessions()
    if sessions:
        st.divider()
        st.subheader("Mở/Đóng điểm danh buổi học")
        session_map = {session_label(row): row for row in sessions}
        sel_label = st.selectbox("Buổi học", list(session_map))
        sel_session = session_map[sel_label]
        st.write(f"Trạng thái hiện tại: **{sel_session['status']}**")
        col_open, col_close = st.columns(2)
        if col_open.button("Mở điểm danh", use_container_width=True):
            change_session_status(int(sel_session["id"]), "open")
            st.success("Đã mở điểm danh.")
            st.rerun()
        if col_close.button("Đóng điểm danh", use_container_width=True):
            change_session_status(int(sel_session["id"]), "closed")
            st.success("Đã đóng điểm danh.")
            st.rerun()


def render_reports() -> None:
    """Báo cáo điểm danh và điều chỉnh thủ công."""
    st.subheader("Báo cáo điểm danh")
    sessions = list_sessions()
    if not sessions:
        st.info("Chưa có buổi học nào để lập báo cáo.")
        return

    session_map = {session_label(row): row for row in sessions}
    selected_label = st.selectbox("Chọn buổi học để xem báo cáo", list(session_map))
    selected = session_map[selected_label]
    session_id = int(selected["id"])

    report = attendance_report(session_id)
    present_count = int((report["Trạng thái"] == "Có mặt").sum()) if not report.empty else 0
    late_count = int((report["Trạng thái"] == "Đi trễ").sum()) if not report.empty else 0
    absent_count = int((report["Trạng thái"] == "Vắng").sum()) if not report.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng sinh viên", len(report))
    c2.metric("Có mặt", present_count)
    c3.metric("Đi trễ", late_count)
    c4.metric("Vắng", absent_count)

    st.dataframe(report, hide_index=True, use_container_width=True)

    csv_data = report.to_csv(index=False).encode("utf-8-sig")
    safe_code = re.sub(r"[^A-Za-z0-9_-]", "_", str(selected["course_code"]))
    st.download_button(
        "Tải báo cáo CSV",
        data=csv_data,
        file_name=f"attendance_{safe_code}_session_{session_id}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Điều chỉnh điểm danh thủ công (Manual Correction)")
    with st.expander("Mở biểu mẫu can thiệp / sửa đổi điểm danh"):
        with get_connection() as conn:
            roster_rows = conn.execute(
                """
                SELECT st.id, st.student_code, st.full_name
                FROM session_enrollments se
                JOIN students st ON st.id = se.student_id
                WHERE se.session_id = ?
                ORDER BY st.student_code
                """,
                (session_id,),
            ).fetchall()

        if roster_rows:
            student_map = {
                f"{r['student_code']} - {r['full_name']}": int(r["id"]) for r in roster_rows
            }
            with st.form("manual_correction_form"):
                sel_student = st.selectbox("Chọn sinh viên", list(student_map))
                sel_status = st.selectbox(
                    "Trạng thái mới",
                    ["present", "late", "absent"],
                    format_func=lambda s: {"present": "Có mặt", "late": "Đi trễ", "absent": "Vắng"}[
                        s
                    ],
                )
                lecturer_name = st.text_input("Giảng viên phê duyệt", value="Giảng viên phụ trách")
                reason = st.text_input(
                    "Lý do can thiệp",
                    placeholder="Ví dụ: Camera mờ, xác nhận thẻ sinh viên trực tiếp",
                )
                submit_corr = st.form_submit_button("Xác nhận điều chỉnh")

            if submit_corr:
                if not reason.strip():
                    st.error("Bắt buộc phải nhập lý do can thiệp.")
                else:
                    try:
                        record_manual_attendance(
                            session_id=session_id,
                            student_id=student_map[sel_student],
                            status=sel_status,
                            lecturer_id=lecturer_name.strip(),
                            reason=reason.strip(),
                        )
                        st.success(f"Đã cập nhật điểm danh cho {sel_student}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi khi điều chỉnh: {exc}")


def render_admin_page() -> None:
    """Khu vực quản trị hệ thống."""
    st.header("Khu vực Quản trị")
    if not render_admin_auth():
        return

    if st.sidebar.button("Đăng xuất quản trị"):
        st.session_state.admin_authenticated = False
        st.rerun()

    tab_students, tab_sessions, tab_reports = st.tabs(
        ["Sinh viên & Khuôn mặt", "Môn học & Buổi học", "Báo cáo điểm danh"]
    )
    with tab_students:
        render_student_management()
    with tab_sessions:
        render_course_session_management()
    with tab_reports:
        render_reports()
