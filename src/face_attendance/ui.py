"""Module giao diện người dùng (User Interface) xây dựng bằng Streamlit.

Cung cấp hai khu vực chính:
1. Điểm danh trực tiếp bằng video WebRTC.
2. Quản trị sinh viên, môn học, buổi học, báo cáo và bảo mật.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import date
from datetime import time as dt_time
from typing import Any

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from .application import process_student_enrollment, record_manual_attendance
from .config import RECOGNITION_POLICY
from .database import (
    attendance_report,
    change_session_status,
    create_attendance_session,
    create_course,
    get_connection,
    get_course_roster,
    get_setting,
    list_courses,
    list_sessions,
    remove_student_biometrics,
    set_course_roster,
    set_setting,
    student_table,
)
from .recognition import (
    AttendanceVideoProcessor,
    RecognitionEngine,
    load_templates,
)
from .utils import display_datetime, local_datetime, make_pin_hash, verify_pin


def session_label(row: sqlite3.Row) -> str:
    """Tạo chuỗi nhãn hiển thị thân thiện cho buổi học trong selectbox."""
    return (
        f"#{row['id']} · {row['course_code']} · {row['session_name']} · "
        f"{display_datetime(row['start_at_utc'])}"
    )


def render_admin_auth() -> bool:
    """Xác thực PIN và tạm khóa 60 giây sau năm lần nhập sai.

    Returns:
        bool: True nếu admin đã đăng nhập thành công, False nếu chưa đăng nhập hoặc đang bị khóa.
    """

    stored_pin = get_setting("admin_pin_hash")
    if not stored_pin:
        st.warning("Lần chạy đầu tiên: hãy thiết lập PIN quản trị.")
        with st.form("setup_admin_pin"):
            pin_1 = st.text_input("PIN mới (6-12 chữ số)", type="password")
            pin_2 = st.text_input("Nhập lại PIN", type="password")
            submitted = st.form_submit_button("Thiết lập PIN", type="primary")
        if submitted:
            if pin_1 != pin_2:
                st.error("Hai PIN không khớp.")
            else:
                try:
                    set_setting("admin_pin_hash", make_pin_hash(pin_1))
                    st.session_state.admin_authenticated = True
                    st.success("Đã thiết lập PIN quản trị.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        return False

    if st.session_state.get("admin_authenticated", False):
        return True

    failed_attempts = int(st.session_state.get("admin_failed_attempts", 0))
    locked_until = float(st.session_state.get("admin_locked_until", 0.0))
    remaining = max(0, int(locked_until - time.time()))
    if remaining:
        st.error(f"Đăng nhập tạm khóa. Hãy thử lại sau {remaining} giây.")
        return False

    with st.form("admin_login"):
        pin = st.text_input("PIN quản trị", type="password")
        login = st.form_submit_button("Đăng nhập", type="primary")
    if login:
        if verify_pin(pin, stored_pin):
            st.session_state.admin_authenticated = True
            st.session_state.admin_failed_attempts = 0
            st.rerun()
        else:
            failed_attempts += 1
            st.session_state.admin_failed_attempts = failed_attempts
            if failed_attempts >= 5:
                st.session_state.admin_locked_until = time.time() + 60
                st.session_state.admin_failed_attempts = 0
                st.error("Sai PIN quá 5 lần. Đăng nhập bị khóa trong 60 giây.")
            else:
                st.error(f"PIN không đúng. Còn {5 - failed_attempts} lần thử.")
    return False


def render_attendance_page() -> None:
    st.header("Điểm danh trực tiếp")
    open_sessions = list_sessions("open")
    if not open_sessions:
        st.warning("Hiện chưa có buổi học nào được mở. Giáo viên cần mở buổi học trước.")
        return

    session_options = {session_label(row): row for row in open_sessions}
    selected_label = st.selectbox("Chọn buổi học", list(session_options))
    selected = session_options[selected_label]
    require_blink = st.checkbox(
        "Yêu cầu chớp mắt trước khi điểm danh",
        value=True,
        help="Đây là bước kiểm tra người thật cơ bản, không chống được mọi hình thức giả mạo.",
    )

    templates = load_templates(int(selected["id"]))
    unique_students = len({item.student_id for item in templates})
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sinh viên đã đăng ký", unique_students)
    col_b.metric("Ảnh tham chiếu", len(templates))
    col_c.metric("Ngưỡng nhận diện", f"≤ {RECOGNITION_POLICY.distance_threshold:.2f}")

    if not templates:
        st.error("Chưa có dữ liệu khuôn mặt. Hãy đăng ký sinh viên trong khu vực quản trị.")
        return

    st.caption(
        "Nhìn thẳng camera, giữ ổn định và chớp mắt một lần. "
        "Mỗi sinh viên chỉ được ghi một lần trong buổi học."
    )

    engine = RecognitionEngine(int(selected["id"]), require_blink)
    context = webrtc_streamer(
        key=f"attendance-{selected['id']}-blink-{int(require_blink)}",
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
    st.subheader("Đăng ký khuôn mặt sinh viên")
    st.info(
        "Cần tối thiểu 5 ảnh/người ở góc nhìn và ánh sáng khác nhau. "
        "Hệ thống chỉ lưu vector 128 chiều và không lưu ảnh gốc."
    )
    student_code = st.text_input("Mã sinh viên", placeholder="23DH113428")
    full_name = st.text_input("Họ và tên", placeholder="Hà Minh Thông")
    class_name = st.text_input("Lớp", placeholder="23DH... ")
    uploaded = st.file_uploader(
        "Tải nhiều ảnh tham chiếu",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    captured = st.camera_input("Hoặc chụp một ảnh từ camera", resolution="720p")
    consent = st.checkbox("Đã có sự đồng ý của sinh viên về việc xử lý dữ liệu khuôn mặt")
    if st.button("Đăng ký/Cập nhật sinh viên", type="primary"):
        if not consent:
            st.error("Cần xác nhận sự đồng ý trước khi đăng ký.")
        else:
            sources: list[Any] = list(uploaded or [])
            if captured is not None:
                sources.append(captured)
            try:
                saved, messages = process_student_enrollment(
                    student_code,
                    full_name,
                    class_name,
                    sources,
                    consent_given=consent,
                    consent_policy_version="biometric-consent-v1",
                )
                st.success(f"Đã lưu {saved} ảnh tham chiếu hợp lệ.")
                for message in messages:
                    st.warning(message)
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Danh sách sinh viên")
    students = student_table()
    st.dataframe(
        students.drop(columns=["id"], errors="ignore"),
        hide_index=True,
        use_container_width=True,
    )
    if not students.empty:
        option_map = {
            f"{row['MSSV']} - {row['Họ tên']}": int(row["id"]) for _, row in students.iterrows()
        }
        selected_label = st.selectbox("Chọn sinh viên cần thu hồi dữ liệu", list(option_map))
        confirm_delete = st.checkbox(
            "Tôi xác nhận xóa toàn bộ vector khuôn mặt và vô hiệu hóa sinh viên này"
        )
        if st.button("Thu hồi dữ liệu khuôn mặt", disabled=not confirm_delete):
            remove_student_biometrics(option_map[selected_label])
            st.success("Đã thu hồi dữ liệu khuôn mặt.")
            st.rerun()


def render_course_session_management() -> None:
    st.subheader("Môn học")
    with st.form("create_course"):
        col_1, col_2, col_3 = st.columns(3)
        course_code = col_1.text_input("Mã môn")
        course_name = col_2.text_input("Tên môn")
        lecturer = col_3.text_input("Giảng viên")
        create_course_button = st.form_submit_button("Thêm môn học")
    if create_course_button:
        try:
            create_course(course_code, course_name, lecturer)
            st.success("Đã tạo môn học.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Mã môn học đã tồn tại.")
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))

    courses = list_courses()
    if not courses:
        st.info("Hãy tạo môn học trước khi tạo buổi học.")
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
    default_roster_labels = [
        label for label, student_id in student_option_map.items() if student_id in current_roster
    ]
    roster_labels = st.multiselect(
        "Sinh viên thuộc môn học",
        list(student_option_map),
        default=default_roster_labels,
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
        selected_course = st.selectbox("Môn học", list(course_map))
        session_name = st.text_input("Tên buổi học", placeholder="Buổi 01")
        col_s1, col_s2 = st.columns(2)
        start_day = col_s1.date_input("Ngày bắt đầu", value=date.today())
        start_clock = col_s2.time_input("Giờ bắt đầu", value=dt_time(7, 0))
        col_e1, col_e2 = st.columns(2)
        end_day = col_e1.date_input("Ngày kết thúc", value=date.today())
        end_clock = col_e2.time_input("Giờ kết thúc", value=dt_time(9, 30))
        late_minutes = st.number_input(
            "Tính đi trễ sau (phút)", min_value=0, max_value=180, value=15
        )
        create_session_button = st.form_submit_button("Tạo buổi học")
    if create_session_button:
        try:
            create_attendance_session(
                course_map[selected_course],
                session_name,
                local_datetime(start_day, start_clock),
                local_datetime(end_day, end_clock),
                int(late_minutes),
            )
            st.success("Đã tạo buổi học.")
            st.rerun()
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))

    sessions = list_sessions()
    if sessions:
        st.divider()
        st.subheader("Mở/đóng buổi học")
        session_map = {session_label(row): row for row in sessions}
        selected_session_label = st.selectbox("Buổi học", list(session_map))
        selected_session = session_map[selected_session_label]
        st.write(f"Trạng thái hiện tại: **{selected_session['status']}**")
        col_open, col_close = st.columns(2)
        if col_open.button("Mở điểm danh", use_container_width=True):
            try:
                change_session_status(int(selected_session["id"]), "open")
                st.success("Đã mở điểm danh.")
                st.rerun()
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))
        if col_close.button("Đóng điểm danh", use_container_width=True):
            try:
                change_session_status(int(selected_session["id"]), "closed")
                st.success("Đã đóng điểm danh.")
                st.rerun()
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))


def render_reports() -> None:
    st.subheader("Báo cáo điểm danh")
    sessions = list_sessions()
    if not sessions:
        st.info("Chưa có buổi học để lập báo cáo.")
        return
    session_map = {session_label(row): row for row in sessions}
    selected_label = st.selectbox("Chọn buổi học để xem báo cáo", list(session_map))
    selected = session_map[selected_label]
    report = attendance_report(int(selected["id"]))
    present_count = int((report.get("Trạng thái") == "Có mặt").sum()) if not report.empty else 0
    late_count = int((report.get("Trạng thái") == "Đi trễ").sum()) if not report.empty else 0
    absent_count = int((report.get("Trạng thái") == "Vắng").sum()) if not report.empty else 0
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Đã điểm danh", present_count + late_count)
    col_2.metric("Có mặt", present_count)
    col_3.metric("Đi trễ", late_count)
    col_4.metric("Vắng", absent_count)
    st.dataframe(report, hide_index=True, use_container_width=True)
    csv_data = report.to_csv(index=False).encode("utf-8-sig")
    safe_code = re.sub(r"[^A-Za-z0-9_-]", "_", str(selected["course_code"]))
    st.download_button(
        "Tải báo cáo CSV",
        data=csv_data,
        file_name=f"attendance_{safe_code}_session_{selected['id']}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Điều chỉnh điểm danh thủ công (Human Oversight / Override)")
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
                (int(selected["id"]),),
            ).fetchall()

        if roster_rows:
            student_map = {
                f"{r['student_code']} - {r['full_name']}": int(r["id"]) for r in roster_rows
            }
            with st.form("manual_correction_form"):
                sel_student = st.selectbox("Chọn sinh viên cần điều chỉnh", list(student_map))
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
                    placeholder="Ví dụ: Camera mờ, đã xác nhận trực tiếp bằng thẻ sinh viên",
                )
                submit_corr = st.form_submit_button("Xác nhận điều chỉnh & Ghi vết kiểm toán")

            if submit_corr:
                if not reason.strip():
                    st.error("Bắt buộc phải nhập lý do can thiệp điểm danh.")
                else:
                    try:
                        record_manual_attendance(
                            session_id=int(selected["id"]),
                            student_id=student_map[sel_student],
                            status=sel_status,
                            lecturer_id=lecturer_name.strip(),
                            reason=reason.strip(),
                        )
                        st.success(f"Đã cập nhật điểm danh cho {sel_student} và lưu vết kiểm toán.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi khi cập nhật: {exc}")


def render_security_settings() -> None:
    st.subheader("Bảo mật")
    with st.form("change_pin"):
        pin_1 = st.text_input("PIN mới", type="password")
        pin_2 = st.text_input("Nhập lại PIN mới", type="password")
        change = st.form_submit_button("Đổi PIN")
    if change:
        if pin_1 != pin_2:
            st.error("Hai PIN không khớp.")
        else:
            try:
                set_setting("admin_pin_hash", make_pin_hash(pin_1))
                st.success("Đã đổi PIN.")
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Đăng xuất quản trị"):
        st.session_state.admin_authenticated = False
        st.rerun()

    st.divider()
    st.caption(
        "Dữ liệu được lưu tại SQLite; ảnh gốc không được lưu. Khi triển khai thật, "
        "hãy đặt thư mục dữ liệu trên ổ đĩa bền vững, giới hạn quyền truy cập và sao lưu định kỳ."
    )


def render_admin_page() -> None:
    st.header("Khu vực quản trị")
    if not render_admin_auth():
        return
    tab_students, tab_sessions, tab_reports, tab_security = st.tabs(
        ["Sinh viên", "Môn học & Buổi học", "Báo cáo", "Bảo mật"]
    )
    with tab_students:
        render_student_management()
    with tab_sessions:
        render_course_session_management()
    with tab_reports:
        render_reports()
    with tab_security:
        render_security_settings()
