"""Module giao diện người dùng (Streamlit Dashboard).

Cung cấp hệ thống giao diện cao cấp, trực quan và toàn diện cho:
1. Bảng điều khiển tổng quan (Dashboard Overview & KPIs)
2. Điểm danh trực tiếp (WebRTC Realtime & Camera Snapshot Fallback)
3. Quản lý sinh viên & Thu hồi sinh trắc học (Enrollment & Biometric Revocation)
4. Quản lý môn học, danh sách lớp & buổi học (Courses, Roster, Sessions Snapshot)
5. Báo cáo điểm danh & Điều chỉnh thủ công (Reports, Export UTF-8 CSV, Manual Override)
6. Cấu hình & Tham số hệ thống (Recognition Config & SQLite Diagnostics)
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from datetime import time as dt_time
from typing import Any

import pandas as pd
import streamlit as st

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    WebRtcMode = None  # type: ignore[assignment, misc]
    webrtc_streamer = None  # type: ignore[assignment]

from .attendance import (
    DuplicateAttendanceError,
    record_biometric_attendance,
    record_manual_attendance,
)
from .config import (
    ADMIN_PIN,
    CONFIRMATION_FRAMES,
    DB_PATH,
    DEFAULT_CONFIG,
    FACE_DISTANCE_THRESHOLD,
    IDENTITY_MARGIN,
    PROCESS_EVERY_N_FRAMES,
)
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
from .enrollment import decode_and_validate_face, enroll_student
from .matcher import match_face
from .recognition import (
    AttendanceVideoProcessor,
    RecognitionEngine,
    load_templates,
)
from .styles import (
    render_hero_banner,
    render_kpi_card,
    render_student_recognition_card,
)
from .utils import display_datetime, local_datetime, utc_now


def session_label(row: sqlite3.Row) -> str:
    """Tạo chuỗi nhãn hiển thị thân thiện cho buổi học."""
    return (
        f"#{row['id']} · {row['course_code']} · {row['session_name']} · "
        f"{display_datetime(row['start_at_utc'])}"
    )


def render_admin_auth() -> bool:
    """Xác thực PIN quản trị viên với giao diện sang trọng."""
    if st.session_state.get("admin_authenticated", False):
        return True

    st.markdown(
        """
        <div style="max-width: 480px; margin: 2rem auto; padding: 2.2rem; background: white; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔐</div>
            <h3 style="margin: 0; font-weight: 800; color: #1e1b4b;">Khu vực Quản trị viên</h3>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0 1.5rem 0;">Vui lòng nhập mã PIN quản trị để truy cập cấu hình học vụ và dữ liệu sinh trắc học.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        col_pad1, col_form, col_pad2 = st.columns([1, 1.2, 1])
        with col_form:
            with st.form("admin_login_form"):
                pin = st.text_input(
                    "Mã PIN Quản trị", type="password", placeholder="Nhập mã PIN..."
                )
                submitted = st.form_submit_button(
                    "Xác thực Đăng nhập", use_container_width=True, type="primary"
                )

            if submitted:
                if pin == ADMIN_PIN:
                    st.session_state.admin_authenticated = True
                    st.toast("Đăng nhập quản trị thành công!", icon="✅")
                    st.rerun()
                else:
                    st.error("Mã PIN không đúng. Vui lòng kiểm tra lại.")

    return False


# ==============================================================================
# 1. BẢNG ĐIỀU KHIỂN TỔNG QUAN (DASHBOARD OVERVIEW)
# ==============================================================================


def render_overview_page() -> None:
    """Trang tổng quan hệ thống, thống kê số liệu và quy trình nhận diện."""
    render_hero_banner(
        title="Hệ thống Điểm danh Sinh viên Thông minh",
        subtitle="Giải pháp điểm danh tự động tích hợp Computer Vision, nhận diện tập mở (Open-Set Recognition), kiểm tra liveness chớp mắt (EAR) và xác thực ổn định đa khung hình.",
        badge_text="Realtime AI",
    )

    # Thống kê tổng thể
    students = student_table()
    total_students = len(students)
    courses = list_courses()
    total_courses = len(courses)
    sessions = list_sessions()
    open_sessions = [s for s in sessions if s["status"] == "open"]

    with get_connection() as conn:
        today_date = utc_now().strftime("%Y-%m-%d")
        today_attendance = conn.execute(
            "SELECT COUNT(*) as count FROM attendance WHERE check_in_at_utc LIKE ?",
            (f"{today_date}%",),
        ).fetchone()
        today_count = today_attendance["count"] if today_attendance else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Tổng Sinh viên",
            total_students,
            icon="👨‍🎓",
            subtext="Hồ sơ trong hệ thống",
            variant="primary",
        )
    with c2:
        render_kpi_card(
            "Môn học Đang quản lý",
            total_courses,
            icon="📚",
            subtext="Học phần giảng dạy",
            variant="info",
        )
    with c3:
        render_kpi_card(
            "Buổi học Đang mở",
            len(open_sessions),
            icon="🟢",
            subtext=f"Trên tổng {len(sessions)} buổi",
            variant="success" if open_sessions else "warning",
        )
    with c4:
        render_kpi_card(
            "Lượt Điểm danh Hôm nay",
            today_count,
            icon="⏱️",
            subtext=f"Ngày {today_date}",
            variant="primary",
        )

    st.write("")
    st.write("")

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.subheader("Buổi học đang hoạt động & Gần đây")
        if not sessions:
            st.info(
                "Hệ thống chưa có buổi học nào. Giảng viên hãy vào tab **Môn học & Buổi học** để tạo buổi học mới."
            )
        else:
            session_data = []
            for s in sessions[:6]:
                status_badge = {
                    "open": "🟢 Đang mở điểm danh",
                    "closed": "⚪ Đã kết thúc",
                    "scheduled": "🟡 Đã lên lịch",
                }.get(s["status"], s["status"])
                session_data.append(
                    {
                        "Buổi học": f"#{s['id']} - {s['session_name']}",
                        "Mã môn": s["course_code"],
                        "Tên môn học": s["course_name"],
                        "Bắt đầu": display_datetime(s["start_at_utc"]),
                        "Trạng thái": status_badge,
                    }
                )
            st.dataframe(pd.DataFrame(session_data), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("Kiến trúc luồng xử lý 8 bước")
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; font-size: 0.88rem; line-height: 1.7;">
                <div>1️⃣ <strong>Enrollment Quality Gate:</strong> ≥5 ảnh, đủ sáng, không mờ.</div>
                <div>2️⃣ <strong>Identity Consistency:</strong> Khoảng cách pairwise ≤ 0.50.</div>
                <div>3️⃣ <strong>128D Embeddings:</strong> Lưu BLOB, không lưu file ảnh thô.</div>
                <div>4️⃣ <strong>Single-Face Gate:</strong> Bắt buộc duy nhất 1 mặt trước camera.</div>
                <div>5️⃣ <strong>Open-Set Recognition:</strong> L2 Top-2 Mean (Dist ≤ 0.50, Margin ≥ 0.05).</div>
                <div>6️⃣ <strong>Basic Blink Challenge:</strong> FSM Eye Aspect Ratio (EAR).</div>
                <div>7️⃣ <strong>Temporal Confirmation:</strong> ≥3 frame & ≥500ms ổn định.</div>
                <div>8️⃣ <strong>Auto Attendance:</strong> Tự động phân loại Có mặt / Đi trễ.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 2. ĐIỂM DANH TRỰC TIẾP (LIVE ATTENDANCE)
# ==============================================================================


def render_attendance_page() -> None:
    """Trang điểm danh trực tiếp: Hỗ trợ cả WebRTC Video Stream và Camera Snapshot."""
    render_hero_banner(
        title="Điểm danh Trực tiếp",
        subtitle="Nhận diện khuôn mặt sinh viên qua camera theo thời gian thực. Hệ thống tự động xác thực danh tính, kiểm tra chớp mắt và ghi nhận học vụ.",
        badge_text="Live Camera",
    )

    open_sessions = list_sessions("open")
    if not open_sessions:
        st.markdown(
            """
            <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 12px; padding: 2rem; text-align: center; margin: 1.5rem 0;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔔</div>
                <h3 style="color: #92400e; margin: 0;">Hiện chưa có buổi học nào được mở điểm danh</h3>
                <p style="color: #b45309; font-size: 0.95rem; margin-top: 0.5rem;">
                    Giảng viên cần vào phần <strong>Quản trị &gt; Môn học &amp; Buổi học</strong> để chuyển trạng thái buổi học sang <em>'open'</em>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    session_options = {session_label(row): row for row in open_sessions}
    selected_label = st.selectbox("📌 Chọn buổi học đang mở điểm danh:", list(session_options))
    selected = session_options[selected_label]
    session_id = int(selected["id"])

    templates = load_templates(session_id)
    unique_students = len({item.student_id for item in templates})

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Môn học", f"{selected['course_code']} - {selected['course_name']}")
    with m2:
        st.metric("Sinh viên trong danh sách", unique_students)
    with m3:
        st.metric("Ảnh mẫu tham chiếu (128D)", len(templates))
    with m4:
        st.metric(
            "Ngưỡng nhận diện (L2)", f"≤ {FACE_DISTANCE_THRESHOLD:.2f} (Δ≥{IDENTITY_MARGIN:.2f})"
        )

    if not templates:
        st.error(
            "⚠️ Chưa có mẫu khuôn mặt nào của sinh viên cho buổi học này. "
            "Vui lòng vào tab Quản lý sinh viên để tải lên ít nhất 5 ảnh mẫu cho sinh viên."
        )
        return

    st.divider()

    # Hỗ trợ 2 chế độ camera
    cam_mode = st.radio(
        "Lựa chọn chế độ Camera:",
        [
            "🎥 Camera Trực tiếp (WebRTC Realtime)",
            "📷 Chụp ảnh / Tải ảnh Điểm danh (Snapshot Fast Check-in)",
        ],
        horizontal=True,
    )

    if cam_mode == "🎥 Camera Trực tiếp (WebRTC Realtime)":
        if webrtc_streamer is None or WebRtcMode is None:
            st.warning(
                "⚠️ Thư viện `streamlit-webrtc` chưa được cài đặt trên môi trường hiện tại. Vui lòng chuyển sang chế độ **Chụp ảnh Điểm danh (Snapshot)** bên cạnh để điểm danh."
            )
        else:
            st.caption(
                "💡 **Hướng dẫn:** Nhìn thẳng vào webcam, giữ ổn định khuôn mặt trong 3 khung hình liên tiếp và chớp mắt một lần để hoàn tất điểm danh."
            )
            engine = RecognitionEngine(session_id, require_blink=True)
            context = webrtc_streamer(
                key=f"attendance-live-{session_id}",
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

    else:
        st.caption(
            "💡 **Chế độ Snapshot:** Chụp trực tiếp từ camera hoặc tải ảnh chân dung sinh viên để hệ thống trích xuất vector 128D và đối soát tức thì."
        )
        c_cam, c_up = st.columns(2)
        with c_cam:
            captured_photo = st.camera_input("Chụp ảnh điểm danh từ camera thiết bị")
        with c_up:
            uploaded_photo = st.file_uploader(
                "Hoặc tải ảnh chụp khuôn mặt", type=["jpg", "jpeg", "png"]
            )

        photo_source = captured_photo or uploaded_photo
        if photo_source is not None:
            with st.spinner("Đang phân tích khuôn mặt và so khớp Open-Set..."):
                try:
                    bytes_data = photo_source.getvalue()
                    sample = decode_and_validate_face(bytes_data)

                    match_res = match_face(
                        embedding=sample.embedding,
                        templates=templates,
                        distance_threshold=FACE_DISTANCE_THRESHOLD,
                        identity_margin=IDENTITY_MARGIN,
                        top_k=2,
                    )

                    if match_res.is_matched and match_res.template:
                        tpl = match_res.template
                        try:
                            att_res = record_biometric_attendance(
                                session_id=session_id,
                                student_id=tpl.student_id,
                                distance=match_res.distance,
                                margin=match_res.margin,
                            )
                            render_student_recognition_card(
                                student_code=tpl.student_code,
                                full_name=tpl.full_name,
                                class_name=selected["course_name"],
                                status=att_res.status,
                                distance=match_res.distance,
                                margin=match_res.margin,
                                check_in_time=display_datetime(att_res.check_in_at_utc),
                            )
                            st.toast(f"Đã điểm danh thành công: {tpl.full_name}!", icon="🎉")
                        except DuplicateAttendanceError:
                            st.warning(
                                f"Sinh viên **{tpl.full_name} ({tpl.student_code})** đã điểm danh trước đó trong buổi học này."
                            )
                    else:
                        st.error(
                            f"Khuôn mặt không xác định được danh tính (Unknown Rejection). "
                            f"Khoảng cách tốt nhất: {match_res.distance:.4f} (yêu cầu ≤ {FACE_DISTANCE_THRESHOLD:.2f}), "
                            f"Margin chênh lệch: {match_res.margin:.4f} (yêu cầu ≥ {IDENTITY_MARGIN:.2f})."
                        )
                except Exception as exc:
                    st.error(f"Lỗi nhận diện: {exc}")

    # Bảng danh sách điểm danh thời gian thực
    st.divider()
    st.subheader("Bảng theo dõi điểm danh buổi học hiện tại")
    report = attendance_report(session_id)
    checked_in = report[report["Trạng thái"] != "Vắng"] if not report.empty else pd.DataFrame()
    if not checked_in.empty:
        st.dataframe(
            checked_in[
                [
                    "MSSV",
                    "Họ tên",
                    "Lớp",
                    "Trạng thái",
                    "Thời gian điểm danh",
                    "Khoảng cách",
                    "Margin",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Chưa có lượt sinh viên nào điểm danh trong buổi học này.")


# ==============================================================================
# 3. QUẢN LÝ SINH VIÊN & KHUÔN MẶT (STUDENTS & BIOMETRICS)
# ==============================================================================


def render_student_management() -> None:
    """Quản lý sinh viên, đăng ký mẫu khuôn mặt và thu hồi dữ liệu sinh trắc học."""
    render_hero_banner(
        title="Quản lý Sinh viên & Dữ liệu Sinh trắc học",
        subtitle="Đăng ký hồ sơ sinh viên kèm vector đặc trưng 128D qua cổng kiểm tra chất lượng ảnh nghiêm ngặt. Hỗ trợ quyền thu hồi dữ liệu sinh trắc học bảo vệ quyền riêng tư.",
        badge_text="Biometrics",
    )

    tab_enroll, tab_roster, tab_revoke = st.tabs(
        [
            "➕ Đăng ký Sinh viên Mới",
            "📋 Danh bạ Sinh viên",
            "🗑️ Thu hồi Dữ liệu Sinh trắc học",
        ]
    )

    with tab_enroll:
        st.subheader("Đăng ký Mẫu Khuôn mặt Sinh viên")
        st.markdown(
            """
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem; font-size: 0.9rem;">
                <strong style="color: #166534;">Yêu cầu kiểm tra chất lượng ảnh mẫu (Quality Gates):</strong>
                <ul style="margin: 0.5rem 0 0 0; padding-left: 1.4rem; color: #15803d;">
                    <li>Cần tối thiểu <strong>5 ảnh chân dung khác nhau</strong> (ở nhiều góc nhìn và điều kiện ánh sáng tự nhiên).</li>
                    <li>Mỗi ảnh phải có <strong>chính xác 1 khuôn mặt</strong> (kích thước mặt tối thiểu 100 × 100 px).</li>
                    <li>Ảnh phải sắc nét (Laplacian variance ≥ 40.0) và đủ sáng (trung bình thang xám 40.0 - 220.0).</li>
                    <li>Hệ thống chỉ trích xuất và lưu trữ <strong>vector 128 chiều</strong>, tuyệt đối không lưu file ảnh thô.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("student_enroll_form"):
            col1, col2, col3 = st.columns(3)
            student_code = col1.text_input("Mã số sinh viên (MSSV) *", placeholder="23DH113428")
            full_name = col2.text_input("Họ và tên *", placeholder="Hà Minh Thông")
            class_name = col3.text_input("Lớp sinh hoạt *", placeholder="23DTH01")

            uploaded = st.file_uploader(
                "Tải lên các ảnh chân dung (tối thiểu 5 ảnh) *",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
            )
            captured = st.camera_input("Hoặc bổ sung ảnh chụp trực tiếp từ camera")

            consent = st.checkbox(
                "Sinh viên đã được thông báo và đồng ý cung cấp dữ liệu khuôn mặt phục vụ công tác điểm danh học vụ (Biometric Privacy Consent).",
                value=True,
            )

            submit_enroll = st.form_submit_button(
                "Tiến hành Đăng ký Sinh viên", type="primary", use_container_width=True
            )

        if submit_enroll:
            if not student_code.strip() or not full_name.strip():
                st.error("Vui lòng điền đầy đủ Mã sinh viên và Họ tên.")
            elif not consent:
                st.error(
                    "Bắt buộc phải có sự đồng ý (Consent) của sinh viên trước khi xử lý dữ liệu khuôn mặt."
                )
            else:
                sources: list[Any] = list(uploaded or [])
                if captured is not None:
                    sources.append(captured)

                if len(sources) < 5:
                    st.error(
                        f"Bạn mới tải lên {len(sources)} ảnh. Cần tối thiểu 5 ảnh hợp lệ để đăng ký."
                    )
                else:
                    with st.spinner("Đang kiểm tra chất lượng và trích xuất vector 128D..."):
                        try:
                            saved_count, warnings = enroll_student(
                                student_code=student_code,
                                full_name=full_name,
                                class_name=class_name,
                                image_sources=sources,
                                consent_given=consent,
                            )
                            st.success(
                                f"🎉 Đã đăng ký thành công sinh viên **{full_name}** với **{saved_count} vector khuôn mặt**."
                            )
                            for w in warnings:
                                st.warning(f"Lưu ý: {w}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Đăng ký thất bại: {exc}")

    with tab_roster:
        st.subheader("Danh bạ Sinh viên Hệ thống")
        students = student_table()
        if not students.empty:
            search_query = st.text_input(
                "🔍 Tìm kiếm sinh viên theo MSSV, Họ tên hoặc Lớp:",
                placeholder="Nhập từ khóa tìm kiếm...",
            )
            display_df = students.copy()
            if search_query.strip():
                q = search_query.strip().lower()
                display_df = display_df[
                    display_df["MSSV"].astype(str).str.lower().str.contains(q)
                    | display_df["Họ tên"].astype(str).str.lower().str.contains(q)
                    | display_df["Lớp"].astype(str).str.lower().str.contains(q)
                ]

            st.dataframe(
                display_df.drop(columns=["id"], errors="ignore"),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Hiển thị {len(display_df)} / {len(students)} sinh viên.")
        else:
            st.info("Chưa có sinh viên nào trong hệ thống.")

    with tab_revoke:
        st.subheader("Thu hồi Dữ liệu Sinh trắc học (Right to Erasure)")
        st.markdown(
            """
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem; font-size: 0.9rem; color: #991b1b;">
                <strong>Lưu ý bảo vệ quyền riêng tư:</strong>
                Chức năng này cho phép xóa vĩnh viễn toàn bộ vector khuôn mặt 128D của sinh viên ra khỏi cơ sở dữ liệu khi sinh viên thu hồi sự đồng ý.
                Lịch sử điểm danh và kết quả học vụ trong quá khứ vẫn được giữ nguyên độc lập.
            </div>
            """,
            unsafe_allow_html=True,
        )
        students = student_table()
        if not students.empty:
            option_map = {
                f"{row['MSSV']} - {row['Họ tên']} (Đang có {row['Số ảnh mẫu']} vector)": int(
                    row["id"]
                )
                for _, row in students.iterrows()
            }
            selected_student_label = st.selectbox(
                "Chọn sinh viên cần thu hồi dữ liệu khuôn mặt:", list(option_map)
            )
            student_to_revoke = option_map[selected_student_label]

            confirm_revoke = st.checkbox(
                "Tôi xác nhận sinh viên đã yêu cầu thu hồi và chấp nhận xóa toàn bộ vector mẫu của sinh viên này."
            )
            if st.button("Tiến hành Thu hồi Dữ liệu", type="primary", disabled=not confirm_revoke):
                deleted_rows = remove_student_biometrics(student_to_revoke)
                st.success(
                    f"Đã thu hồi và xóa thành công {deleted_rows} vector khuôn mặt của sinh viên."
                )
                st.rerun()
        else:
            st.info("Chưa có dữ liệu sinh viên để thu hồi.")


# ==============================================================================
# 4. QUẢN LÝ MÔN HỌC, DANH SÁCH LỚP & BUỔI HỌC
# ==============================================================================


def render_course_session_management() -> None:
    """Quản lý học phần, xếp lớp môn học và tạo buổi học snapshot."""
    render_hero_banner(
        title="Quản lý Môn học & Buổi học",
        subtitle="Thiết lập danh mục học phần, phân bổ sinh viên vào môn học và tạo buổi học điểm danh với cơ chế Session Roster Snapshot an toàn.",
        badge_text="Curriculum",
    )

    tab_course, tab_roster, tab_session = st.tabs(
        [
            "📚 Danh mục Môn học",
            "👥 Xếp Danh sách Môn học",
            "🗓️ Buổi học Điểm danh",
        ]
    )

    with tab_course:
        st.subheader("Thêm Môn học Mới")
        with st.form("create_course_form"):
            c1, c2, c3 = st.columns(3)
            course_code = c1.text_input("Mã môn học *", placeholder="CS101")
            course_name = c2.text_input("Tên môn học *", placeholder="Nhập môn Trí tuệ Nhân tạo")
            lecturer = c3.text_input("Giảng viên phụ trách *", placeholder="TS. Trần Văn Bình")
            create_btn = st.form_submit_button("Tạo Môn học Mới", type="primary")

        if create_btn:
            try:
                create_course(course_code, course_name, lecturer)
                st.success(f"Đã tạo thành công môn học: **{course_code} - {course_name}**.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error(f"Mã môn học '{course_code}' đã tồn tại trong hệ thống.")
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Danh sách Môn học Hiện hành")
        courses = list_courses()
        if courses:
            courses_df = pd.DataFrame([dict(r) for r in courses])
            st.dataframe(
                courses_df.rename(
                    columns={
                        "course_code": "Mã môn",
                        "course_name": "Tên môn học",
                        "lecturer": "Giảng viên",
                        "created_at_utc": "Ngày tạo (UTC)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Chưa có môn học nào được tạo.")

    with tab_roster:
        st.subheader("Phân bổ Sinh viên vào Môn học (Course Roster)")
        courses = list_courses()
        if not courses:
            st.info("Cần tạo môn học trước khi xếp danh sách sinh viên.")
            return

        course_map = {f"{c['course_code']} - {c['course_name']}": int(c["id"]) for c in courses}
        selected_course_label = st.selectbox("Chọn môn học:", list(course_map))
        selected_course_id = course_map[selected_course_label]

        students = student_table()
        if students.empty:
            st.info("Chưa có sinh viên nào trong danh bạ để xếp lớp.")
            return

        active_students = students[students["Hoạt động"] == "Có"]
        student_option_map = {
            f"{row['MSSV']} - {row['Họ tên']} - {row['Lớp']}": int(row["id"])
            for _, row in active_students.iterrows()
        }

        current_roster = get_course_roster(selected_course_id)
        default_selected = [
            label for label, s_id in student_option_map.items() if s_id in current_roster
        ]

        selected_labels = st.multiselect(
            "Chọn sinh viên theo học môn này:",
            list(student_option_map),
            default=default_selected,
        )

        col_save, col_all = st.columns([1, 1])
        if col_save.button("Lưu Danh sách Môn học", type="primary", use_container_width=True):
            try:
                ids = [student_option_map[label] for label in selected_labels]
                saved_count = set_course_roster(selected_course_id, ids)
                st.success(f"Đã cập nhật thành công {saved_count} sinh viên vào môn học.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with tab_session:
        st.subheader("Tạo Buổi học Mới")
        courses = list_courses()
        if not courses:
            st.info("Cần tạo môn học trước khi tạo buổi học.")
            return

        course_map = {f"{c['course_code']} - {c['course_name']}": int(c["id"]) for c in courses}
        with st.form("create_session_form"):
            c_course = st.selectbox("Môn học *", list(course_map))
            s_name = st.text_input(
                "Tên buổi học *", placeholder="Buổi 01: Giới thiệu môn học & Thị giác máy tính"
            )

            col_s1, col_s2 = st.columns(2)
            s_day = col_s1.date_input("Ngày bắt đầu", value=date.today())
            s_time = col_s2.time_input("Giờ bắt đầu", value=dt_time(7, 30))

            col_e1, col_e2 = st.columns(2)
            e_day = col_e1.date_input("Ngày kết thúc", value=date.today())
            e_time = col_e2.time_input("Giờ kết thúc", value=dt_time(11, 0))

            late_mins = st.number_input(
                "Thời gian cho phép trễ (phút) *", min_value=0, max_value=180, value=15
            )
            create_session_btn = st.form_submit_button("Khởi tạo Buổi học", type="primary")

        if create_session_btn:
            try:
                start_dt = local_datetime(s_day, s_time)
                end_dt = local_datetime(e_day, e_time)
                sess = create_attendance_session(
                    course_id=course_map[c_course],
                    session_name=s_name,
                    start_at=start_dt,
                    end_at=end_dt,
                    late_after_minutes=int(late_mins),
                )
                st.success(
                    f"Đã tạo thành công buổi học #{sess['id']} và tự động snapshot danh sách sinh viên."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Điều khiển Mở / Đóng Điểm danh Buổi học")
        sessions = list_sessions()
        if sessions:
            sess_map = {session_label(row): row for row in sessions}
            sel_sess_label = st.selectbox("Chọn buổi học:", list(sess_map))
            current_sess = sess_map[sel_sess_label]
            status_text = {
                "open": "🟢 Đang mở điểm danh",
                "closed": "⚪ Đã đóng điểm danh",
                "scheduled": "🟡 Đã lên lịch (Chưa mở)",
            }.get(current_sess["status"], current_sess["status"])
            st.write(f"Trạng thái hiện tại: **{status_text}**")

            col_open, col_close = st.columns(2)
            if col_open.button("🔓 Mở Điểm danh", use_container_width=True, type="primary"):
                change_session_status(int(current_sess["id"]), "open")
                st.success("Đã mở điểm danh cho buổi học.")
                st.rerun()
            if col_close.button("🔒 Đóng Điểm danh", use_container_width=True):
                change_session_status(int(current_sess["id"]), "closed")
                st.success("Đã đóng điểm danh cho buổi học.")
                st.rerun()


# ==============================================================================
# 5. BÁO CÁO ĐIỂM DANH & ĐIỀU CHỈNH THỦ CÔNG
# ==============================================================================


def render_reports() -> None:
    """Báo cáo thống kê chuyên cần, xuất CSV UTF-8 và điều chỉnh thủ công có audit log."""
    render_hero_banner(
        title="Báo cáo Điểm danh & Điều chỉnh Thủ công",
        subtitle="Tổng hợp dữ liệu chuyên cần buổi học, xuất báo cáo chuẩn UTF-8 tương thích Microsoft Excel và thực hiện điều chỉnh thủ công có kiểm toán (Manual Audit Override).",
        badge_text="Reports",
    )

    sessions = list_sessions()
    if not sessions:
        st.info("Chưa có buổi học nào để lập báo cáo.")
        return

    session_map = {session_label(row): row for row in sessions}
    selected_label = st.selectbox("Chọn buổi học để xem báo cáo:", list(session_map))
    selected = session_map[selected_label]
    session_id = int(selected["id"])

    report = attendance_report(session_id)
    total_enrolled = len(report)
    present_count = int((report["Trạng thái"] == "Có mặt").sum()) if not report.empty else 0
    late_count = int((report["Trạng thái"] == "Đi trễ").sum()) if not report.empty else 0
    absent_count = int((report["Trạng thái"] == "Vắng").sum()) if not report.empty else 0
    attendance_rate = (
        (present_count + late_count) / total_enrolled * 100 if total_enrolled > 0 else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng Sinh viên", total_enrolled)
    c2.metric("Có mặt Đúng giờ", present_count)
    c3.metric("Đi trễ", late_count)
    c4.metric("Vắng mặt", absent_count)
    c5.metric("Tỷ lệ Chuyên cần", f"{attendance_rate:.1f}%")

    st.write("")
    filter_status = st.radio(
        "Lọc danh sách sinh viên:",
        ["Tất cả", "Có mặt", "Đi trễ", "Vắng"],
        horizontal=True,
    )

    filtered_report = report.copy()
    if filter_status != "Tất cả" and not filtered_report.empty:
        filtered_report = filtered_report[filtered_report["Trạng thái"] == filter_status]

    st.dataframe(filtered_report, hide_index=True, use_container_width=True)

    # Xuất file CSV UTF-8-SIG
    csv_bytes = report.to_csv(index=False).encode("utf-8-sig")
    safe_code = re.sub(r"[^A-Za-z0-9_-]", "_", str(selected["course_code"]))
    st.download_button(
        label="📥 Tải Báo cáo Điểm danh (Excel CSV UTF-8)",
        data=csv_bytes,
        file_name=f"diem_danh_{safe_code}_buoi_{session_id}.csv",
        mime="text/csv",
    )

    # Điều chỉnh điểm danh thủ công (Manual Correction with Audit Log)
    st.divider()
    st.subheader("Điều chỉnh Điểm danh Thủ công (Manual Correction & Audit)")
    with st.expander("🛠️ Mở form điều chỉnh điểm danh có lưu vết kiểm toán"):
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
                sel_student = st.selectbox("Chọn sinh viên cần điều chỉnh:", list(student_map))
                new_status = st.selectbox(
                    "Trạng thái cập nhật:",
                    ["present", "late", "absent"],
                    format_func=lambda s: {"present": "Có mặt", "late": "Đi trễ", "absent": "Vắng"}[
                        s
                    ],
                )
                lecturer_name = st.text_input(
                    "Giảng viên / Người phê duyệt *", value="Giảng viên phụ trách"
                )
                reason = st.text_input(
                    "Lý do can thiệp (Bắt buộc) *",
                    placeholder="Ví dụ: Camera hỏng, xác thực thẻ SV trực tiếp...",
                )
                submit_corr = st.form_submit_button("Xác nhận Điều chỉnh Học vụ", type="primary")

            if submit_corr:
                if not reason.strip():
                    st.error("Bắt buộc phải nhập lý do can thiệp để lưu vết kiểm toán.")
                else:
                    try:
                        record_manual_attendance(
                            session_id=session_id,
                            student_id=student_map[sel_student],
                            status=new_status,
                            lecturer_id=lecturer_name.strip(),
                            reason=reason.strip(),
                        )
                        st.success(f"Đã cập nhật trạng thái điểm danh cho **{sel_student}**.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")
        else:
            st.info("Buổi học này chưa có danh sách sinh viên nào.")


# ==============================================================================
# 6. CÀI ĐẶT & THAM SỐ HỆ THỐNG
# ==============================================================================


def render_system_settings() -> None:
    """Hiển thị cấu hình nhận diện và chẩn đoán kết nối SQLite."""
    render_hero_banner(
        title="Cấu hình & Tham số Hệ thống",
        subtitle="Chi tiết các thông số kỹ thuật Computer Vision, ngưỡng nhận diện Open-Set và trạng thái cơ sở dữ liệu SQLite.",
        badge_text="Config & Health",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tham số Thuật toán Nhận diện khuôn mặt")
        config_data = [
            {
                "Tham số": "Ngưỡng khoảng cách L2 (distance_threshold)",
                "Giá trị": f"≤ {DEFAULT_CONFIG.distance_threshold:.2f}",
                "Ý nghĩa": "Ngưỡng Euclidean tối đa để chấp nhận match",
            },
            {
                "Tham số": "Độ phân biệt tối thiểu (identity_margin)",
                "Giá trị": f"≥ {DEFAULT_CONFIG.identity_margin:.2f}",
                "Ý nghĩa": "Khoảng cách giữa Top-1 và Top-2 để chống mơ hồ",
            },
            {
                "Tham số": "Chiến lược gom mẫu (top_k)",
                "Giá trị": f"K = {DEFAULT_CONFIG.top_k}",
                "Ý nghĩa": "Lấy K mẫu gần nhất tính khoảng cách trung bình",
            },
            {
                "Tham số": "Khung hình xác nhận (min_confirmations)",
                "Giá trị": f"{CONFIRMATION_FRAMES} khung hình",
                "Ý nghĩa": "Số frame liên tiếp cùng danh tính",
            },
            {
                "Tham số": "Thời gian ổn định (stable_duration_ms)",
                "Giá trị": f"{DEFAULT_CONFIG.stable_duration_ms} ms",
                "Ý nghĩa": "Tracking ổn định tối thiểu",
            },
            {
                "Tham số": "Ngưỡng nhắm mắt (eye_closed_threshold)",
                "Giá trị": f"{DEFAULT_CONFIG.eye_closed_threshold:.2f}",
                "Ý nghĩa": "Tỉ lệ EAR khi nhắm mắt",
            },
            {
                "Tham số": "Ngưỡng mở mắt (eye_open_threshold)",
                "Giá trị": f"{DEFAULT_CONFIG.eye_open_threshold:.2f}",
                "Ý nghĩa": "Tỉ lệ EAR khi mở mắt",
            },
            {
                "Tham số": "Tần suất xử lý khung hình",
                "Giá trị": f"Mỗi {PROCESS_EVERY_N_FRAMES} frame",
                "Ý nghĩa": "Tối ưu hóa hiệu năng CPU/GPU",
            },
        ]
        st.dataframe(pd.DataFrame(config_data), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Chẩn đoán Hệ thống & Cơ sở Dữ liệu")
        try:
            with get_connection() as conn:
                journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                quick_check = conn.execute("PRAGMA quick_check;").fetchone()[0]
                total_tables = conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table';"
                ).fetchone()[0]

            db_size_kb = DB_PATH.stat().st_size / 1024 if DB_PATH.exists() else 0.0
            health_data = [
                {"Chỉ số": "Đường dẫn Database", "Giá trị": str(DB_PATH)},
                {"Chỉ số": "Dung lượng Database", "Giá trị": f"{db_size_kb:.2f} KB"},
                {"Chỉ số": "Chế độ ghi (Journal Mode)", "Giá trị": str(journal_mode).upper()},
                {"Chỉ số": "Kiểm tra toàn vẹn (Integrity)", "Giá trị": f"✅ {quick_check}"},
                {"Chỉ số": "Tổng số bảng dữ liệu", "Giá trị": f"{total_tables} bảng"},
                {"Chỉ số": "Môi trường thực thi", "Giá trị": "Python 3.13 (Native / No Docker)"},
            ]
            st.dataframe(pd.DataFrame(health_data), hide_index=True, use_container_width=True)
            st.success("Cơ sở dữ liệu SQLite hoạt động ổn định và sẵn sàng phục vụ.")
        except Exception as exc:
            st.error(f"Lỗi chẩn đoán database: {exc}")


# ==============================================================================
# 7. KHU VỰC QUẢN TRỊ VIÊN
# ==============================================================================


def render_admin_page() -> None:
    """Trang quản trị toàn diện gồm 4 phân hệ chính."""
    if not render_admin_auth():
        return

    col_h, col_out = st.columns([4, 1])
    with col_h:
        st.caption("Quản trị viên đã đăng nhập.")
    with col_out:
        if st.button("🚪 Đăng xuất", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.toast("Đã đăng xuất quản trị.")
            st.rerun()

    tab_students, tab_sessions, tab_reports, tab_settings = st.tabs(
        [
            "👥 Quản lý Sinh viên",
            "📚 Môn học & Buổi học",
            "📋 Báo cáo & Điều chỉnh",
            "⚙️ Cài đặt Hệ thống",
        ]
    )

    with tab_students:
        render_student_management()
    with tab_sessions:
        render_course_session_management()
    with tab_reports:
        render_reports()
    with tab_settings:
        render_system_settings()
