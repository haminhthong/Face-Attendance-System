"""Module quản lý cơ sở dữ liệu SQLite cho hệ thống điểm danh khuôn mặt.

Cung cấp các thao tác lưu trữ, truy vấn cho:
- Sinh viên và mẫu vector đặc trưng (128D embeddings).
- Môn học và danh sách sinh viên theo môn.
- Buổi học điểm danh và snapshot danh sách sinh viên theo buổi.
- Ghi nhận kết quả điểm danh, chống trùng lặp và điều chỉnh thủ công.
- Báo cáo tổng hợp điểm danh buổi học.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .config import DB_PATH  # noqa: F401
from .utils import (
    display_datetime,
    normalize_course_code,
    normalize_person_name,
    normalize_student_code,
    utc_iso,
    utc_now,
)

LOGGER = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Tạo kết nối SQLite an toàn với foreign_keys và busy_timeout."""
    connection = sqlite3.connect(config.DB_PATH, timeout=15, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


def init_database() -> None:
    """Khởi tạo schema cơ sở dữ liệu và chỉ mục cần thiết."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        consent_given INTEGER NOT NULL DEFAULT 1 CHECK (consent_given IN (0, 1)),
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        blur_score REAL NOT NULL,
        brightness REAL NOT NULL,
        face_width INTEGER NOT NULL,
        face_height INTEGER NOT NULL,
        image_sha256 TEXT,
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_code TEXT NOT NULL UNIQUE,
        course_name TEXT NOT NULL,
        lecturer TEXT NOT NULL,
        created_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS course_enrollments (
        course_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        enrolled_at_utc TEXT NOT NULL,
        PRIMARY KEY (course_id, student_id),
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS attendance_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL,
        session_name TEXT NOT NULL,
        start_at_utc TEXT NOT NULL,
        end_at_utc TEXT NOT NULL,
        late_after_minutes INTEGER NOT NULL DEFAULT 15,
        status TEXT NOT NULL DEFAULT 'scheduled'
            CHECK (status IN ('scheduled', 'open', 'closed')),
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE RESTRICT
    );

    CREATE TABLE IF NOT EXISTS session_enrollments (
        session_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        enrolled_at_utc TEXT NOT NULL,
        PRIMARY KEY (session_id, student_id),
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE RESTRICT
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        check_in_at_utc TEXT NOT NULL,
        attendance_status TEXT NOT NULL
            CHECK (attendance_status IN ('present', 'late', 'absent')),
        recognition_distance REAL NOT NULL,
        identity_margin REAL NOT NULL DEFAULT 0.0,
        source TEXT NOT NULL DEFAULT 'face_webrtc',
        corrected_by TEXT,
        correction_reason TEXT,
        corrected_at_utc TEXT,
        UNIQUE (session_id, student_id),
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE RESTRICT
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
    CREATE INDEX IF NOT EXISTS idx_course_enrollments_student ON course_enrollments(student_id);
    CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
    CREATE INDEX IF NOT EXISTS idx_session_enrollments_student ON session_enrollments(student_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON attendance_sessions(status);
    """
    with get_connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(schema)


# ==============================================================================
# SINH VIÊN (STUDENTS) & VECTOR EMBEDDINGS
# ==============================================================================


def upsert_student(
    student_code: str,
    full_name: str,
    class_name: str,
    consent_given: bool = True,
    **kwargs: Any,
) -> sqlite3.Row:
    """Tạo mới hoặc cập nhật thông tin sinh viên."""
    code = normalize_student_code(student_code)
    name = normalize_person_name(full_name)
    cls_name = class_name.strip()
    now_str = utc_iso()
    consent_val = 1 if consent_given else 0

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO students(student_code, full_name, class_name, consent_given, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_code) DO UPDATE SET
                full_name = excluded.full_name,
                class_name = excluded.class_name,
                consent_given = excluded.consent_given,
                updated_at_utc = excluded.updated_at_utc
            """,
            (code, name, cls_name, consent_val, now_str, now_str),
        )
        row = conn.execute("SELECT * FROM students WHERE student_code = ?", (code,)).fetchone()
        return row


def get_student_by_code(student_code: str) -> sqlite3.Row | None:
    """Tìm sinh viên theo mã số."""
    code = normalize_student_code(student_code)
    with get_connection() as conn:
        return conn.execute("SELECT * FROM students WHERE student_code = ?", (code,)).fetchone()


def get_student_by_id(student_id: int) -> sqlite3.Row | None:
    """Tìm sinh viên theo ID."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()


def student_table() -> pd.DataFrame:
    """Lấy danh sách sinh viên kèm số lượng ảnh khuôn mặt mẫu."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.student_code AS "MSSV",
                s.full_name AS "Họ tên",
                s.class_name AS "Lớp",
                COUNT(fe.id) AS "Số ảnh mẫu",
                CASE WHEN s.active = 1 THEN 'Có' ELSE 'Không' END AS "Hoạt động",
                CASE WHEN s.consent_given = 1 THEN 'Đã đồng ý' ELSE 'Chưa' END AS "Đồng ý"
            FROM students s
            LEFT JOIN face_embeddings fe ON fe.student_id = s.id
            GROUP BY s.id
            ORDER BY s.student_code
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def save_embedding(
    student_id: int,
    embedding: np.ndarray,
    image_sha256: str = "",
    blur_score: float = 0.0,
    brightness: float = 0.0,
    face_width: int = 0,
    face_height: int = 0,
) -> bool:
    """Lưu vector khuôn mặt 128D dạng BLOB vào SQLite."""
    emb_bytes = np.asarray(embedding, dtype=np.float64).tobytes()
    now_str = utc_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO face_embeddings(
                student_id, embedding, blur_score, brightness,
                face_width, face_height, image_sha256, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                emb_bytes,
                float(blur_score),
                float(brightness),
                int(face_width),
                int(face_height),
                image_sha256,
                now_str,
            ),
        )
    return True


def get_student_embeddings(student_id: int) -> list[np.ndarray]:
    """Lấy toàn bộ vector khuôn mặt 128D của một sinh viên."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT embedding FROM face_embeddings WHERE student_id = ? ORDER BY id",
            (student_id,),
        ).fetchall()
    return [np.frombuffer(row["embedding"], dtype=np.float64).copy() for row in rows]


def remove_student_biometrics(student_id: int) -> int:
    """Xóa toàn bộ vector khuôn mặt của sinh viên khi thu hồi dữ liệu."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        return cursor.rowcount


# ==============================================================================
# MÔN HỌC (COURSES) & DANH SÁCH LỚP (ROSTER)
# ==============================================================================


def create_course(course_code: str, course_name: str, lecturer: str) -> sqlite3.Row:
    """Tạo môn học mới."""
    code = normalize_course_code(course_code)
    name = course_name.strip()
    lect = normalize_person_name(lecturer)
    now_str = utc_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO courses(course_code, course_name, lecturer, created_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            (code, name, lect, now_str),
        )
        return conn.execute("SELECT * FROM courses WHERE course_code = ?", (code,)).fetchone()


def list_courses() -> list[sqlite3.Row]:
    """Lấy danh sách tất cả các môn học."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM courses ORDER BY course_code").fetchall()


def get_course_roster(course_id: int) -> list[int]:
    """Lấy danh sách ID sinh viên thuộc môn học."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT student_id FROM course_enrollments WHERE course_id = ? ORDER BY student_id",
            (course_id,),
        ).fetchall()
    return [int(row["student_id"]) for row in rows]


def set_course_roster(course_id: int, student_ids: list[int]) -> int:
    """Cập nhật toàn bộ danh sách sinh viên của môn học."""
    now_str = utc_iso()
    with get_connection() as conn:
        conn.execute("DELETE FROM course_enrollments WHERE course_id = ?", (course_id,))
        conn.executemany(
            """
            INSERT INTO course_enrollments(course_id, student_id, enrolled_at_utc)
            VALUES (?, ?, ?)
            """,
            [(course_id, sid, now_str) for sid in set(student_ids)],
        )
    return len(student_ids)


# ==============================================================================
# BUỔI HỌC (ATTENDANCE SESSIONS) & ROSTER SNAPSHOT
# ==============================================================================


def create_attendance_session(
    course_id: int,
    session_name: str,
    start_at: datetime,
    end_at: datetime,
    late_after_minutes: int = 15,
) -> sqlite3.Row:
    """Tạo buổi học và snapshot danh sách sinh viên hiện tại vào buổi đó."""
    name = session_name.strip()
    start_iso = utc_iso(start_at)
    end_iso = utc_iso(end_at)
    now_str = utc_iso()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO attendance_sessions(
                course_id, session_name, start_at_utc, end_at_utc, late_after_minutes, status, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, 'scheduled', ?)
            """,
            (course_id, name, start_iso, end_iso, late_after_minutes, now_str),
        )
        session_id = cursor.lastrowid

        # Snapshot toàn bộ roster môn học sang session_enrollments để đảm bảo
        # lịch sử điểm danh độc lập với thay đổi roster môn học sau này.
        conn.execute(
            """
            INSERT INTO session_enrollments(session_id, student_id, enrolled_at_utc)
            SELECT ?, student_id, ?
            FROM course_enrollments
            WHERE course_id = ?
            """,
            (session_id, now_str, course_id),
        )

        return conn.execute(
            "SELECT * FROM attendance_sessions WHERE id = ?", (session_id,)
        ).fetchone()


def list_sessions(status_filter: str | None = None) -> list[sqlite3.Row]:
    """Lấy danh sách các buổi học kèm thông tin môn học."""
    query = """
    SELECT
        s.id,
        s.course_id,
        c.course_code,
        c.course_name,
        s.session_name,
        s.start_at_utc,
        s.end_at_utc,
        s.late_after_minutes,
        s.status
    FROM attendance_sessions s
    JOIN courses c ON c.id = s.course_id
    """
    params: list[Any] = []
    if status_filter:
        query += " WHERE s.status = ?"
        params.append(status_filter)
    query += " ORDER BY s.id DESC"

    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


def get_session(session_id: int) -> sqlite3.Row | None:
    """Lấy chi tiết một buổi học."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT s.*, c.course_code, c.course_name
            FROM attendance_sessions s
            JOIN courses c ON c.id = s.course_id
            WHERE s.id = ?
            """,
            (session_id,),
        ).fetchone()


def change_session_status(session_id: int, status: str) -> bool:
    """Đổi trạng thái buổi học ('scheduled', 'open', 'closed')."""
    if status not in {"scheduled", "open", "closed"}:
        raise ValueError(f"Trạng thái buổi học không hợp lệ: {status}")
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE attendance_sessions SET status = ? WHERE id = ?",
            (status, session_id),
        )
        return cursor.rowcount > 0


# ==============================================================================
# ĐIỂM DANH (ATTENDANCE) & BÁO CÁO (REPORT)
# ==============================================================================


def mark_attendance(
    session_id: int,
    student_id: int,
    distance: float,
    margin: float = 0.0,
    source: str = "face_webrtc",
) -> tuple[str, str]:
    """Ghi nhận điểm danh sinh viên trong buổi học.

    Returns:
        tuple[str, str]: (result_code, attendance_status)
        - result_code: 'created' nếu điểm danh thành công, 'already' nếu đã điểm danh trước đó.
        - attendance_status: 'present' hoặc 'late'.
    """
    now = utc_now()
    now_str = utc_iso(now)

    with get_connection() as conn:
        # Kiểm tra đã điểm danh chưa
        existing = conn.execute(
            "SELECT attendance_status FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if existing:
            return "already", str(existing["attendance_status"])

        # Lấy thông tin session để tính Present/Late
        session = conn.execute(
            "SELECT start_at_utc, late_after_minutes, status FROM attendance_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not session:
            raise ValueError(f"Không tìm thấy buổi học #{session_id}")
        if session["status"] != "open":
            raise ValueError("Buổi học hiện không mở để điểm danh.")

        # Kiểm tra student có trong session roster snapshot không
        in_roster = conn.execute(
            "SELECT 1 FROM session_enrollments WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if not in_roster:
            raise ValueError("Sinh viên không thuộc danh sách buổi học này.")

        start_dt = datetime.fromisoformat(session["start_at_utc"])
        late_limit_seconds = int(session["late_after_minutes"]) * 60
        diff_seconds = (now - start_dt).total_seconds()
        attendance_status = "present" if diff_seconds <= late_limit_seconds else "late"

        try:
            conn.execute(
                """
                INSERT INTO attendance(
                    session_id, student_id, check_in_at_utc, attendance_status,
                    recognition_distance, identity_margin, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    now_str,
                    attendance_status,
                    float(distance),
                    float(margin),
                    source,
                ),
            )
            return "created", attendance_status
        except sqlite3.IntegrityError:
            # Race condition: bản ghi vừa được ghi nhận bởi luồng song song khác
            existing_after = conn.execute(
                "SELECT attendance_status FROM attendance WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            ).fetchone()
            status_str = str(existing_after["attendance_status"]) if existing_after else "present"
            return "already", status_str


def manual_attendance_correction(
    session_id: int,
    student_id: int,
    status: str,
    lecturer_name: str,
    reason: str,
) -> bool:
    """Giảng viên điều chỉnh trạng thái điểm danh thủ công (Manual Override)."""
    if status not in {"present", "late", "absent"}:
        raise ValueError(f"Trạng thái không hợp lệ: {status}")
    if not reason.strip():
        raise ValueError("Lý do điều chỉnh không được để trống.")

    now_str = utc_iso()
    with get_connection() as conn:
        in_roster = conn.execute(
            "SELECT 1 FROM session_enrollments WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if not in_roster:
            raise ValueError("Sinh viên không thuộc danh sách buổi học này.")

        conn.execute(
            """
            INSERT INTO attendance(
                session_id, student_id, check_in_at_utc, attendance_status,
                recognition_distance, identity_margin, source,
                corrected_by, correction_reason, corrected_at_utc
            )
            VALUES (?, ?, ?, ?, 0.0, 0.0, 'manual', ?, ?, ?)
            ON CONFLICT(session_id, student_id) DO UPDATE SET
                attendance_status = excluded.attendance_status,
                corrected_by = excluded.corrected_by,
                correction_reason = excluded.correction_reason,
                corrected_at_utc = excluded.corrected_at_utc
            """,
            (
                session_id,
                student_id,
                now_str,
                status,
                lecturer_name.strip(),
                reason.strip(),
                now_str,
            ),
        )
    return True


def attendance_report(session_id: int) -> pd.DataFrame:
    """Tạo báo cáo điểm danh cho buổi học (bao gồm cả sinh viên vắng mặt)."""
    query = """
    SELECT
        s.student_code AS "MSSV",
        s.full_name AS "Họ tên",
        s.class_name AS "Lớp",
        a.attendance_status,
        a.check_in_at_utc,
        a.recognition_distance,
        a.identity_margin,
        a.source,
        a.corrected_by,
        a.correction_reason
    FROM session_enrollments se
    JOIN students s ON s.id = se.student_id
    LEFT JOIN attendance a ON a.session_id = se.session_id AND a.student_id = se.student_id
    WHERE se.session_id = ?
    ORDER BY s.student_code
    """
    with get_connection() as conn:
        rows = conn.execute(query, (session_id,)).fetchall()

    data = []
    status_map = {"present": "Có mặt", "late": "Đi trễ", "absent": "Vắng"}
    for r in rows:
        st_raw = r["attendance_status"]
        status_label = status_map.get(st_raw, "Vắng")
        check_in = display_datetime(r["check_in_at_utc"]) if r["check_in_at_utc"] else ""
        dist = f"{r['recognition_distance']:.3f}" if r["recognition_distance"] is not None else ""
        margin = f"{r['identity_margin']:.3f}" if r["identity_margin"] is not None else ""

        note_parts = []
        if r["source"] == "manual":
            note_parts.append(f"Chỉnh sửa bởi: {r['corrected_by']} ({r['correction_reason']})")
        note = "; ".join(note_parts)

        data.append(
            {
                "MSSV": r["MSSV"],
                "Họ tên": r["Họ tên"],
                "Lớp": r["Lớp"],
                "Trạng thái": status_label,
                "Thời gian điểm danh": check_in,
                "Khoảng cách": dist,
                "Margin": margin,
                "Ghi chú": note,
            }
        )

    return pd.DataFrame(data)
