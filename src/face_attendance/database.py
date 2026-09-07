"""Module quản lý cơ sở dữ liệu SQLite và các nghiệp vụ điểm danh.

Bao gồm khởi tạo schema, quản lý thông tin sinh viên, lưu trữ vector đặc trưng khuôn mặt (128D),
quản lý môn học/danh sách lớp/buổi học, ghi nhận điểm danh có transaction cách ly,
chính sách lưu trữ sinh trắc học (GDPR retention) và ghi nhật ký audit log.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import config
from .config import BIOMETRIC_RETENTION_DAYS, DB_PATH  # noqa: F401 - giữ API test/legacy
from .policy import DEFAULT_RECOGNITION_POLICY, RecognitionPolicy
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
    """Tạo kết nối tới cơ sở dữ liệu SQLite với cấu hình PRAGMA an toàn đa luồng.

    Khóa ngoại (Foreign Keys) và Timeout chờ ghi dữ liệu (busy_timeout = 15s)
    được bật để đảm bảo tính toàn vẹn và chống deadlock.

    Returns:
        sqlite3.Connection: Đối tượng kết nối SQLite với row_factory dạng Row.
    """
    LOGGER.debug("GET_CONN DB_PATH=%s", config.DB_PATH)
    connection = sqlite3.connect(config.DB_PATH, timeout=15, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


def init_database() -> None:
    """Khởi tạo bảng cơ sở dữ liệu, chỉ mục (indexes) và thực hiện migration nếu cần.

    Bật chế độ ghi nhật ký trước (WAL mode) để tối ưu hiệu năng đọc/ghi song song.
    """
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = """
    CREATE TABLE IF NOT EXISTS app_settings (
        setting_key TEXT PRIMARY KEY,
        setting_value TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        consent_at_utc TEXT,
        consent_policy_version TEXT,
        consent_status TEXT NOT NULL DEFAULT 'pending'
            CHECK (consent_status IN ('pending', 'granted', 'revoked')),
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        image_sha256 TEXT NOT NULL,
        image_phash TEXT,
        blur_score REAL NOT NULL,
        brightness REAL NOT NULL,
        face_width INTEGER NOT NULL,
        face_height INTEGER NOT NULL,
        embedding_dim INTEGER NOT NULL DEFAULT 128,
        embedding_model TEXT NOT NULL DEFAULT 'dlib_face_recognition_resnet_v1',
        embedding_model_version TEXT NOT NULL DEFAULT '1',
        revoked_at_utc TEXT,
        created_at_utc TEXT NOT NULL,
        UNIQUE (student_id, image_sha256),
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
        attendance_status TEXT NOT NULL CHECK (attendance_status IN ('present', 'late', 'absent')),
        recognition_distance REAL NOT NULL,
        threshold_used REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'face_webrtc',
        identity_margin REAL DEFAULT 0.0,
        margin_threshold REAL DEFAULT 0.05,
        liveness_policy TEXT DEFAULT 'ear_blink_v1',
        recognition_policy_version TEXT DEFAULT 'face-policy-v1',
        confirmation_frames INTEGER DEFAULT 3,
        stable_duration_ms INTEGER DEFAULT 0,
        aggregation_strategy TEXT DEFAULT 'top_k_mean',
        embedding_model_version TEXT DEFAULT '1',
        recognition_policy_hash TEXT,
        original_status TEXT,
        final_status TEXT,
        corrected_by TEXT,
        correction_reason TEXT,
        corrected_at_utc TEXT,
        UNIQUE (session_id, student_id),
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE RESTRICT
    );

    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_detail TEXT NOT NULL,
        created_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS biometric_consents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        policy_version TEXT NOT NULL,
        granted_at_utc TEXT NOT NULL,
        revoked_at_utc TEXT,
        status TEXT NOT NULL CHECK (status IN ('granted', 'revoked')),
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS recognition_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        candidate_student_id INTEGER,
        decision TEXT NOT NULL,
        rejection_reason TEXT,
        distance REAL,
        margin REAL,
        policy_version TEXT NOT NULL,
        liveness_passed INTEGER NOT NULL DEFAULT 0,
        stable_duration_ms INTEGER NOT NULL DEFAULT 0,
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (candidate_student_id) REFERENCES students(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_student
        ON face_embeddings(student_id);
    CREATE INDEX IF NOT EXISTS idx_course_enrollments_student
        ON course_enrollments(student_id);
    CREATE INDEX IF NOT EXISTS idx_attendance_session
        ON attendance(session_id);
    CREATE INDEX IF NOT EXISTS idx_session_enrollments_student
        ON session_enrollments(student_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_status
        ON attendance_sessions(status);
    """
    with get_connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(schema)

        # Migration an toàn: Tự động thêm các cột bằng chứng nếu dùng DB cũ
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(attendance)").fetchall()
        }
        evidence_columns = [
            ("identity_margin", "REAL DEFAULT 0.0"),
            ("margin_threshold", "REAL DEFAULT 0.05"),
            ("liveness_policy", "TEXT DEFAULT 'ear_blink_v1'"),
            ("recognition_policy_version", "TEXT DEFAULT 'face-policy-v1'"),
            ("confirmation_frames", "INTEGER DEFAULT 3"),
            ("stable_duration_ms", "INTEGER DEFAULT 0"),
            ("aggregation_strategy", "TEXT DEFAULT 'top_k_mean'"),
            ("embedding_model_version", "TEXT DEFAULT '1'"),
            ("recognition_policy_hash", "TEXT"),
            ("original_status", "TEXT"),
            ("final_status", "TEXT"),
            ("corrected_by", "TEXT"),
            ("correction_reason", "TEXT"),
            ("corrected_at_utc", "TEXT"),
        ]
        for col_name, col_type in evidence_columns:
            if col_name not in columns:
                connection.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type}")

        student_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(students)").fetchall()
        }
        for col_name, col_type in [
            ("consent_policy_version", "TEXT"),
            ("consent_status", "TEXT NOT NULL DEFAULT 'pending'"),
        ]:
            if col_name not in student_columns:
                connection.execute(f"ALTER TABLE students ADD COLUMN {col_name} {col_type}")

        embedding_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(face_embeddings)").fetchall()
        }
        for col_name, col_type in [
            ("image_phash", "TEXT"),
            ("embedding_dim", "INTEGER NOT NULL DEFAULT 128"),
            (
                "embedding_model",
                "TEXT NOT NULL DEFAULT 'dlib_face_recognition_resnet_v1'",
            ),
            ("embedding_model_version", "TEXT NOT NULL DEFAULT '1'"),
            ("revoked_at_utc", "TEXT"),
        ]:
            if col_name not in embedding_columns:
                connection.execute(f"ALTER TABLE face_embeddings ADD COLUMN {col_name} {col_type}")

        migration_key = "session_roster_migrated_v1"
        migrated = connection.execute(
            "SELECT 1 FROM app_settings WHERE setting_key = ?", (migration_key,)
        ).fetchone()
        if migrated is None:
            # Chuyển danh sách của các buổi cũ đúng một lần khi nâng cấp dữ liệu.
            connection.execute(
                """
                INSERT OR IGNORE INTO session_enrollments(
                    session_id, student_id, enrolled_at_utc
                )
                SELECT ses.id, ce.student_id, ses.created_at_utc
                FROM attendance_sessions ses
                JOIN course_enrollments ce ON ce.course_id = ses.course_id
                """
            )

            connection.execute(
                """
                INSERT INTO app_settings(setting_key, setting_value, updated_at_utc)
                VALUES (?, '1', ?)
                """,
                (migration_key, utc_iso()),
            )


def audit(connection: sqlite3.Connection, event_type: str, detail: str) -> None:
    """Ghi lịch sử thao tác quan trọng (Audit Log) vào cơ sở dữ liệu."""
    connection.execute(
        "INSERT INTO audit_logs(event_type, event_detail, created_at_utc) VALUES (?, ?, ?)",
        (event_type, detail[:500], utc_iso()),
    )


def log_recognition_attempt(
    session_id: int,
    decision: str,
    policy_version: str,
    rejection_reason: str | None = None,
    candidate_student_id: int | None = None,
    distance: float | None = None,
    margin: float | None = None,
    liveness_passed: bool = False,
    stable_duration_ms: int = 0,
) -> None:
    """Lưu telemetry không chứa raw image hoặc unknown embedding."""
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO recognition_attempts(
                    session_id, candidate_student_id, decision, rejection_reason,
                    distance, margin, policy_version, liveness_passed,
                    stable_duration_ms, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    candidate_student_id,
                    decision,
                    rejection_reason,
                    distance,
                    margin,
                    policy_version,
                    int(liveness_passed),
                    int(stable_duration_ms),
                    utc_iso(),
                ),
            )
    except sqlite3.Error:
        # Telemetry không được làm hỏng quyết định nghiệp vụ chính.
        LOGGER.warning("Không thể ghi recognition attempt cho session=%s", session_id)


def get_setting(key: str) -> str | None:
    """Đọc giá trị cấu hình hệ thống từ bảng app_settings."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key = ?", (key,)
        ).fetchone()
    return str(row["setting_value"]) if row else None


def set_setting(key: str, value: str) -> None:
    """Ghi hoặc cập nhật giá trị cấu hình vào bảng app_settings (có ghi Audit Log)."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO app_settings(setting_key, setting_value, updated_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET
                setting_value = excluded.setting_value,
                updated_at_utc = excluded.updated_at_utc
            """,
            (key, value, utc_iso()),
        )
        audit(connection, "setting_updated", key)


def upsert_student(
    student_code: str,
    full_name: str,
    class_name: str,
    consent_given: bool | None = None,
    consent_policy_version: str | None = None,
) -> sqlite3.Row:
    """Thêm mới hoặc cập nhật hồ sơ sinh viên dựa trên mã sinh viên (MSSV).

    Args:
        student_code (str): Mã sinh viên.
        full_name (str): Họ và tên.
        class_name (str): Tên lớp sinh hoạt.

    Returns:
        sqlite3.Row: Dòng dữ liệu sinh viên trong database.
    """
    code = normalize_student_code(student_code)
    name = normalize_person_name(full_name)
    class_value = " ".join(class_name.strip().split())
    if not (1 <= len(class_value) <= 80):
        raise ValueError("Tên lớp phải dài từ 1 đến 80 ký tự.")

    if consent_given is False:
        raise ValueError("Không thể tạo/cập nhật hồ sơ khi chưa có consent rõ ràng.")

    now = utc_iso()
    policy_version = consent_policy_version or DEFAULT_RECOGNITION_POLICY.policy_version
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM students WHERE student_code = ?", (code,)
        ).fetchone()
        if existing is None:
            # Khi caller chưa truyền consent, chỉ tạo hồ sơ pending để tương thích
            # với các tool seed cũ; application enrollment luôn phải truyền True.
            consent_at = now if consent_given is True else ""
            status = "granted" if consent_given is True else "pending"
            connection.execute(
                """
                INSERT INTO students(
                    student_code, full_name, class_name, active,
                    consent_at_utc, consent_policy_version, consent_status,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (code, name, class_value, consent_at, policy_version if consent_given else None,
                 status, now, now),
            )
            if consent_given is True:
                connection.execute(
                    """
                    INSERT INTO biometric_consents(
                        student_id, policy_version, granted_at_utc, status
                    )
                    SELECT id, ?, ?, 'granted'
                    FROM students WHERE student_code = ?
                    """,
                    (policy_version, now, code),
                )
        else:
            connection.execute(
                """
                UPDATE students
                SET full_name = ?, class_name = ?, active = 1, updated_at_utc = ?
                WHERE student_code = ?
                """,
                (name, class_value, now, code),
            )
            if consent_given is True:
                connection.execute(
                    """
                    UPDATE students
                    SET consent_at_utc = ?, consent_policy_version = ?,
                        consent_status = 'granted', active = 1, updated_at_utc = ?
                    WHERE student_code = ?
                    """,
                    (now, policy_version, now, code),
                )
                connection.execute(
                    """
                    UPDATE biometric_consents
                    SET status = 'revoked', revoked_at_utc = ?
                    WHERE student_id = (SELECT id FROM students WHERE student_code = ?)
                      AND status = 'granted'
                    """,
                    (now, code),
                )
                connection.execute(
                    """
                    INSERT INTO biometric_consents(
                        student_id, policy_version, granted_at_utc, status
                    )
                    SELECT id, ?, ?, 'granted'
                    FROM students WHERE student_code = ?
                    """,
                    (policy_version, now, code),
                )
        row = connection.execute(
            "SELECT * FROM students WHERE student_code = ?", (code,)
        ).fetchone()
        audit(
            connection,
            "student_upserted",
            f"{code};consent={'granted' if consent_given is True else 'unchanged_or_pending'}",
        )
        if row is None:
            raise RuntimeError("Không thể tạo hồ sơ sinh viên.")
        return row


def save_embedding(
    student_id: int,
    embedding: np.ndarray,
    image_hash: str,
    blur_score: float,
    brightness: float,
    face_width: int,
    face_height: int,
    policy: RecognitionPolicy | None = None,
    phash: str | None = None,
) -> bool:
    """Lưu trữ vector đặc trưng khuôn mặt (128D BLOB) và thông số chất lượng ảnh.

    Không lưu ảnh gốc để bảo vệ dữ liệu riêng tư (GDPR compliance).

    Returns:
        bool: True nếu thêm mới thành công, False nếu ảnh bị trùng sha256 hash.
    """
    selected_policy = policy or DEFAULT_RECOGNITION_POLICY
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != (selected_policy.embedding_dimension,) or not np.isfinite(vector).all():
        raise ValueError("Embedding phải có 128 giá trị hữu hạn.")
    payload = sqlite3.Binary(vector.tobytes())
    with get_connection() as connection:
        try:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO face_embeddings(
                    student_id, embedding, image_sha256, blur_score,
                    image_phash, brightness, face_width, face_height, embedding_dim,
                    embedding_model, embedding_model_version, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    payload,
                    image_hash,
                    blur_score,
                    phash,
                    brightness,
                    face_width,
                    face_height,
                    selected_policy.embedding_dimension,
                    selected_policy.embedding_model,
                    selected_policy.embedding_model_version,
                    utc_iso(),
                ),
            )
            if cursor.rowcount == 0:
                return False
            audit(connection, "face_embedding_added", f"student_id={student_id}")
            return True
        except sqlite3.IntegrityError as exc:
            raise ValueError("Không thể lưu dữ liệu khuôn mặt cho sinh viên này.") from exc


def student_table() -> pd.DataFrame:
    """Lấy danh sách sinh viên và số ảnh tham chiếu dưới dạng DataFrame."""
    query = """
    SELECT s.id, s.student_code AS 'MSSV', s.full_name AS 'Họ tên',
           s.class_name AS 'Lớp', s.active AS 'Hoạt động',
           COUNT(fe.id) AS 'Số ảnh tham chiếu'
    FROM students s
    LEFT JOIN face_embeddings fe ON fe.student_id = s.id
    GROUP BY s.id
    ORDER BY s.student_code
    """
    with get_connection() as connection:
        frame = pd.read_sql_query(query, connection)
    if not frame.empty:
        frame["Hoạt động"] = frame["Hoạt động"].map({1: "Có", 0: "Không"})
    return frame


def get_student_by_code(student_code: str) -> sqlite3.Row | None:
    """Lấy hồ sơ sinh viên theo mã để kiểm tra profile biometric hiện có."""
    code = normalize_student_code(student_code)
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM students WHERE student_code = ?", (code,)
        ).fetchone()


def get_student_embeddings(
    student_id: int,
    policy: RecognitionPolicy | None = None,
) -> list[np.ndarray]:
    """Đọc các embedding còn hiệu lực của một sinh viên theo model version."""
    selected_policy = policy or DEFAULT_RECOGNITION_POLICY
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT embedding, embedding_dim
            FROM face_embeddings
            WHERE student_id = ?
              AND revoked_at_utc IS NULL
              AND embedding_dim = ?
              AND embedding_model = ?
              AND embedding_model_version = ?
            ORDER BY id
            """,
            (
                student_id,
                selected_policy.embedding_dimension,
                selected_policy.embedding_model,
                selected_policy.embedding_model_version,
            ),
        ).fetchall()
    vectors: list[np.ndarray] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float64).copy()
        if vector.shape == (selected_policy.embedding_dimension,) and np.isfinite(vector).all():
            vectors.append(vector)
    return vectors


def remove_student_biometrics(student_id: int, audit_event: str = "student_biometrics_removed") -> None:
    """Xóa vector khuôn mặt và vô hiệu hóa sinh viên.

    Args:
        student_id (int): ID của sinh viên cần thu hồi dữ liệu.
        audit_event (str): Tên sự kiện ghi vào audit log.
    """
    with get_connection() as connection:
        connection.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        connection.execute(
            """
            UPDATE biometric_consents
            SET status = 'revoked', revoked_at_utc = ?
            WHERE student_id = ? AND status = 'granted'
            """,
            (utc_iso(), student_id),
        )
        connection.execute(
            """
            UPDATE students
            SET active = 0, consent_status = 'revoked', updated_at_utc = ?
            WHERE id = ?
            """,
            (utc_iso(), student_id),
        )
        audit(connection, audit_event, f"student_id={student_id}")


def revoke_student_consent(student_id: int) -> None:
    """Rút lại quyền sử dụng dữ liệu sinh trắc học của sinh viên.

    Hành động:
    1. Xóa toàn bộ vector đặc trưng khuôn mặt khỏi face_embeddings.
    2. Chuyển sinh viên sang trạng thái active = 0.
    3. Ghi audit log 'consent_revoked'.
    4. Giữ nguyên lịch sử các bản ghi điểm danh quá khứ (không xóa bảng attendance).

    Args:
        student_id (int): ID sinh viên.
    """
    now = utc_iso()
    with get_connection() as connection:
        connection.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        connection.execute(
            """
            UPDATE biometric_consents
            SET status = 'revoked', revoked_at_utc = ?
            WHERE student_id = ? AND status = 'granted'
            """,
            (now, student_id),
        )
        connection.execute(
            """
            UPDATE students
            SET active = 0, consent_status = 'revoked', updated_at_utc = ?
            WHERE id = ?
            """,
            (now, student_id),
        )
        audit(connection, "consent_revoked", f"student_id={student_id}")


def purge_expired_biometrics(retention_days: int = BIOMETRIC_RETENTION_DAYS) -> int:
    """Xóa vector khuôn mặt đã hết thời hạn lưu trữ.

    Sinh viên không còn vector tham chiếu sẽ được chuyển sang trạng thái không hoạt động.

    Args:
        retention_days: Số ngày lưu tối đa, mặc định đọc từ cấu hình.

    Returns:
        int: Số lượng bản ghi embedding đã bị xóa.
    """
    if not 1 <= retention_days <= 3650:
        raise ValueError("Thời hạn lưu trữ phải nằm trong khoảng 1-3650 ngày.")
    cutoff = utc_iso(utc_now() - timedelta(days=retention_days))
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM face_embeddings WHERE created_at_utc < ?", (cutoff,)
        )
        deleted = max(0, int(cursor.rowcount))
        connection.execute(
            """
            UPDATE students
            SET active = 0, updated_at_utc = ?
            WHERE active = 1
              AND NOT EXISTS (
                  SELECT 1 FROM face_embeddings fe WHERE fe.student_id = students.id
              )
            """,
            (utc_iso(),),
        )
        audit(connection, "expired_biometrics_purged", f"deleted={deleted}")
        return deleted


def create_course(course_code: str, course_name: str, lecturer: str) -> None:
    """Tạo thông tin môn học mới.

    Args:
        course_code (str): Mã môn học (ví dụ: 'CS101').
        course_name (str): Tên môn học.
        lecturer (str): Tên giảng viên phụ trách.
    """
    code = normalize_course_code(course_code)
    name = " ".join(course_name.strip().split())
    teacher = normalize_person_name(lecturer)
    if not (2 <= len(name) <= 120):
        raise ValueError("Tên môn học phải dài từ 2 đến 120 ký tự.")
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO courses(course_code, course_name, lecturer, created_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            (code, name, teacher, utc_iso()),
        )
        audit(connection, "course_created", code)


def list_courses() -> list[sqlite3.Row]:
    """Lấy danh sách tất cả các môn học sắp xếp theo mã môn."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM courses ORDER BY course_code"
        ).fetchall()


def get_course_roster(course_id: int) -> set[int]:
    """Lấy tập hợp các student_id thuộc danh sách môn học."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_id FROM course_enrollments WHERE course_id = ?",
            (course_id,),
        ).fetchall()
    return {int(row["student_id"]) for row in rows}


def set_course_roster(course_id: int, student_ids: Iterable[int]) -> int:
    """Cập nhật danh sách sinh viên đăng ký học môn học.

    Args:
        course_id (int): ID môn học.
        student_ids (Iterable[int]): Danh sách ID sinh viên tham gia môn học.

    Returns:
        int: Số lượng sinh viên trong danh sách sau khi cập nhật.
    """
    normalized_ids = sorted({int(student_id) for student_id in student_ids})
    with get_connection() as connection:
        course = connection.execute(
            "SELECT id FROM courses WHERE id = ?", (course_id,)
        ).fetchone()
        if course is None:
            raise ValueError("Không tìm thấy môn học.")
        if normalized_ids:
            placeholders = ",".join("?" for _ in normalized_ids)
            rows = connection.execute(
                f"SELECT id FROM students WHERE active = 1 AND id IN ({placeholders})",
                normalized_ids,
            ).fetchall()
            valid_ids = {int(row["id"]) for row in rows}
            if valid_ids != set(normalized_ids):
                raise ValueError("Danh sách có sinh viên không tồn tại hoặc đã bị vô hiệu hóa.")

        connection.execute(
            "DELETE FROM course_enrollments WHERE course_id = ?", (course_id,)
        )
        now = utc_iso()
        connection.executemany(
            """
            INSERT INTO course_enrollments(course_id, student_id, enrolled_at_utc)
            VALUES (?, ?, ?)
            """,
            [(course_id, student_id, now) for student_id in normalized_ids],
        )
        audit(
            connection,
            "course_roster_updated",
            f"course={course_id},students={len(normalized_ids)}",
        )
        return len(normalized_ids)


def create_attendance_session(
    course_id: int,
    session_name: str,
    start_local: datetime,
    end_local: datetime,
    late_after_minutes: int,
) -> None:
    """Tạo một buổi học điểm danh mới và snapshot danh sách sinh viên môn học vào buổi học.

    Args:
        course_id (int): ID môn học.
        session_name (str): Tên buổi học (ví dụ: 'Buổi 01 - Tổng quan').
        start_local (datetime): Thời gian bắt đầu buổi học (giờ địa phương).
        end_local (datetime): Thời gian kết thúc buổi học (giờ địa phương).
        late_after_minutes (int): Số phút cho phép sau thời gian bắt đầu trước khi tính là đi trễ.
    """
    name = " ".join(session_name.strip().split())
    if not 1 <= len(name) <= 120:
        raise ValueError("Tên buổi học phải dài từ 1 đến 120 ký tự.")
    if end_local <= start_local:
        raise ValueError("Thời gian kết thúc phải sau thời gian bắt đầu.")
    if not 0 <= late_after_minutes <= 180:
        raise ValueError("Số phút tính đi trễ phải nằm trong khoảng 0-180.")
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO attendance_sessions(
                course_id, session_name, start_at_utc, end_at_utc,
                late_after_minutes, status, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, 'scheduled', ?)
            """,
            (
                course_id,
                name,
                utc_iso(start_local),
                utc_iso(end_local),
                late_after_minutes,
                utc_iso(),
            ),
        )
        session_id = int(cursor.lastrowid)
        # Snapshot danh sách sinh viên hiện tại của môn học vào buổi học này
        connection.execute(
            """
            INSERT INTO session_enrollments(session_id, student_id, enrolled_at_utc)
            SELECT ?, student_id, ?
            FROM course_enrollments
            WHERE course_id = ?
            """,
            (session_id, utc_iso(), course_id),
        )
        audit(connection, "session_created", name)


def list_sessions(status: str | None = None) -> list[sqlite3.Row]:
    """Lấy danh sách các buổi học (tùy chọn lọc theo trạng thái 'scheduled', 'open', 'closed')."""
    query = """
    SELECT s.*, c.course_code, c.course_name, c.lecturer
    FROM attendance_sessions s
    JOIN courses c ON c.id = s.course_id
    """
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE s.status = ?"
        params = (status,)
    query += " ORDER BY s.start_at_utc DESC"
    with get_connection() as connection:
        return connection.execute(query, params).fetchall()


def change_session_status(session_id: int, new_status: str) -> None:
    """Chuyển đổi trạng thái buổi học (Scheduled -> Open -> Closed)."""
    if new_status not in {"scheduled", "open", "closed"}:
        raise ValueError("Trạng thái buổi học không hợp lệ.")
    with get_connection() as connection:
        current = connection.execute(
            "SELECT status FROM attendance_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if current is None:
            raise ValueError("Không tìm thấy buổi học.")
        current_status = str(current["status"])
        if current_status == new_status:
            return
        allowed_transitions = {"scheduled": "open", "open": "closed"}
        if allowed_transitions.get(current_status) != new_status:
            raise ValueError(
                f"Không thể chuyển trạng thái từ {current_status} sang {new_status}."
            )
        connection.execute(
            "UPDATE attendance_sessions SET status = ? WHERE id = ?",
            (new_status, session_id),
        )
        audit(connection, "session_status_changed", f"{session_id}:{new_status}")


def attendance_report(session_id: int) -> pd.DataFrame:
    """Tạo báo cáo điểm danh chi tiết theo buổi học dưới dạng pandas DataFrame.

    Args:
        session_id (int): ID của buổi học cần trích xuất báo cáo.

    Returns:
        pd.DataFrame: Bảng thông tin chứa MSSV, Họ tên, Lớp, Mã môn, Buổi học,
                     Trạng thái (Có mặt/Đi trễ/Vắng), Thời gian điểm danh,
                     Khoảng cách khuôn mặt và Ngưỡng chấp nhận.
    """
    query = """
    SELECT st.student_code AS 'MSSV', st.full_name AS 'Họ tên',
           st.class_name AS 'Lớp', c.course_code AS 'Mã môn',
           ses.session_name AS 'Buổi học',
           COALESCE(a.attendance_status, 'absent') AS 'Trạng thái',
           a.check_in_at_utc AS 'Thời gian UTC',
           a.recognition_distance AS 'Khoảng cách khuôn mặt',
           a.threshold_used AS 'Ngưỡng'
    FROM attendance_sessions ses
    JOIN courses c ON c.id = ses.course_id
    JOIN session_enrollments se ON se.session_id = ses.id
    JOIN students st ON st.id = se.student_id
    LEFT JOIN attendance a
        ON a.session_id = ses.id AND a.student_id = st.id
    WHERE ses.id = ?
    ORDER BY a.check_in_at_utc IS NULL, a.check_in_at_utc, st.student_code
    """
    with get_connection() as connection:
        frame = pd.read_sql_query(query, connection, params=(session_id,))
    if not frame.empty:
        frame["Thời gian điểm danh"] = frame["Thời gian UTC"].map(display_datetime)
        frame.drop(columns=["Thời gian UTC"], inplace=True)
        frame["Trạng thái"] = frame["Trạng thái"].map(
            {"present": "Có mặt", "late": "Đi trễ", "absent": "Vắng"}
        )
        # Ép kiểu trước khi làm tròn vì sinh viên vắng có giá trị None/NaN.
        frame["Khoảng cách khuôn mặt"] = pd.to_numeric(
            frame["Khoảng cách khuôn mặt"], errors="coerce"
        ).round(4)
    return frame


def mark_attendance(
    session_id: int,
    student_id: int,
    distance: float,
    identity_margin: float = 0.0,
    margin_threshold: float = DEFAULT_RECOGNITION_POLICY.identity_margin,
    liveness_policy: str = DEFAULT_RECOGNITION_POLICY.liveness_policy,
    policy_version: str = DEFAULT_RECOGNITION_POLICY.policy_version,
    confirmation_frames: int = DEFAULT_RECOGNITION_POLICY.minimum_observations,
    stable_duration_ms: int = 0,
    aggregation_strategy: str = DEFAULT_RECOGNITION_POLICY.aggregation_strategy,
    embedding_model_version: str = DEFAULT_RECOGNITION_POLICY.embedding_model_version,
    recognition_policy_hash: str | None = None,
    source: str = "face_webrtc",
    tolerance: float | None = None,
) -> tuple[str, str]:
    """Ghi nhận điểm danh cho sinh viên trong buổi học bằng ACID Transaction cách ly cao.

    Dùng BEGIN IMMEDIATE để chống ghi trùng khi nhiều client chạy song song.
    Phân loại tự động trạng thái 'Có mặt' (present) hoặc 'Đi trễ' (late) dựa trên cấu hình buổi học.
    Lưu trữ đầy đủ các bằng chứng nhận diện (distance, margin, liveness, policy version) phục vụ audit.

    Args:
        session_id (int): ID buổi học.
        student_id (int): ID sinh viên.
        distance (float): Khoảng cách khuôn mặt so với mẫu tham chiếu.
        identity_margin: Chênh lệch margin giữa Top-1 và Top-2.
        margin_threshold: Ngưỡng margin yêu cầu.
        liveness_policy: Tên thuật toán liveness được áp dụng.
        policy_version: Phiên bản chính sách nhận diện.
        confirmation_frames: Số khung hình nhận diện liên tiếp hợp lệ.
        source: Nguồn gốc điểm danh ('face_webrtc', 'manual', etc.).
        tolerance: Ngưỡng khoảng cách tùy biến (None lấy mặc định config).

    Returns:
        tuple[str, str]: (mã_kết_quả, thông_báo_chi_tiết).
    """
    actual_tolerance = (
        tolerance
        if tolerance is not None
        else DEFAULT_RECOGNITION_POLICY.distance_threshold
    )
    if not np.isfinite(distance) or not 0 <= distance <= actual_tolerance:
        return "rejected", "Kết quả nhận diện không đạt ngưỡng cho phép."

    now = utc_now()
    connection = get_connection()
    try:
        # Bắt đầu transaction ngay lập tức với khóa ghi (reserved lock)
        connection.execute("BEGIN IMMEDIATE")
        session = connection.execute(
            "SELECT * FROM attendance_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        student = connection.execute(
            """
            SELECT st.*
            FROM students st
            JOIN session_enrollments se ON se.student_id = st.id
            WHERE st.id = ? AND st.active = 1 AND se.session_id = ?
            """,
            (student_id, session_id),
        ).fetchone()
        if session is None or session["status"] != "open":
            connection.rollback()
            return "closed", "Buổi học chưa mở hoặc đã đóng."
        if student is None:
            connection.rollback()
            return "inactive", "Sinh viên không hoạt động hoặc không thuộc môn học này."

        start_at = datetime.fromisoformat(session["start_at_utc"])
        end_at = datetime.fromisoformat(session["end_at_utc"])
        if now < start_at:
            connection.rollback()
            return "outside", "Chưa đến thời gian của buổi học."
        if now > end_at:
            connection.execute(
                "UPDATE attendance_sessions SET status = 'closed' WHERE id = ?",
                (session_id,),
            )
            audit(connection, "session_auto_closed", str(session_id))
            connection.commit()
            return "outside", "Buổi học đã hết thời gian và được đóng tự động."

        existing = connection.execute(
            "SELECT attendance_status, check_in_at_utc FROM attendance "
            "WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if existing:
            connection.rollback()
            return (
                "already",
                f"{student['student_code']} đã điểm danh lúc "
                f"{display_datetime(existing['check_in_at_utc'])}.",
            )

        late_cutoff = start_at + timedelta(minutes=int(session["late_after_minutes"]))
        attendance_status = "late" if now > late_cutoff else "present"

        connection.execute(
            """
            INSERT INTO attendance(
                session_id, student_id, check_in_at_utc, attendance_status,
                recognition_distance, threshold_used, source,
                identity_margin, margin_threshold, liveness_policy,
                recognition_policy_version, confirmation_frames
                , stable_duration_ms, aggregation_strategy,
                embedding_model_version, recognition_policy_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                student_id,
                utc_iso(now),
                attendance_status,
                float(distance),
                float(actual_tolerance),
                source,
                float(identity_margin),
                float(margin_threshold),
                str(liveness_policy),
                str(policy_version),
                int(confirmation_frames),
                int(stable_duration_ms),
                str(aggregation_strategy),
                str(embedding_model_version),
                recognition_policy_hash,
            ),
        )
        audit(
            connection,
            "attendance_created",
            f"session={session_id},student={student_id},status={attendance_status},dist={distance:.4f},margin={identity_margin:.4f}",
        )
        connection.commit()
        label = "ĐI TRỄ" if attendance_status == "late" else "CÓ MẶT"
        return "created", f"{student['student_code']} - {label}"
    except sqlite3.IntegrityError:
        connection.rollback()
        return "already", "Sinh viên đã được điểm danh trong buổi này."
    except sqlite3.Error:
        connection.rollback()
        LOGGER.exception("Không thể ghi nhận điểm danh")
        return "error", "Không thể ghi nhận điểm danh do lỗi cơ sở dữ liệu."
    finally:
        connection.close()


def manual_attendance_correction(
    session_id: int,
    student_id: int,
    new_status: str,
    lecturer_id: str,
    reason: str,
) -> tuple[bool, str]:
    """Giảng viên can thiệp sửa hoặc ghi nhận điểm danh thủ công với lưu vết kiểm toán (First-Class Audit Trail).

    Args:
        session_id: ID buổi học.
        student_id: ID sinh viên.
        new_status: 'present', 'late', hoặc 'absent'.
        lecturer_id: Định danh giảng viên thực hiện sửa.
        reason: Lý do điều chỉnh thủ công.

    Returns:
        tuple[bool, str]: (Thành công hay không, Thông báo).
    """
    if new_status not in {"present", "late", "absent"}:
        return False, "Trạng thái mới không hợp lệ ('present', 'late', 'absent')."
    if not reason.strip():
        return False, "Cần cung cấp lý do điều chỉnh điểm danh."

    now = utc_now()
    now_iso = utc_iso(now)
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        student = connection.execute(
            "SELECT student_code, full_name FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        if student is None:
            connection.rollback()
            return False, "Không tìm thấy sinh viên."

        existing = connection.execute(
            "SELECT id, attendance_status FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()

        if existing:
            orig_status = existing["attendance_status"]
            connection.execute(
                """
                UPDATE attendance SET
                    original_status = COALESCE(original_status, ?),
                    final_status = ?,
                    attendance_status = ?,
                    corrected_by = ?,
                    correction_reason = ?,
                    corrected_at_utc = ?,
                    source = 'manual_correction'
                WHERE id = ?
                """,
                (
                    orig_status,
                    new_status,
                    new_status,
                    lecturer_id,
                    reason.strip(),
                    now_iso,
                    existing["id"],
                ),
            )
        else:
            # Chưa từng có bản ghi -> ghi nhận thủ công trực tiếp
            connection.execute(
                """
                INSERT INTO attendance(
                    session_id, student_id, check_in_at_utc, attendance_status,
                    recognition_distance, threshold_used, source,
                    original_status, final_status, corrected_by,
                    correction_reason, corrected_at_utc
                ) VALUES (?, ?, ?, ?, 0.0, 0.50, 'manual', 'none', ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    now_iso,
                    new_status,
                    new_status,
                    lecturer_id,
                    reason.strip(),
                    now_iso,
                ),
            )

        audit(
            connection,
            "attendance_manual_corrected",
            f"session={session_id},student={student_id},status={new_status},by={lecturer_id},reason={reason.strip()}",
        )
        connection.commit()
        return True, f"Đã cập nhật trạng thái {student['student_code']} thành '{new_status}'."
    except sqlite3.Error:
        connection.rollback()
        LOGGER.exception("Lỗi khi sửa điểm danh thủ công")
        return False, "Lỗi cơ sở dữ liệu khi sửa điểm danh."
    finally:
        connection.close()
