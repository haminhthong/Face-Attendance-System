# 🎓 Face Attendance System

[![CI](https://github.com/haminhthong/Face-Attendance-System/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Face-Attendance-System/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)

Hệ thống điểm danh sinh viên bằng khuôn mặt, gồm giao diện Streamlit/WebRTC, REST API FastAPI và cơ sở dữ liệu SQLite. Mã nguồn được tổ chức quanh một pipeline duy nhất: từ consent và quality gate, tạo embedding, open-set matching trong roster, liveness, temporal confirmation, kiểm tra policy ở application service, đến transaction điểm danh và báo cáo audit.

Đây là prototype kỹ thuật có kiểm soát, không phải hệ thống PAD chống deepfake/replay và chưa đưa ra claim accuracy biometric production. Embedding vẫn là dữ liệu sinh trắc học nhạy cảm dù ảnh gốc không được lưu.

## 1. Bài toán và phạm vi ứng dụng

### Bài toán

Trong một buổi học, hệ thống cần xác định người đang đứng trước camera có thuộc danh sách sinh viên của buổi đó hay không, chỉ ghi nhận một lần, phân biệt có mặt/đi trễ theo thời gian buổi học và bảo đảm quyết định có thể truy vết. Khi nhận diện không đủ bằng chứng, hệ thống phải từ chối thay vì cố đoán.

### Phạm vi hiện tại

- Quản trị sinh viên, môn học, roster môn học và snapshot roster theo từng buổi học.
- Đăng ký tối thiểu 5 ảnh hợp lệ sau khi có consent rõ ràng.
- Kiểm tra đúng một khuôn mặt, kích thước, độ mờ, độ sáng, hash byte và pHash gần trùng.
- Lưu embedding 128 chiều cùng metadata model/version, chất lượng và hash ảnh; không lưu ảnh gốc.
- Điểm danh realtime qua WebRTC với open-set matching, Top-2 Mean, distance threshold, identity margin, liveness chớp mắt và temporal confirmation.
- Điểm danh/sửa thủ công có giảng viên, lý do và audit trail; không giả mạo liveness AI.
- Báo cáo theo snapshot roster, trong đó sinh viên chưa có bản ghi được hiển thị là vắng.
- API có API key, policy metadata, báo cáo và hai luồng ghi nhận biometric/manual.

### Ngoài phạm vi

- Không tự động hiệu chuẩn threshold từ dữ liệu synthetic.
- Không tuyên bố FAR/FRR/TAR hoặc khả năng chống giả mạo khi chưa có private validation/test set hợp lệ.
- Liveness EAR blink chỉ là heuristic người thật cơ bản, không thay thế PAD chuyên dụng.
- API key là cơ chế xác thực tích hợp nội bộ; chưa có OAuth/RBAC/attestation cho client camera.

## 2. Luồng logic và data flow duy nhất

RecognitionPolicy trong src/face_attendance/policy.py là nguồn sự thật cho model, dimension, metric, Top-2 Mean, threshold, margin, temporal và liveness. AttendanceService không tin threshold tùy ý từ client; nó kiểm tra policy evidence trước khi gọi transaction database. Consent policy version là metadata pháp lý riêng, không được so sánh với recognition policy version.

```mermaid
flowchart TD
    A[Admin mở ứng dụng] --> B[Thiết lập PIN quản trị]
    B --> C[Nhập sinh viên + xác nhận consent]
    C --> D{Có consent rõ ràng?}
    D -- Không --> D1[Từ chối trước khi giải mã ảnh]
    D -- Có --> E[Nhận tối thiểu 5 ảnh]
    E --> F[Quality gate: file, OpenCV, đúng 1 mặt, kích thước, blur, brightness]
    F --> G[SHA-256 + pHash near-duplicate]
    G --> H{Còn ít nhất 5 ảnh duy nhất?}
    H -- Không --> H1[Yêu cầu bổ sung ảnh]
    H -- Có --> I[Face encoding 128D]
    I --> J[Identity consistency trong batch và với profile cũ]
    J --> K[SQLite: students + biometric_consents + face_embeddings]
    K --> L[Tạo môn học và course roster]
    L --> M[Tạo buổi học]
    M --> N[Snapshot course roster vào session_enrollments]

    N --> O[WebRTC frame]
    O --> P[Skip frame + resize 0.25x + HOG face detection]
    P --> Q{Đúng 1 khuôn mặt?}
    Q -- 0 --> Q0[Chờ frame tiếp theo]
    Q -- >1 --> Q1[Single-Face Gate: từ chối frame]
    Q -- 1 --> R[Encoding + landmarks EAR]
    R --> S[Nạp template active, granted, còn hiệu lực, đúng model/version, thuộc session]
    S --> T[Matcher theo student: Top-2 Mean]
    T --> U{Distance <= threshold và margin >= threshold?}
    U -- Không --> U1[Unknown hoặc ambiguous; không tạo attendance]
    U -- Có --> V[Liveness ear_blink_v1 bắt buộc]
    V --> W{Đủ blink + >= 3 quan sát + >= 500ms?}
    W -- Không --> W1[Không cộng temporal evidence]
    W -- Có --> X[Tạo RecognitionDecision đầy đủ evidence]
    X --> Y[AttendanceService kiểm tra policy hash/version/model/temporal]
    Y --> Z[Kiểm tra consent hiện tại]
    Z -- Không --> Z1[Rejected: no_consent + recognition_attempt]
    Z -- Có --> AA[BEGIN IMMEDIATE transaction]
    AA --> AB[Kiểm tra session open, time window, roster snapshot, duplicate]
    AB -- Không --> AB1[Rejected/closed/outside/already + audit]
    AB -- Có --> AC[Ghi attendance present/late + evidence]
    AC --> AD[Ghi audit_logs + recognition_attempts]
    AD --> AE[Báo cáo SQLite/Pandas]

    N --> AF[Giảng viên mở manual correction]
    AF --> AG[Kiểm tra student thuộc session snapshot]
    AG --> AH[Ghi/sửa status + lecturer + reason + audit]
    AH --> AE

    AI[API client] --> AJ[X-API-Key]
    AJ --> AK[/policy, /sessions, /attendance/biometric, /attendance/manual]
    AK --> Y
```

### Các điểm chặn chính

1. Consent được kiểm tra trước khi giải mã hoặc xử lý ảnh.
2. Enrollment chỉ tạo/cập nhật profile sau khi batch ảnh vượt quality gate và consistency gate.
3. Runtime chỉ nạp template có active = 1, consent_status = granted, embedding chưa bị revoke, đúng dimension/model/version và thuộc session_enrollments.
4. Single-Face Gate dừng frame có 0 hoặc nhiều hơn 1 khuôn mặt.
5. Open-set matcher từ chối unknown và match mơ hồ; Top-2 Mean là chiến lược runtime duy nhất.
6. Blink, số quan sát liên tiếp và thời gian ổn định là ba điều kiện evidence của realtime pipeline.
7. Application service kiểm tra policy evidence và consent trước database transaction.
8. Database kiểm tra buổi học, thời gian, roster snapshot và unique attendance trong BEGIN IMMEDIATE.
9. Báo cáo lấy snapshot roster làm mẫu số nên sinh viên không có attendance row vẫn xuất hiện là Vắng.

## 3. Recognition policy đang dùng

Policy mặc định nằm trong DEFAULT_RECOGNITION_POLICY; có thể nạp file JSON qua RECOGNITION_POLICY_PATH. Artifact mẫu: configs/recognition_policy.example.json.

| Thuộc tính | Giá trị mặc định |
|---|---|
| Embedding model | dlib_face_recognition_resnet_v1 version 1 |
| Dimension/metric | 128 / euclidean_l2 |
| Aggregation | top_k_mean, top_k = 2 |
| Distance threshold | 0.50 |
| Identity margin | 0.05 |
| Temporal | tối thiểu 3 quan sát và 500 ms |
| Liveness | ear_blink_v1, TTL 10 giây |
| Policy hash | SHA-256 trên JSON policy chuẩn hóa |
| Calibrated | false ở policy mẫu/mặc định |

stable_duration_ms trong evidence là giá trị quan sát được và phải lớn hơn hoặc bằng mức policy yêu cầu; không so bằng tuyệt đối. Client biometric phải gửi đủ evidence. Server trả 409 nếu version, hash, threshold, model, aggregation, liveness hoặc temporal policy không khớp.

Nếu một session chỉ có một identity, matcher không có ứng viên Top-2 thật; nó dùng sentinel hữu hạn đúng bằng margin policy để evidence vẫn serialize được và không coi đó là một khoảng cách đo được.

Production chỉ khởi động khi APP_ENV=production, API key không phải giá trị mặc định và policy được đánh dấu calibrated=true. Giá trị calibrated phải là JSON boolean thật, không phải chuỗi "false".

## 4. Dữ liệu, consent và retention

SQLite tạo các nhóm bảng:

- students: hồ sơ học vụ, active và trạng thái consent. active là trạng thái học vụ, không phải consent.
- biometric_consents: lịch sử grant/revoke theo phiên bản văn bản consent.
- face_embeddings: vector và metadata chất lượng/model/hash ảnh, không có raw image.
- courses, course_enrollments, attendance_sessions, session_enrollments: môn học, roster và snapshot buổi học.
- attendance: trạng thái cuối, bằng chứng nhận diện hoặc marker manual.
- recognition_attempts: telemetry metadata của quyết định, không lưu raw image/unknown embedding.
- audit_logs, app_settings: audit nghiệp vụ và cấu hình PIN/setting.

Khi thu hồi consent, hệ thống xóa toàn bộ embedding, revoke consent và giữ hồ sơ học vụ, roster snapshot, báo cáo và lịch sử điểm danh. Khi retention job xóa embedding quá hạn, sinh viên chuyển về consent_status='pending' để đăng ký lại; active vẫn giữ nguyên. API chạy purge_expired_biometrics() lúc startup.

Dữ liệu thật phải nằm ngoài Git:

```text
data/private/manifest.json
data/private/enrollment/...
data/private/validation/known/...
data/private/validation/unknown/...
data/private/test/known/...
data/private/test/unknown/...
```

Manifest mẫu data/manifest.example.json chỉ mô tả schema. Data card và quy ước lưu trữ nằm ở data/README.md.

## 5. Cấu trúc thư mục

```text
face-attendance-system/
├── app.py                              # Entry point Streamlit
├── src/face_attendance/
│   ├── domain/                         # Entity, enum, exception nghiệp vụ
│   ├── application/
│   │   ├── attendance_service.py       # Biometric/manual application flow
│   │   └── enrollment_service.py       # Consent gate cho enrollment
│   ├── api.py                          # FastAPI health, policy, session, attendance
│   ├── config.py                       # .env, path, giới hạn và policy runtime
│   ├── database.py                     # Schema, migration, transaction, report, audit
│   ├── liveness.py                     # EAR blink state machine
│   ├── matcher.py                      # Open-set matcher và Top-2 Mean
│   ├── policy.py                       # Schema, hash và validation policy
│   ├── recognition.py                  # Quality gate, enrollment, WebRTC engine
│   ├── ui.py                           # Dashboard Streamlit/admin
│   └── utils.py                        # Chuẩn hóa dữ liệu, UTC, PIN
├── configs/
│   └── recognition_policy.example.json # Policy chưa calibrated
├── data/
│   ├── README.md                       # Data card và quy ước dữ liệu
│   ├── manifest.example.json           # Manifest private minh họa
│   ├── private/                        # Dữ liệu thật, tool tạo khi cần và bị ignore
│   └── results/                        # Kết quả cục bộ, tool tạo khi chạy evaluation
├── tools/
│   ├── evaluate_biometrics.py          # Private manifest/leakage gate
│   ├── evaluate_matching.py            # Matcher sanity/evaluation entry point
│   ├── matcher_sanity_check.py         # Synthetic matcher sanity check
│   └── prepare_dataset.py              # Chuẩn bị và kiểm tra dataset
├── tests/                               # Unit/integration tests
├── .env.example                         # Cấu hình môi trường mẫu
├── .github/workflows/ci.yml             # CI lint, format, compile, test
├── Dockerfile
├── LICENSE
└── pyproject.toml                       # Dependency và tool configuration
```

Dependency duy nhất được quản lý trong pyproject.toml; requirements.txt dư thừa đã được loại bỏ để tránh hai nguồn version khác nhau. Cache Python, test, IDE, runtime DB và dữ liệu private bị ignore.

## 6. API

Tất cả endpoint trừ /health cần header X-API-Key. API key được so sánh bằng secrets.compare_digest.

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | /health | Health check công khai |
| GET | /policy | Metadata policy đang chạy |
| GET | /sessions | Danh sách buổi học |
| GET | /sessions/{session_id}/attendance | Báo cáo attendance theo snapshot |
| POST | /attendance/biometric | Nhận RecognitionDecision đầy đủ evidence |
| POST | /attendance/manual | Giảng viên ghi/sửa status, bắt buộc reason |

Route /attendance legacy đã được loại bỏ để không tạo đường tắt bỏ qua liveness, temporal evidence và policy gate. Biometric chỉ đi qua /attendance/biometric; manual là luồng human oversight riêng.

Ví dụ payload biometric phải lấy metadata từ GET /policy và gửi đầy đủ:

```json
{
  "session_id": 12,
  "student_id": 34,
  "student_code": "SV001",
  "full_name": "Nguyen Van A",
  "distance": 0.32,
  "second_distance": 0.55,
  "margin": 0.23,
  "liveness_passed": true,
  "confirmation_frames": 3,
  "policy_version": "face-policy-v1",
  "distance_threshold": 0.5,
  "margin_threshold": 0.05,
  "aggregation_strategy": "top_k_mean",
  "embedding_model": "dlib_face_recognition_resnet_v1",
  "embedding_model_version": "1",
  "stable_duration_ms": 500,
  "liveness_policy": "ear_blink_v1",
  "recognition_policy_hash": "<sha256-64-characters>"
}
```

Manual request gồm session_id, student_id, status (present|late|absent), lecturer_id và reason. Database kiểm tra sinh viên thuộc snapshot của session trước khi ghi.

## 7. Cài đặt và chạy

### Windows PowerShell

```powershell
git clone https://github.com/haminhthong/Face-Attendance-System.git
cd Face-Attendance-System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Nếu PowerShell chặn script activate, có thể chạy trực tiếp .\.venv\Scripts\python.exe -m pip install -e ".[dev]".

### Linux/CI

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

face-recognition/dlib và OpenCV cần native build/runtime. Linux CI cài build-essential, cmake, libopenblas-dev, libgl1, libglib2.0-0; Dockerfile cài cùng nhóm package.

### Cấu hình .env

File .env được nạp tự động nếu python-dotenv có trong môi trường. Tối thiểu khi chạy local:

```dotenv
APP_ENV=development
FACE_ATTENDANCE_API_KEY=local-dev-secret
DATABASE_PATH=face_attendance_data/face_attendance.db
RECOGNITION_POLICY_PATH=
BIOMETRIC_RETENTION_DAYS=365
MAX_UPLOAD_SIZE_MB=8
PROCESS_EVERY_N_FRAMES=3
LOG_LEVEL=INFO
```

DATABASE_PATH được ưu tiên hơn FACE_ATTENDANCE_DATA_DIR. Không commit .env, API key, SQLite DB, ảnh, embedding thật hoặc manifest private.

### Chạy dashboard và API

```bash
streamlit run app.py
uvicorn face_attendance.api:app --reload --port 8000
```

Dashboard ở http://localhost:8501; Swagger UI ở http://127.0.0.1:8000/docs. Lần chạy đầu tiên, khu vực quản trị yêu cầu tạo PIN 6–12 chữ số.

### Chạy Docker

```bash
docker build -t face-attendance-system .
docker run --rm -p 8501:8501 \
  -e APP_ENV=development \
  -e FACE_ATTENDANCE_API_KEY=local-dev-secret \
  -v face_attendance_data:/app/face_attendance_data \
  face-attendance-system
```

## 8. Kiểm thử, CI và đánh giá

`httpx` được khai báo trong dependency chính vì các bài kiểm thử API dùng `FastAPI TestClient`; do đó cả `pip install -e .` và `pip install -e ".[dev]"` đều có đủ dependency này.

Các lệnh CI chính:

```bash
python -m compileall -q app.py src tests
ruff check app.py src tools tests
ruff format --check app.py src tools tests
pytest -q -p no:cacheprovider
```

GitHub Actions dùng Python 3.11, cài native dependencies trước khi cài .[dev], đặt permissions: contents: read, timeout 20 phút và hủy run cũ cùng branch. Test ảnh/WebRTC được chạy trong CI đầy đủ dependency; local thiếu OpenCV sẽ không đại diện cho CI.

Chạy local:

```bash
python -m ruff check app.py src tools tests
python -m ruff format --check app.py src tools tests
python -m pytest -q -p no:cacheprovider
```

Các tool evaluation tách khỏi claim production:

```bash
python tools/prepare_dataset.py
python tools/matcher_sanity_check.py
python tools/evaluate_matching.py
python tools/evaluate_biometrics.py --manifest data/private/manifest.json
```

Matcher sanity dùng vector synthetic để kiểm tra toán Top-K/centroid/min-distance, margin, reject và latency; kết quả không deployable. evaluate_biometrics.py không fallback sang synthetic, chỉ gate manifest/split/leakage và không tự chọn threshold. Validation dùng để chọn policy; test chỉ chạy sau khi policy khóa.

## 9. Repo cleanliness và giới hạn triển khai

- Chỉ còn hai file Markdown có mục đích: README này và data/README.md.
- Không chứa file hướng dẫn cũ, report build hoặc requirements trùng với pyproject.
- data/private và data/results không chứa artifact runtime trong repo; tool tự tạo khi cần.
- CI kiểm tra format để ngăn whitespace và import drift quay lại.
- Audit log không thay thế phân quyền; cần đặt SQLite trên volume bền vững, giới hạn filesystem permission và backup có kiểm soát.
- Cần private validation/test set, đánh giá theo identity/capture session và review pháp lý trước khi gọi là production.

## 10. License

MIT License. Xem LICENSE.
