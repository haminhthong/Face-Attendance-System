# 🎓 Hệ thống điểm danh bằng khuôn mặt

> Ứng dụng prototype dùng Streamlit/WebRTC để hỗ trợ điểm danh trong lớp học bằng embedding khuôn mặt, nhận diện open-set trong phạm vi roster, kiểm tra chớp mắt cơ bản và lưu quyết định bằng SQLite.

Dự án ưu tiên tính đúng của luồng nghiệp vụ, khả năng truy vết và bảo vệ dữ liệu sinh trắc học. Đây chưa phải hệ thống chống giả mạo chuyên dụng hoặc một báo cáo độ chính xác biometric production.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Luồng xử lý hiện tại

### Đăng ký sinh viên

1. Tầng ứng dụng yêu cầu consent rõ ràng trước khi giải mã ảnh.
2. Luồng giao diện yêu cầu tối thiểu 5 ảnh của cùng một người.
3. Quality gate kiểm tra kích thước, độ mờ, ánh sáng, đúng một khuôn mặt và ảnh gần trùng bằng pHash.
4. Hệ thống tạo embedding 128 chiều, kiểm tra tính nhất quán giữa các ảnh và kiểm tra với profile hiện có.
5. Chỉ embedding kèm metadata model/version/policy được lưu vào SQLite; ảnh gốc không được lưu.

### Điểm danh realtime

~~~text
WebRTC frame
  -> lấy mẫu frame và chuẩn hóa ảnh
  -> phát hiện khuôn mặt, bắt buộc đúng 1 mặt
  -> tạo embedding
  -> chỉ so khớp template của roster trong session
  -> Top-2 Mean + distance threshold + identity margin
  -> kiểm tra chớp mắt EAR
  -> cùng identity đủ số quan sát và thời gian ổn định
  -> RecognitionDecision chứa đầy đủ policy evidence
  -> AttendanceService kiểm tra nghiệp vụ
  -> SQLite transaction ghi attendance/audit/telemetry
~~~

Các điều kiện từ chối được giữ riêng: không có mặt, nhiều mặt, ảnh kém chất lượng, unknown, ambiguous, liveness không đạt, session đóng, ngoài roster hoặc đã điểm danh. Human oversight là bước xử lý ngoại lệ/sửa có lý do; không phải yêu cầu giáo viên duyệt thủ công mọi lượt nhận diện.

## 2. Recognition policy

Policy dùng chung nằm ở `src/face_attendance/policy.py`. Cấu hình mẫu ở `configs/recognition_policy.example.json`.

Mặc định hiện tại:

| Thuộc tính | Giá trị |
|---|---|
| Model | `dlib_face_recognition_resnet_v1`, version `1` |
| Kích thước embedding | `128` |
| Metric | `euclidean_l2` |
| Runtime aggregation | `top_k_mean`, `top_k = 2` |
| Distance threshold | `0.50` |
| Identity margin | `0.05` |
| Temporal confirmation | ít nhất `3` quan sát và ổn định `500 ms` |
| Liveness | `ear_blink_v1`, TTL `10` giây |
| Calibrated | `false` trong policy mẫu |

`RECOGNITION_POLICY_PATH` cho phép nạp một policy JSON khác. Tất cả client biometric phải gửi policy evidence khớp policy server; API trả lỗi `409` nếu khác version, hash, threshold, model hoặc chiến lược.

Môi trường production bị từ chối khởi động khi policy chưa được hiệu chuẩn trên dữ liệu private phù hợp. Không cấu hình threshold rời rạc qua nhiều biến môi trường.

## 3. Dữ liệu và quyền riêng tư

- Không commit ảnh khuôn mặt, embedding thật, database hoặc secret vào Git.
- Consent được lưu theo phiên bản; thu hồi consent sẽ vô hiệu hóa và xóa embedding nhưng giữ lịch sử attendance/audit cần thiết.
- Embedding vẫn là dữ liệu sinh trắc học nhạy cảm, dù ảnh gốc không được lưu.
- `BIOMETRIC_RETENTION_DAYS` được dùng để dọn template quá hạn khi API khởi động.
- Bộ dữ liệu đánh giá private phải có sự đồng ý hợp lệ và được bảo vệ ngoài repository.
- `data/manifest.example.json` chỉ là ví dụ schema, không chứa ảnh thật.

Manifest private có dạng:

~~~text
data/private/manifest.json
data/private/enrollment/...
data/private/validation/known/...
data/private/validation/unknown/...
data/private/test/known/...
data/private/test/unknown/...
~~~

Mỗi capture cần `identity_id`, `role`, `path`, `split` và `capture_session`. Tool kiểm tra hash SHA-256, split hợp lệ và không dùng chung capture session giữa enrollment với validation/test của cùng identity.

## 4. Đánh giá và hiệu chuẩn

Hai loại kiểm tra được tách biệt:

### Synthetic matcher sanity

~~~powershell
python tools/prepare_dataset.py
python tools/matcher_sanity_check.py
~~~

`matcher_sanity_check.py` và `evaluate_matching.py` chỉ dùng vector mô phỏng để kiểm tra toán matcher, Top-K/centroid/min-distance, margin, reject logic và latency. Kết quả luôn có `deployable: false`; không được dùng để chọn policy production.

Min Distance và Centroid chỉ là chiến lược benchmark. Runtime attendance dùng Top-2 Mean.

### Private manifest gate

~~~powershell
python tools/evaluate_biometrics.py --manifest data/private/manifest.json
~~~

Tool này không có fallback synthetic. Nó hiện kiểm tra tính hợp lệ của manifest và chống leakage; chưa tự tạo metric biometric hay tự freeze policy. Vì vậy kết quả gate không phải bằng chứng accuracy và không được gọi là production calibration.

Ngưỡng chỉ được chọn trên validation. Test phải được giữ riêng, chạy sau khi policy đã khóa, và báo cáo tối thiểu FAR, FRR/TAR, wrong-ID, ambiguous rate và latency nếu đã triển khai đầy đủ phép đo.

## 5. Cấu trúc dự án

~~~text
face-attendance-system/
├── app.py                              # Điểm chạy Streamlit
├── src/face_attendance/
│   ├── domain/                         # Entity, enum và exception nghiệp vụ
│   ├── application/
│   │   ├── attendance_service.py       # Biometric/manual attendance
│   │   └── enrollment_service.py       # Luồng đăng ký có consent
│   ├── api.py                          # FastAPI endpoints
│   ├── config.py                       # Environment và policy runtime
│   ├── database.py                     # SQLite schema, migration, transaction, audit
│   ├── liveness.py                     # EAR blink state machine
│   ├── matcher.py                      # Open-set matching và aggregation
│   ├── policy.py                       # Schema/hash/default recognition policy
│   ├── recognition.py                  # Quality gate và realtime engine
│   ├── ui.py                           # Streamlit dashboard
│   └── utils.py                        # Tiện ích thời gian, chuỗi và bảo mật
├── configs/
│   └── recognition_policy.example.json # Policy mẫu, chưa calibrated
├── data/
│   ├── manifest.example.json            # Manifest minh họa
│   ├── private/                         # Dữ liệu thật, bị gitignore
│   └── results/                         # Kết quả chạy cục bộ, bị gitignore
├── tools/
│   ├── evaluate_biometrics.py           # Private manifest gate
│   ├── evaluate_matching.py             # Tên tương thích cho sanity check
│   ├── matcher_sanity_check.py          # Entry point sanity check
│   └── prepare_dataset.py               # Tạo thư mục và kiểm tra leakage
├── tests/                               # Unit/integration tests
├── .env.example                         # Cấu hình mẫu
├── Dockerfile
└── pyproject.toml
~~~

Các thư mục `__pycache__`, `.pytest_cache`, `scratch` và dữ liệu runtime không thuộc source deliverable; chúng đã được ignore khỏi Git.

## 6. API

Các endpoint cần `X-API-Key` trừ `/health`:

- `GET /health`: kiểm tra service.
- `GET /policy`: metadata policy đang chạy.
- `GET /sessions`: danh sách buổi học.
- `GET /sessions/{session_id}/attendance`: báo cáo điểm danh.
- `POST /attendance`: endpoint tương thích legacy.
- `POST /attendance/biometric`: nhận `RecognitionDecision` đầy đủ và ghi attendance.
- `POST /attendance/manual`: giảng viên điều chỉnh hoặc ghi nhận thủ công, bắt buộc lý do.

API biometric kiểm tra policy evidence ở server trước khi gọi application service. API không nhận threshold tùy ý để thay đổi quyết định runtime.

## 7. Cài đặt và chạy

### Tạo môi trường

~~~powershell
git clone <repository-url>
cd face-attendance-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
~~~

### Kiểm tra code

~~~powershell
python -m ruff check src tools
python -m pytest -q -p no:cacheprovider
~~~

Nếu môi trường thiếu OpenCV hoặc dependency native, hãy cài đầy đủ nhóm dependency trong `pyproject.toml` trước khi chạy test liên quan ảnh/WebRTC.

### Chạy dashboard

~~~powershell
streamlit run app.py
~~~

Mở `http://localhost:8501`.

### Chạy API

~~~powershell
uvicorn face_attendance.api:app --reload --port 8000
~~~

Swagger UI: `http://127.0.0.1:8000/docs`.

### Cấu hình

Copy `.env.example` thành `.env` và thay tối thiểu:

~~~text
APP_ENV=development
FACE_ATTENDANCE_API_KEY=<secret>
DATABASE_PATH=face_attendance_data/face_attendance.db
RECOGNITION_POLICY_PATH=
~~~

Không đưa `.env`, API key, database hoặc dữ liệu private vào Git.

## 8. Kiểm tra logic và giới hạn đã biết

- Enrollment application flow chặn thiếu consent trước khi decode ảnh và yêu cầu đủ 5 ảnh hợp lệ.
- Runtime chỉ nạp template đã consent, còn hiệu lực, đúng model/version/dimension và thuộc session roster.
- Matcher dùng distance và margin; temporal confirmation không thay thế liveness.
- Liveness hiện là blink heuristic, không phải hệ thống PAD chống replay/deepfake.
- Attendance được ghi trong transaction, chống duplicate và lưu evidence của quyết định.
- Telemetry chỉ lưu metadata attempt, không lưu embedding thô.
- `database.upsert_student(..., consent_given=None)` còn được giữ cho seed/legacy compatibility. Luồng ứng dụng chính không dùng giá trị này; khi xây production nên loại bỏ compatibility path sau khi migrate dữ liệu cũ.
- Private biometric evaluation chưa hoàn tất phần encoding/metric tự động; không có claim accuracy trong repository.

## 9. License

Dự án được phân phối theo MIT License.
