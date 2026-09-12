# 🎓 Face Attendance System

[![CI](https://github.com/haminhthong/Face-Attendance-System/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Face-Attendance-System/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)

Hệ thống điểm danh sinh viên bằng khuôn mặt (Face Attendance System) xây dựng trên nền tảng **Computer Vision & Biometrics Engineering**, tích hợp giao diện thời gian thực **Streamlit WebRTC**, dịch vụ phục vụ **FastAPI**, và cơ sở dữ liệu **SQLite**.

Dự án tập trung giải quyết bài toán cốt lõi trong nhận diện khuôn mặt thực tế: **Open-Set Face Recognition** (từ chối người lạ, không ép nhận diện vào một sinh viên gần nhất), kết hợp kiểm tra tương tác chớp mắt (**Eye Aspect Ratio - EAR**) và xác thực ổn định đa khung hình (**Temporal Confirmation**).

---

## 1. Kiến trúc luồng xử lý (8 Bước cốt lõi)

Toàn bộ quy trình từ đăng ký mẫu đến điểm danh tự động được thiết kế trực diện, minh bạch và giải thích được:

```text
       Enrollment images (5 ảnh)
                  │
                  ▼
          Detect one face
       (Blur, Brightness, Size)
                  │
                  ▼
       Identity Consistency Check
        (Pairwise distance <= 0.50)
                  │
                  ▼
        Create 128D embeddings
           (Store in SQLite)
                  │
                  ▼
            Webcam frame
         (Streamlit WebRTC)
                  │
                  ▼
         Single-Face Gate
       (Exactly 1 face in frame)
                  │
                  ▼
        Open-Set Recognition
      Euclidean L2 (Top-K Mean)
     distance <= 0.50  &  margin >= 0.05
                  │
                  ▼
       Blink Challenge (EAR FSM)
       (eyes open -> closed -> open)
                  │
                  ▼
         Temporal Confirmation
      (>= 3 frames, >= 500 ms stable)
                  │
                  ▼
         Mark attendance
      (Present / Late in SQLite)
```

---

## 2. Kỹ thuật Computer Vision & Nhận diện sinh trắc học

### 2.1. Open-Set Recognition vs Closed-Set Classification

Trong môi trường thực tế, hệ thống không thể hoạt động như một bộ phân loại đóng (Closed-Set Classifier - luôn gán khuôn mặt vào 1 trong các sinh viên có sẵn). Dự án triển khai thuật toán **Open-Set Recognition** với cơ chế từ chối kép:

1. **Khoảng cách tối đa (Distance Threshold $\le 0.50$):**
   - Tính khoảng cách Euclidean L2 giữa vector truy vấn (128D) và các mẫu tham chiếu của sinh viên.
   - Nếu khoảng cách gần nhất $> 0.50$, hệ thống từ chối và xếp vào nhóm **Người lạ (Unknown)**.
2. **Độ phân biệt Top-1 vs Top-2 (Identity Margin $\ge 0.05$):**
   - Trường hợp ứng viên Top-1 có $d_1 = 0.41$ và ứng viên Top-2 có $d_2 = 0.42$: Dù cả hai đều $< 0.50$, việc khoảng cách quá sát nhau ($\Delta = 0.01$) chứng tỏ sự mơ hồ danh tính cao.
   - Hệ thống yêu cầu $\text{Margin} = d_2 - d_1 \ge 0.05$ để khẳng định sự tách bạch rõ ràng giữa hai danh tính.

$$\text{Decision} = \begin{cases} \text{Accept}(ID_1), & \text{nếu } d_1 \le 0.50 \text{ và } (d_2 - d_1) \ge 0.05 \\ \text{Unknown}, & \text{ngược lại} \end{cases}$$

### 2.2. Chiến lược gom mẫu Top-K Mean ($K=2$)

Mỗi sinh viên được đăng ký tối thiểu 5 ảnh ở các góc mặt và điều kiện ánh sáng khác nhau. Khi so khớp:
- Hệ thống lấy $K=2$ template tham chiếu gần vector truy vấn nhất của sinh viên đó và tính khoảng cách trung bình:

$$d_{\text{rep}} = \frac{1}{K} \sum_{i=1}^{K} d_{(i)}$$

- **Lợi ích kỹ thuật:** Giảm phụ thuộc vào một ảnh chụp bất thường duy nhất (như Min-Distance), đồng thời giữ được độ nhạy với các góc nghiêng tự nhiên tốt hơn so với Centroid đơn lẻ.

### 2.3. Cổng kiểm tra chất lượng ảnh đăng ký (Enrollment Quality Gates)

Khi đăng ký hồ sơ sinh viên:
- **Single-Face Enforcement:** Bắt buộc mỗi ảnh có chính xác 1 khuôn mặt.
- **Độ sắc nét (Blur Score):** Biến thiên Laplacian $\ge 40.0$ để loại bỏ ảnh mờ, nhòe do chuyển động.
- **Độ sáng (Brightness):** Trung bình thang xám từ $40.0$ đến $220.0$ để tránh ngược sáng hoặc cháy sáng.
- **Kích thước mặt:** Vùng khuôn mặt tối thiểu $100 \times 100\text{ px}$.
- **Tính nhất quán danh tính (Identity Consistency):** Tính khoảng cách pairwise giữa toàn bộ các ảnh tải lên của sinh viên. Nếu phát hiện khoảng cách giữa hai ảnh bất kỳ $> 0.50$, toàn bộ đợt đăng ký sẽ bị hủy để tránh trộn ảnh nhiều người vào cùng một hồ sơ.

### 2.4. Basic Blink Challenge (Eye Aspect Ratio - EAR)

Nhằm giảm thiểu việc sử dụng ảnh tĩnh để qua mặt camera:
- Tính tỉ lệ mắt (Eye Aspect Ratio) dựa trên 6 điểm mốc mí mắt:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- **Finite State Machine (FSM):**
  1. `can_mo` (Mắt mở, $\text{EAR} \ge 0.23$)
  2. `can_nham` (Mắt nhắm, $\text{EAR} \le 0.19$)
  3. `can_mo_lai` (Mắt mở lại, $\text{EAR} \ge 0.23$)
  4. `da_xac_minh` (Hoàn tất xác nhận chớp mắt).

> [!NOTE]
> *Blink challenge là một heuristic tương tác cơ bản, giúp ngăn chặn gian lận bằng ảnh in tĩnh đơn giản. Hệ thống không tự nhận là giải pháp Face PAD/Anti-spoofing chuyên dụng chống deepfake hoặc video replay.*

### 2.5. Single-Face Gate & Temporal Confirmation

- **Single-Face Gate:** Trong luồng camera realtime, nếu phát hiện 0 hoặc $> 1$ khuôn mặt, hệ thống lập tức đặt trạng thái cảnh báo và reset bộ đếm xác thực. Chỉ khi có duy nhất 1 người trước camera, pipeline nhận diện mới chạy.
- **Temporal Confirmation:** Yêu cầu danh tính duy nhất phải khớp liên tiếp tối thiểu 3 khung hình và duy trì ổn định ít nhất $500\text{ ms}$ trước khi ghi nhận điểm danh, triệt tiêu hiện tượng giật nhãn (label jittering).

---

## 3. Nghiệp vụ điểm danh (Business Logic)

- **Snapshot danh sách lớp (Session Roster Snapshot):** Khi tạo buổi học, hệ thống tạo bản chụp (snapshot) danh sách sinh viên của môn học tại thời điểm đó vào `session_enrollments`. Điều này đảm bảo lịch sử điểm danh không bị thay đổi nếu danh sách lớp môn học có sự điều chỉnh sau này.
- **Phân loại Có mặt / Đi trễ:**
  - $\text{check\_in} \le \text{start\_time} + \text{late\_after\_minutes} \rightarrow \text{Có mặt (Present)}$
  - $\text{check\_in} > \text{start\_time} + \text{late\_after\_minutes} \rightarrow \text{Đi trễ (Late)}$
  - Không điểm danh $\rightarrow \text{Vắng (Absent)}$ trong báo cáo.
- **Chống điểm danh trùng lặp (Duplicate Prevention):** Mỗi sinh viên chỉ có thể điểm danh 1 lần duy nhất trong mỗi buổi học (ràng buộc `UNIQUE(session_id, student_id)` trong SQLite kết hợp khóa mức dịch vụ).
- **Can thiệp thủ công (Manual Override):** Giảng viên có thể điều chỉnh trạng thái điểm danh cho sinh viên kèm thông tin người duyệt và lý do (ví dụ: camera mờ, xác nhận qua thẻ SV trực tiếp).

---

## 4. Cấu trúc mã nguồn

Kiến trúc được tổ chức gọn gàng, module hóa cao và không bị overengineering:

```text
face-attendance-system/
├── app.py                      # Điểm chạy chính Streamlit Dashboard
├── pyproject.toml              # Cấu hình dự án và dependencies
├── src/
│   └── face_attendance/
│       ├── __init__.py         # Package entry & version
│       ├── config.py           # RecognitionConfig, tham số ngưỡng & môi trường
│       ├── database.py         # SQLite connection, schema 6 bảng & truy vấn
│       ├── matcher.py          # Thuật toán Open-Set matching & Top-K Mean
│       ├── liveness.py         # EAR calculation & BlinkDetector FSM
│       ├── enrollment.py       # Quality gates, identity consistency & đăng ký mẫu
│       ├── attendance.py       # Nghiệp vụ điểm danh, roster & manual override
│       ├── recognition.py      # RecognitionEngine, Single-Face Gate, WebRTC processor
│       ├── api.py              # FastAPI endpoints (/health, /recognize, /attendance)
│       ├── ui.py               # Giao diện Streamlit: Điểm danh & Quản trị
│       └── utils.py            # Tiện ích thời gian VN/UTC, chuẩn hóa chuỗi
├── tests/
│   ├── test_matcher.py         # Test ngưỡng khoảng cách, margin, top-k mean
│   ├── test_liveness.py        # Test EAR & chu trình chớp mắt FSM
│   ├── test_single_face_gate.py# Test Single-Face Gate & candidate switching
│   ├── test_enrollment.py      # Test quality gates & identity consistency
│   ├── test_attendance.py      # Test present/late, duplicate check, manual override
│   ├── test_database.py        # Test SQLite CRUD & session roster snapshot
│   ├── test_load_concurrency.py# Test chịu tải & khóa ghi nhận đồng thời
│   ├── test_image_validation.py# Test giải mã ảnh lỗi, ảnh mờ, tối
│   ├── test_api.py             # Test FastAPI serving
│   └── test_config.py          # Test thông số RecognitionConfig
└── .github/
    └── workflows/
        └── ci.yml              # CI workflow: Ruff lint, format & Pytest
```

---

## 5. Hướng dẫn cài đặt & Chạy ứng dụng

### 5.1. Yêu cầu hệ thống

- Python $\ge$ 3.11
- Thư viện C/C++ build tools cho `dlib` (CMake, build-essential/MSVC C++ build tools)

### 5.2. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/haminhthong/Face-Attendance-System.git
cd Face-Attendance-System

# Tạo và kích hoạt virtualenv
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt project và dev dependencies
pip install -e ".[dev]"
```

### 5.3. Khởi chạy Dashboard Streamlit

```bash
streamlit run app.py
```
Mở trình duyệt tại `http://localhost:8501`.
- **Trang Điểm danh:** Chọn buổi học đang mở và cho phép webcam trên trình duyệt.
- **Trang Quản trị:** Nhập mã PIN (mặc định: `123456`, có thể chỉnh qua biến môi trường `ADMIN_PIN`).

### 5.4. Khởi chạy FastAPI Service

```bash
uvicorn face_attendance.api:app --reload --port 8000
```
Tài liệu tương tác Swagger UI có sẵn tại `http://localhost:8000/docs`:
- `GET /health`: Kiểm tra sức khỏe dịch vụ.
- `POST /recognize`: Nhận diện khuôn mặt từ vector 128D.
- `GET /sessions/{session_id}/attendance`: Lấy danh sách và báo cáo điểm danh của buổi học.

### 5.5. Chạy kiểm thử tự động (Unit Tests & Linting)

```bash
# Kiểm tra code style với Ruff
ruff check app.py src tests
ruff format --check app.py src tests

# Chạy toàn bộ test suite
pytest -v -p no:cacheprovider
```

---

## 6. Chính sách bảo mật & Dữ liệu cá nhân

- **Không lưu trữ ảnh gốc:** Hệ thống chỉ giải mã ảnh trong RAM, trích xuất vector đặc trưng 128D và lưu BLOB vector vào SQLite; file ảnh gốc bị hủy ngay sau khi xử lý.
- **Biometric Consent:** Bắt buộc có sự đồng ý của sinh viên trước khi trích xuất hoặc lưu trữ vector khuôn mặt.
- **Quyền thu hồi dữ liệu:** Quản trị viên có chức năng thu hồi/xóa toàn bộ vector khuôn mặt của sinh viên bất cứ lúc nào, trong khi lịch sử điểm danh học vụ vẫn được bảo lưu.
