"""Script chuẩn bị cấu trúc thư mục dữ liệu đánh giá và kiểm tra rò rỉ dữ liệu (Data Leakage).

Cấu trúc tạo ra:
data/
├── README.md
├── private/
│   ├── enrollment/
│   │   ├── person_001/
│   │   └── person_002/
│   ├── validation/
│   │   ├── known/
│   │   └── unknown/
│   └── test/
│       ├── known/
│       └── unknown/
└── results/
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PRIVATE_DIR = DATA_DIR / "private"
RESULTS_DIR = DATA_DIR / "results"

ENROLLMENT_DIR = PRIVATE_DIR / "enrollment"
VALIDATION_DIR = PRIVATE_DIR / "validation"
TEST_DIR = PRIVATE_DIR / "test"

SUB_DIRS = [
    ENROLLMENT_DIR,
    VALIDATION_DIR / "known",
    VALIDATION_DIR / "unknown",
    TEST_DIR / "known",
    TEST_DIR / "unknown",
    RESULTS_DIR,
]


def calculate_file_hash(path: Path) -> str:
    """Tính giá trị SHA-256 hash của một file.

    Args:
        path (Path): Đường dẫn file.

    Returns:
        str: Chuỗi hex hash SHA-256.
    """
    return sha256(path.read_bytes()).hexdigest()


def init_evaluation_dataset_structure() -> None:
    """Tạo cấu trúc thư mục đánh giá nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for folder in SUB_DIRS:
        folder.mkdir(parents=True, exist_ok=True)
        # Tạo file .gitkeep nếu thư mục rỗng
        gitkeep = folder / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    readme_path = DATA_DIR / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Data Directory Structure for Face Recognition Evaluation\n\n"
            "Chứa cấu trúc đánh giá AI độc lập:\n"
            "- `private/enrollment/`: Ảnh đăng ký của các sinh viên tham chiếu (3-5 ảnh/người).\n"
            "- `private/validation/`: Tập kiểm định dùng để dò threshold (chọn ngưỡng).\n"
            "- `private/test/`: Tập kiểm thử độc lập chỉ chạy sau khi đã chốt threshold.\n"
            "- `results/`: Kết quả chạy benchmark và biểu đồ.\n\n"
            "> **Lưu ý bảo mật**: Tất cả ảnh thật nằm trong `data/private/` được loại trừ bởi `.gitignore`.\n",
            encoding="utf-8",
        )
    LOGGER.info("Đã tạo cấu trúc thư mục dữ liệu tại %s", DATA_DIR)


def check_data_leakage() -> Dict[str, List[str]]:
    """Phát hiện ảnh trùng giữa các tập Enrollment, Validation, và Test bằng hash SHA-256.

    Returns:
        Dict[str, List[str]]: Mapping từ SHA-256 hash đến danh sách đường dẫn file bị lặp.
    """
    manifest_path = PRIVATE_DIR / "manifest.json"
    if manifest_path.exists():
        validate_capture_manifest(manifest_path)

    hashes: Dict[str, List[Path]] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root_dir in [ENROLLMENT_DIR, VALIDATION_DIR, TEST_DIR]:
        if not root_dir.exists():
            continue
        for file_path in root_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                file_hash = calculate_file_hash(file_path)
                hashes.setdefault(file_hash, []).append(file_path)

    duplicates = {
        h: [str(p.relative_to(DATA_DIR)) for p in paths]
        for h, paths in hashes.items()
        if len(paths) > 1
    }

    if duplicates:
        LOGGER.warning("⚠️  PHÁT HIỆN RÒ RỈ DỮ LIỆU (%d ảnh trùng giữa các tập):", len(duplicates))
        for h, file_list in duplicates.items():
            LOGGER.warning("  Hash %s...: %s", h[:8], " <-> ".join(file_list))
    else:
        LOGGER.info("✅ Không phát hiện ảnh trùng byte-for-byte (SHA-256) giữa các tập dữ liệu.")

    return duplicates


def validate_capture_manifest(manifest_path: Path | None = None) -> None:
    """É enforce capture-session split trong private manifest.

    Image gần nhau giữa các split phải bị loại ở bước pHash bên dưới; còn
    manifest này chặn trước các lỗi cấu trúc như dùng chung session hoặc đưa
    unknown identity vào enrollment.
    """
    path = manifest_path or PRIVATE_DIR / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    identities = data.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("Manifest phải có identities không rỗng.")

    seen_hashes: dict[str, str] = {}
    for identity in identities:
        identity_id = identity.get("identity_id")
        role = identity.get("role")
        if role not in {"known", "unknown"}:
            raise ValueError(f"{identity_id}: role phải là known hoặc unknown.")
        captures = identity.get("captures", [])
        split_sessions: dict[str, set[str]] = {}
        for capture in captures:
            split = capture.get("split")
            session = capture.get("capture_session")
            relative_path = capture.get("path")
            if split not in {"enrollment", "validation", "test"} or not session or not relative_path:
                raise ValueError(f"{identity_id}: capture thiếu split/session/path hợp lệ.")
            image_path = BASE_DIR / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh trong manifest: {image_path}")
            digest = calculate_file_hash(image_path)
            if digest in seen_hashes:
                raise ValueError(
                    f"Trùng SHA-256 giữa các split: {seen_hashes[digest]} và {relative_path}"
                )
            seen_hashes[digest] = relative_path
            split_sessions.setdefault(split, set()).add(str(session))

        if role == "known" and not split_sessions.get("enrollment"):
            raise ValueError(f"{identity_id}: known identity phải có enrollment.")
        if role == "unknown" and split_sessions.get("enrollment"):
            raise ValueError(f"{identity_id}: unknown identity không được có enrollment.")
        if (
            split_sessions.get("enrollment", set()) & split_sessions.get("validation", set())
            or split_sessions.get("enrollment", set()) & split_sessions.get("test", set())
        ):
            raise ValueError(f"{identity_id}: không được dùng chung capture_session với enrollment.")


def check_near_duplicates_phash(threshold: int = 3) -> Dict[str, List[str]]:
    """Phát hiện ảnh gần trùng (near-duplicates) giữa các tập dữ liệu bằng Perceptual Hash (pHash).

    Args:
        threshold: Ngưỡng khoảng cách Hamming tối đa (mặc định <= 3 coi là gần trùng).

    Returns:
        Dict[str, List[str]]: Danh sách các cặp ảnh bị nghi ngờ rò rỉ hoặc quá giống nhau.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        LOGGER.info("Thư viện imagehash chưa được cài đặt; bỏ qua kiểm tra pHash.")
        return {}

    phashes: list[tuple[str, Path, Any]] = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root_dir in [ENROLLMENT_DIR, VALIDATION_DIR, TEST_DIR]:
        if not root_dir.exists():
            continue
        for file_path in root_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                try:
                    with Image.open(file_path) as img:
                        ph = imagehash.phash(img)
                        phashes.append((str(file_path.relative_to(DATA_DIR)), file_path, ph))
                except Exception:
                    pass

    near_duplicates: Dict[str, List[str]] = {}
    for i in range(len(phashes)):
        for j in range(i + 1, len(phashes)):
            rel_i, _, ph_i = phashes[i]
            rel_j, _, ph_j = phashes[j]
            diff = ph_i - ph_j
            if diff <= threshold:
                key = f"{rel_i} <-> {rel_j}"
                near_duplicates[key] = [f"Hamming distance: {diff}"]

    if near_duplicates:
        LOGGER.warning(
            "⚠️  PHÁT HIỆN %d CẶP ẢNH GẦN TRÙNG (pHash Hamming distance <= %d):",
            len(near_duplicates),
            threshold,
        )
        for pair, detail in near_duplicates.items():
            LOGGER.warning("  %s (%s)", pair, detail[0])
    else:
        LOGGER.info("✅ Không phát hiện ảnh gần trùng (near-duplicates) bằng pHash.")

    return near_duplicates


if __name__ == "__main__":
    init_evaluation_dataset_structure()
    check_data_leakage()
    check_near_duplicates_phash()
