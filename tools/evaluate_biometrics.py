"""Đánh giá biometric trên private dataset đã tách theo capture session.

Tool này cố ý fail nếu dataset/manifest chưa có. Không được fallback sang
synthetic data vì threshold từ synthetic không có giá trị phát hành.
"""

from __future__ import annotations  # noqa: I001

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BASE_DIR / "data" / "private" / "manifest.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Kiểm tra các invariant chống leakage trước khi đánh giá."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Chưa có private manifest: {manifest_path}. "
            "Không được dùng synthetic data để thay thế."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities = data.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("Manifest phải có danh sách identities không rỗng.")

    seen_hashes: dict[str, str] = {}
    for identity in identities:
        role = identity.get("role")
        if role not in {"known", "unknown"}:
            raise ValueError("role phải là known hoặc unknown.")
        captures = identity.get("captures", [])
        if not captures:
            raise ValueError(f"Identity {identity.get('identity_id')} chưa có captures.")
        sessions_by_split: dict[str, set[str]] = {}
        for capture in captures:
            split = capture.get("split")
            session = capture.get("capture_session")
            image_path = BASE_DIR / capture.get("path", "")
            if split not in {"enrollment", "validation", "test"}:
                raise ValueError("split phải là enrollment, validation hoặc test.")
            if not session:
                raise ValueError("Mỗi capture phải có capture_session.")
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh trong manifest: {image_path}")
            digest = _file_hash(image_path)
            previous = seen_hashes.get(digest)
            if previous is not None:
                raise ValueError(
                    f"Phát hiện trùng SHA-256 giữa các split: {previous} và {image_path}"
                )
            seen_hashes[digest] = str(image_path)
            sessions_by_split.setdefault(split, set()).add(str(session))

        if role == "known" and len(sessions_by_split.get("enrollment", set())) == 0:
            raise ValueError(f"Known identity {identity.get('identity_id')} thiếu enrollment.")
        if role == "known" and (
            sessions_by_split.get("enrollment", set()) & sessions_by_split.get("validation", set())
            or sessions_by_split.get("enrollment", set()) & sessions_by_split.get("test", set())
        ):
            raise ValueError(
                f"Identity {identity.get('identity_id')} dùng chung capture_session giữa enrollment và đánh giá."
            )

    return data


def evaluate(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Validate manifest và trả metadata; phần encoding/evaluation chạy sau gate này."""
    manifest = validate_manifest(manifest_path)
    return {
        "schema_version": 1,
        "dataset_type": "private_real_biometric",
        "deployable": True,
        "identities": len(manifest["identities"]),
        "manifest_sha256": _file_hash(manifest_path),
        "note": "Threshold chỉ được chọn trên validation và test chỉ chạy sau khi freeze policy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.manifest)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
