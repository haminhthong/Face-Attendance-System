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
    seen_identity_ids: set[str] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            raise ValueError("Mỗi identity trong manifest phải là một object.")
        identity_id = str(identity.get("identity_id", "")).strip()
        if not identity_id or identity_id in seen_identity_ids:
            raise ValueError("identity_id phải tồn tại và không được trùng.")
        seen_identity_ids.add(identity_id)
        role = identity.get("role")
        if role not in {"known", "unknown"}:
            raise ValueError("role phải là known hoặc unknown.")
        captures = identity.get("captures", [])
        if not isinstance(captures, list) or not captures:
            raise ValueError(f"Identity {identity_id} chưa có captures.")
        sessions_by_split: dict[str, set[str]] = {}
        for capture in captures:
            if not isinstance(capture, dict):
                raise ValueError(f"Identity {identity_id} có capture không hợp lệ.")
            split = capture.get("split")
            session = capture.get("capture_session")
            relative_path = Path(str(capture.get("path", "")))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Đường dẫn capture không được thoát khỏi repository: {relative_path}"
                )
            image_path = (BASE_DIR / relative_path).resolve()
            if split not in {"enrollment", "validation", "test"}:
                raise ValueError("split phải là enrollment, validation hoặc test.")
            if not session:
                raise ValueError("Mỗi capture phải có capture_session.")
            if BASE_DIR not in image_path.parents or not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh trong manifest: {image_path}")
            digest = _file_hash(image_path)
            previous = seen_hashes.get(digest)
            if previous is not None:
                raise ValueError(
                    f"Phát hiện trùng SHA-256 giữa các split: {previous} và {image_path}"
                )
            seen_hashes[digest] = str(image_path)
            sessions_by_split.setdefault(split, set()).add(str(session))

        if role == "known" and not sessions_by_split.get("enrollment"):
            raise ValueError(f"Known identity {identity_id} thiếu enrollment.")
        if role == "unknown" and sessions_by_split.get("enrollment"):
            raise ValueError(f"Unknown identity {identity_id} không được có enrollment.")
        if role == "known" and (
            sessions_by_split.get("enrollment", set()) & sessions_by_split.get("validation", set())
            or sessions_by_split.get("enrollment", set()) & sessions_by_split.get("test", set())
        ):
            raise ValueError(
                f"Identity {identity_id} dùng chung capture_session giữa enrollment và đánh giá."
            )

    return data


def evaluate(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Validate manifest và trả metadata; chưa chạy encoding hoặc hiệu chuẩn."""
    manifest = validate_manifest(manifest_path)
    return {
        "schema_version": 1,
        "dataset_type": "private_real_biometric",
        "deployable": False,
        "status": "manifest_valid_only",
        "identities": len(manifest["identities"]),
        "manifest_sha256": _file_hash(manifest_path),
        "note": (
            "Manifest hợp lệ nhưng tool chưa encode ảnh, tính metric hoặc freeze policy. "
            "Không dùng kết quả này để phát hành threshold."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
