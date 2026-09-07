"""Chính sách nhận diện dùng chung cho toàn bộ pipeline.

Mọi thành phần liên quan đến matcher, liveness và attendance đều nhận cùng một
đối tượng ``RecognitionPolicy``. Nhờ đó ngưỡng được ghi vào cơ sở dữ liệu luôn
trùng với ngưỡng thực sự đã dùng khi nhận diện.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RecognitionPolicy:
    """Cấu hình bất biến cho một phiên bản chính sách nhận diện."""

    schema_version: int = 2
    policy_version: str = "face-policy-v1"
    embedding_model: str = "dlib_face_recognition_resnet_v1"
    embedding_model_version: str = "1"
    embedding_dimension: int = 128
    metric: str = "euclidean_l2"
    aggregation_strategy: str = "top_k_mean"
    top_k: int = 2
    distance_threshold: float = 0.50
    identity_margin: float = 0.05
    minimum_observations: int = 3
    stable_duration_ms: int = 500
    liveness_policy: str = "ear_blink_v1"
    liveness_ttl_seconds: float = 10.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        """Bảo vệ policy khỏi các giá trị có thể làm lệch logic runtime."""
        if self.embedding_dimension != 128:
            raise ValueError("Phiên bản hiện tại chỉ hỗ trợ embedding 128 chiều.")
        if self.metric != "euclidean_l2":
            raise ValueError("Metric hiện tại phải là euclidean_l2.")
        if self.aggregation_strategy != "top_k_mean":
            raise ValueError("Runtime chỉ hỗ trợ chiến lược top_k_mean.")
        if not 0 < self.top_k <= 5:
            raise ValueError("top_k phải nằm trong khoảng 1-5.")
        if not 0 <= self.distance_threshold <= 2:
            raise ValueError("distance_threshold phải nằm trong khoảng 0-2.")
        if not 0 <= self.identity_margin <= 2:
            raise ValueError("identity_margin phải nằm trong khoảng 0-2.")
        if self.minimum_observations < 1 or self.stable_duration_ms < 0:
            raise ValueError("Cấu hình temporal không hợp lệ.")
        if self.liveness_ttl_seconds <= 0:
            raise ValueError("liveness_ttl_seconds phải lớn hơn 0.")

    @property
    def policy_hash(self) -> str:
        """Hash ổn định để lưu bằng chứng policy trong audit trail."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Chuyển policy sang dict JSON-friendly."""
        return asdict(self)

    def matches_evidence(
        self,
        *,
        policy_version: str,
        distance_threshold: float,
        margin_threshold: float,
        aggregation_strategy: str,
        embedding_model: str,
        embedding_model_version: str,
        stable_duration_ms: int,
        liveness_policy: str,
        recognition_policy_hash: str,
    ) -> bool:
        """Kiểm tra evidence có đúng policy đang chạy hay không."""
        return (
            policy_version == self.policy_version
            and distance_threshold == self.distance_threshold
            and margin_threshold == self.identity_margin
            and aggregation_strategy == self.aggregation_strategy
            and embedding_model == self.embedding_model
            and embedding_model_version == self.embedding_model_version
            and stable_duration_ms == self.stable_duration_ms
            and liveness_policy == self.liveness_policy
            and recognition_policy_hash == self.policy_hash
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecognitionPolicy":
        """Đọc cả format phẳng cũ và format nhóm theo embedding/matching."""
        embedding = data.get("embedding", {})
        matching = data.get("matching", {})
        temporal = data.get("temporal", {})
        liveness = data.get("liveness", {})
        return cls(
            schema_version=int(data.get("schema_version", 2)),
            policy_version=str(data.get("policy_version", "face-policy-v1")),
            embedding_model=str(
                data.get("embedding_model", embedding.get("model", cls.embedding_model))
            ),
            embedding_model_version=str(
                data.get(
                    "embedding_model_version",
                    embedding.get("version", cls.embedding_model_version),
                )
            ),
            embedding_dimension=int(data.get("embedding_dim", embedding.get("dimension", 128))),
            metric=str(data.get("distance_metric", matching.get("metric", cls.metric))),
            aggregation_strategy=str(
                data.get(
                    "aggregation_strategy",
                    matching.get("aggregation", cls.aggregation_strategy),
                )
            ),
            top_k=int(data.get("top_k", matching.get("top_k", 2))),
            distance_threshold=float(
                data.get(
                    "distance_threshold",
                    matching.get("distance_threshold", cls.distance_threshold),
                )
            ),
            identity_margin=float(
                data.get(
                    "identity_margin",
                    matching.get("identity_margin", cls.identity_margin),
                )
            ),
            minimum_observations=int(
                data.get(
                    "minimum_observations",
                    temporal.get("minimum_observations", 3),
                )
            ),
            stable_duration_ms=int(
                data.get("stable_duration_ms", temporal.get("stable_duration_ms", 500))
            ),
            liveness_policy=str(
                data.get("liveness_policy", liveness.get("mode", cls.liveness_policy))
            ),
            liveness_ttl_seconds=float(
                data.get(
                    "liveness_ttl_seconds",
                    liveness.get("ttl_seconds", cls.liveness_ttl_seconds),
                )
            ),
            calibrated=bool(data.get("calibrated", False)),
        )


def load_recognition_policy(path: str | Path | None = None) -> RecognitionPolicy:
    """Nạp policy từ file nếu được cấu hình, nếu không dùng policy mặc định an toàn.

    Policy mặc định chỉ phục vụ chạy ứng dụng và kiểm thử. Nó không được xem là
    threshold đã hiệu chuẩn sinh trắc học cho đến khi có private validation set.
    """
    configured_path = path or os.getenv("RECOGNITION_POLICY_PATH", "").strip()
    if not configured_path:
        return RecognitionPolicy()

    policy_path = Path(configured_path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy recognition policy: {policy_path}")
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không thể đọc recognition policy: {policy_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Recognition policy phải là một JSON object.")
    return RecognitionPolicy.from_mapping(data)


DEFAULT_RECOGNITION_POLICY = load_recognition_policy()
