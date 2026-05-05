from __future__ import annotations

from typing import Any

from .config import PipelineConfig


STRICT_RELEASE_PROFILES = {
    "release_candidate",
    "hardware_review",
    "hardware",
    "physics_required",
    "production_upload",
}


def release_profile(config: PipelineConfig) -> str:
    return str(config.release.get("profile", "smoke")).strip().lower() or "smoke"


def release_requires(config: PipelineConfig, key: str) -> bool:
    value = config.release.get(key)
    if value is not None:
        return bool(value)
    return release_profile(config) in STRICT_RELEASE_PROFILES


def release_waiver(config: PipelineConfig, allow_key: str, reason_key: str) -> dict[str, Any]:
    allowed = bool(config.release.get(allow_key, False))
    reason = str(config.release.get(reason_key, "")).strip()
    return {
        "allowed": allowed and bool(reason),
        "configured": allowed,
        "reason": reason,
    }


def is_sample_policy_artifact_path(config: PipelineConfig, artifact_dir: str) -> bool:
    normalized = artifact_dir.replace("\\", "/").rstrip("/")
    configured = str(config.policy.get("artifact_dir", "")).strip().replace("\\", "/").rstrip("/")
    return normalized.endswith("golden/sample_inputs/policy_artifacts") or configured == "golden/sample_inputs/policy_artifacts"
