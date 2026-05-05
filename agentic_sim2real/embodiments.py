from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_DIRS = ("configs", "artifacts", "evaluation", "real_data")
REQUIRED_SESSION_DIRS = (
    "camera_data",
    "camera_data/color",
    "camera_data/depth",
    "joint_data",
    "pose_data",
    "contact_data",
    "episode_labels",
    "calibration",
    "aligned",
)
REQUIRED_TEMPLATE_DIRS = REQUIRED_SESSION_DIRS


def discover_embodiments(root: str | Path) -> list[Path]:
    base = Path(root).expanduser() / "embodiments"
    if not base.exists():
        return []
    found: list[Path] = []
    for type_dir in sorted(base.iterdir()):
        if not type_dir.is_dir() or type_dir.name.startswith("."):
            continue
        for embodiment_dir in sorted(type_dir.iterdir()):
            if embodiment_dir.is_dir() and (embodiment_dir / "real_data").exists():
                found.append(embodiment_dir)
    return found


def validate_embodiments(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser()
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for embodiment_dir in discover_embodiments(root_path):
        result = validate_embodiment(root_path, embodiment_dir)
        results[result["id"]] = result
        if result["status"] != "pass":
            failures.append({"id": result["id"], "errors": result["errors"]})
    return {
        "status": "pass" if not failures else "fail",
        "embodiments": results,
        "failures": failures,
    }


def validate_embodiment(root: Path, embodiment_dir: Path) -> dict[str, Any]:
    rel = embodiment_dir.relative_to(root / "embodiments")
    embodiment_id = str(rel)
    errors: list[str] = []

    manifest_path = embodiment_dir / "embodiment.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append("missing embodiment.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"embodiment.json is invalid JSON: {exc}")

    if manifest:
        if manifest.get("id") != embodiment_id:
            errors.append(f"manifest id {manifest.get('id')!r} does not match path {embodiment_id!r}")
        if manifest.get("embodiment_type") != rel.parts[0]:
            errors.append("manifest embodiment_type does not match path")
        if manifest.get("name") != rel.parts[1]:
            errors.append("manifest name does not match path")
        real_data_contract = manifest.get("real_data", {})
        if not isinstance(real_data_contract.get("accepted_pose_files", []), list):
            errors.append("real_data.accepted_pose_files must be a list")
        if not isinstance(real_data_contract.get("accepted_raw_sources", []), list):
            errors.append("real_data.accepted_raw_sources must be a list")
        if not isinstance(real_data_contract.get("external_ingestor_command", []), list):
            errors.append("real_data.external_ingestor_command must be a list")

    for dirname in REQUIRED_TOP_LEVEL_DIRS:
        if not (embodiment_dir / dirname).is_dir():
            errors.append(f"missing required directory: {dirname}")

    real_data = embodiment_dir / "real_data"
    template_root = real_data / "templates"
    example_root = real_data / "example_session"
    for dirname in REQUIRED_TEMPLATE_DIRS:
        if not (template_root / dirname).is_dir():
            errors.append(f"missing real_data/templates/{dirname}")
    for dirname in REQUIRED_SESSION_DIRS:
        if not (example_root / dirname).is_dir():
            errors.append(f"missing real_data/example_session/{dirname}")

    pose_file = str(manifest.get("real_data", {}).get("canonical_pose_file", "object_pose.csv"))
    for base, label in ((template_root, "templates"), (example_root, "example_session")):
        if not (base / "joint_data" / "joint_states.csv").exists():
            errors.append(f"missing real_data/{label}/joint_data/joint_states.csv")
        if not (base / "pose_data" / pose_file).exists():
            errors.append(f"missing real_data/{label}/pose_data/{pose_file}")
        if not (base / "contact_data" / "contact.csv").exists():
            errors.append(f"missing real_data/{label}/contact_data/contact.csv")
        if not (base / "episode_labels" / "labels.csv").exists():
            errors.append(f"missing real_data/{label}/episode_labels/labels.csv")
        if not (base / "calibration" / "calibration.json").exists():
            errors.append(f"missing real_data/{label}/calibration/calibration.json")
        if not (base / "camera_data" / "index.csv").exists():
            errors.append(f"missing real_data/{label}/camera_data/index.csv")

    if not (example_root / "aligned" / "records.jsonl").exists():
        errors.append("missing real_data/example_session/aligned/records.jsonl")
    if not (example_root / "aligned" / "prepare_summary.json").exists():
        errors.append("missing real_data/example_session/aligned/prepare_summary.json")

    return {
        "id": embodiment_id,
        "path": str(embodiment_dir.relative_to(root)),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "pose_file": pose_file,
        "accepted_raw_sources": manifest.get("real_data", {}).get("accepted_raw_sources", []),
        "top_level_dirs": list(REQUIRED_TOP_LEVEL_DIRS),
        "session_dirs": list(REQUIRED_SESSION_DIRS),
    }
