from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embodiments import discover_embodiments


DEFAULT_ACCEPTED_RAW_SOURCES = ["aligned_records", "csv_session", "rosbag2", "image_sequence"]


@dataclass(frozen=True)
class EmbodimentAdapter:
    root: Path
    embodiment_dir: Path | None
    manifest: dict[str, Any]

    @property
    def embodiment_id(self) -> str:
        if self.manifest.get("id"):
            return str(self.manifest["id"])
        return "generic/unknown"

    @property
    def embodiment_type(self) -> str:
        if self.manifest.get("embodiment_type"):
            return str(self.manifest["embodiment_type"])
        return self.embodiment_id.split("/", 1)[0]

    @property
    def real_data(self) -> dict[str, Any]:
        data = self.manifest.get("real_data", {})
        return data if isinstance(data, dict) else {}

    @property
    def canonical_pose_file(self) -> str:
        return str(self.real_data.get("canonical_pose_file", "object_pose.csv"))

    @property
    def accepted_pose_files(self) -> list[str]:
        files = self.real_data.get("accepted_pose_files")
        if isinstance(files, list) and files:
            return [str(item) for item in files]
        return [self.canonical_pose_file]

    @property
    def accepted_raw_sources(self) -> list[str]:
        sources = self.real_data.get("accepted_raw_sources")
        if isinstance(sources, list) and sources:
            return [str(item) for item in sources]
        return list(DEFAULT_ACCEPTED_RAW_SOURCES)

    @property
    def external_ingestor_command(self) -> list[str]:
        command = self.real_data.get("external_ingestor_command", [])
        if isinstance(command, list):
            return [str(item) for item in command]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.embodiment_id,
            "embodiment_type": self.embodiment_type,
            "path": str(self.embodiment_dir.relative_to(self.root)) if self.embodiment_dir else None,
            "canonical_pose_file": self.canonical_pose_file,
            "accepted_pose_files": self.accepted_pose_files,
            "accepted_raw_sources": self.accepted_raw_sources,
            "has_external_ingestor": bool(self.external_ingestor_command),
        }


def load_embodiment_adapter(
    root: str | Path,
    embodiment_id: str | None = None,
    session_dir: str | Path | None = None,
) -> EmbodimentAdapter:
    root_path = Path(root).expanduser().resolve()
    embodiment_dir = _resolve_embodiment_dir(root_path, embodiment_id, session_dir)
    if embodiment_dir is None:
        return EmbodimentAdapter(root=root_path, embodiment_dir=None, manifest=_default_manifest())

    manifest_path = embodiment_dir / "embodiment.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else _default_manifest()
    return EmbodimentAdapter(root=root_path, embodiment_dir=embodiment_dir, manifest=manifest)


def _resolve_embodiment_dir(
    root: Path,
    embodiment_id: str | None,
    session_dir: str | Path | None,
) -> Path | None:
    if embodiment_id:
        candidate = root / "embodiments" / embodiment_id
        return candidate if candidate.exists() else None

    if session_dir:
        session = Path(session_dir).expanduser()
        if not session.is_absolute():
            root_relative = (root / session).resolve()
            cwd_relative = (Path.cwd() / session).resolve()
            session = root_relative if root_relative.exists() else cwd_relative
        else:
            session = session.resolve()
        for embodiment in discover_embodiments(root):
            real_data = (embodiment / "real_data").resolve()
            if _is_relative_to(session, real_data):
                return embodiment.resolve()
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _default_manifest() -> dict[str, Any]:
    return {
        "id": "generic/unknown",
        "embodiment_type": "generic",
        "name": "unknown",
        "real_data": {
            "canonical_pose_file": "object_pose.csv",
            "accepted_pose_files": ["object_pose.csv", "shaft_pose.csv"],
            "accepted_raw_sources": list(DEFAULT_ACCEPTED_RAW_SOURCES),
            "external_ingestor_command": [],
        },
    }
