from __future__ import annotations

import base64
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .data_quality import evaluate_real_data_quality
from .dataset import load_records


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class MutationSpec:
    mutation_id: str
    description: str
    expected_alert_codes: tuple[str, ...]
    apply: Callable[[list[dict[str, Any]], Path], None]


def run_mutation_suite(
    source_dataset: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Create temporary golden-data variants and verify readiness alerts.

    The canonical golden dataset is never edited. Each variant is written under
    out_dir, then inspected through the same data-readiness gate used for real
    user datasets.
    """

    source = Path(source_dataset).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = [dict(record.raw or {}) for record in load_records(source)]
    variants = []

    for spec in [clean_baseline_spec(), *mutation_specs()]:
        variant_dir = out / spec.mutation_id
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_records = _clean_baseline_records(records)
        spec.apply(variant_records, variant_dir)
        _write_variant(variant_dir, variant_records)
        report = evaluate_real_data_quality(variant_dir, config, root=root)
        observed_codes = [
            str(item["code"])
            for item in report.get("data_readiness", {}).get("action_items", [])
        ]
        missing = sorted(set(spec.expected_alert_codes) - set(observed_codes))
        unexpected = sorted(set(observed_codes) - set(spec.expected_alert_codes))
        passed = not missing and (spec.mutation_id != "clean_baseline" or not observed_codes)
        variants.append(
            {
                "mutation_id": spec.mutation_id,
                "description": spec.description,
                "variant_path": str(variant_dir),
                "expected_alert_codes": list(spec.expected_alert_codes),
                "observed_alert_codes": observed_codes,
                "missing_expected_alert_codes": missing,
                "unexpected_alert_codes": unexpected,
                "status": "pass" if passed else "fail",
                "readiness_status": report.get("data_readiness", {}).get("status"),
                "readiness_summary": {
                    "frame_link_coverage": report.get("data_readiness", {}).get("frame_link_coverage"),
                    "heldout_frame_link_coverage": report.get("data_readiness", {}).get("heldout_frame_link_coverage"),
                    "orphan_frame_count": report.get("data_readiness", {}).get("orphan_frame_count"),
                    "delay_observability_status": report.get("data_readiness", {}).get("delay_observability_status"),
                    "pose_validation_source": report.get("data_readiness", {}).get("pose_validation", {}).get("validation_source"),
                    "success_label_trust": report.get("data_readiness", {}).get("success_labels", {}).get("trust_level"),
                },
            }
        )

    payload = {
        "schema": "agentic_sim2real.golden_mutation_report.v1",
        "status": "pass" if all(item["status"] == "pass" for item in variants) else "fail",
        "source_dataset": str(source),
        "out_dir": str(out),
        "variant_count": len(variants),
        "variants": variants,
        "notes": [
            "The canonical golden dataset is immutable during this run.",
            "Each variant starts from a clean synthetic baseline, then applies one targeted defect.",
            "Expected alert codes must be observed by the generic data-readiness gate.",
        ],
    }
    (out / "mutation_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "mutation_report.md").write_text(render_mutation_report(payload) + "\n")
    return payload


def clean_baseline_spec() -> MutationSpec:
    return MutationSpec(
        mutation_id="clean_baseline",
        description="Repaired synthetic baseline; should have no data-readiness alerts.",
        expected_alert_codes=(),
        apply=lambda records, variant_dir: None,
    )


def mutation_specs() -> list[MutationSpec]:
    return [
        MutationSpec(
            mutation_id="drop_all_camera_links",
            description="Remove every camera link and image file reference.",
            expected_alert_codes=("heldout_frames_missing", "low_frame_link_coverage", "episode_frame_gaps"),
            apply=_drop_all_camera_links,
        ),
        MutationSpec(
            mutation_id="heldout_without_frames",
            description="Remove camera links only from heldout episodes.",
            expected_alert_codes=("heldout_frames_missing", "episode_frame_gaps"),
            apply=_drop_heldout_camera_links,
        ),
        MutationSpec(
            mutation_id="orphan_sbl_frames",
            description="Add extracted SBL frames that are not referenced by canonical records.",
            expected_alert_codes=("orphan_frames_detected",),
            apply=_add_orphan_sbl_frames,
        ),
        MutationSpec(
            mutation_id="low_rate_telemetry",
            description="Lower trajectory sample rate below the configured delay-observability floor.",
            expected_alert_codes=("delay_under_sampled",),
            apply=_lower_sample_rate,
        ),
        MutationSpec(
            mutation_id="auto_positive_labels",
            description="Convert labels to all-positive threshold-generated labels.",
            expected_alert_codes=("auto_positive_success_labels",),
            apply=_make_auto_positive_labels,
        ),
        MutationSpec(
            mutation_id="fk_proxy_pose",
            description="Make pose estimate and reference identical FK proxies.",
            expected_alert_codes=("fk_proxy_pose_validation",),
            apply=_make_fk_proxy_pose,
        ),
    ]


def render_mutation_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Golden Dataset Mutation Report",
        "",
        f"- Status: {payload['status']}",
        f"- Source dataset: `{payload['source_dataset']}`",
        f"- Variants: {payload['variant_count']}",
        "",
        "| Variant | Status | Expected | Observed | Missing |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["variants"]:
        lines.append(
            "| {mutation_id} | {status} | `{expected}` | `{observed}` | `{missing}` |".format(
                mutation_id=item["mutation_id"],
                status=item["status"],
                expected=", ".join(item["expected_alert_codes"]) or "none",
                observed=", ".join(item["observed_alert_codes"]) or "none",
                missing=", ".join(item["missing_expected_alert_codes"]) or "none",
            )
        )
    return "\n".join(lines)


def _clean_baseline_records(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, int] = {}
    cleaned = []
    for record in source_records:
        item = dict(record)
        episode = int(item.get("episode_index", 0) or 0)
        idx = by_episode.get(episode, 0)
        by_episode[episode] = idx + 1
        scenario = _safe_name(str(item.get("scenario") or item.get("planner") or f"episode_{episode}"))
        timestamp = round(episode * 10.0 + idx * 0.01, 4)
        pose_ref = _pose_for_episode(episode)
        pose_est = [
            round(pose_ref[0] + 0.001 + 0.0001 * math.sin(idx), 6),
            round(pose_ref[1] - 0.001, 6),
            round(pose_ref[2] + 0.0005, 6),
            *pose_ref[3:],
        ]
        item.update(
            {
                "timestamp": timestamp,
                "scenario": scenario,
                "planner": scenario,
                "split": str(item.get("split") or "train").lower(),
                "joint_velocity": _vector_like(item.get("joint_state", []), 0.02),
                "contact_force": round(10.0 + 0.5 * math.sin(idx), 4),
                "success": episode % 4 != 3,
                "success_label_source": "human_reviewed",
                "object_pose_reference": pose_ref,
                "object_pose_estimate": pose_est,
                "object_pose_reference_source": "calibration_target",
                "object_pose_estimate_source": "vision_pose_estimator",
                "pose_validation_source": "vision_validated",
                "camera": {
                    "camera_name": "golden_mutation_camera",
                    "color_image": f"camera_data/color/{scenario}/ep{episode:03d}_{idx:04d}.png",
                    "depth_image": f"camera_data/depth/{scenario}/ep{episode:03d}_{idx:04d}.png",
                    "timestamp": timestamp,
                    "timestamp_source": "synchronized",
                },
                "notes": "Dynamic golden mutation baseline.",
            }
        )
        for key in ("frame_path", "image_path", "rgb_path", "color_image", "depth_image", "camera_frame", "camera_frame_path"):
            item.pop(key, None)
        cleaned.append(item)
    return cleaned


def _write_variant(variant_dir: Path, records: list[dict[str, Any]]) -> None:
    (variant_dir / "aligned").mkdir(parents=True, exist_ok=True)
    (variant_dir / "camera_data").mkdir(parents=True, exist_ok=True)
    (variant_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (variant_dir / "aligned" / "records.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    _write_camera_files_and_index(variant_dir, records)
    _write_calibration(variant_dir)


def _write_camera_files_and_index(variant_dir: Path, records: list[dict[str, Any]]) -> None:
    index_path = variant_dir / "camera_data" / "index.csv"
    with index_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "camera_name", "color_image", "depth_image"])
        writer.writeheader()
        for record in records:
            camera = record.get("camera")
            if not isinstance(camera, dict):
                continue
            color = str(camera.get("color_image") or "")
            depth = str(camera.get("depth_image") or "")
            if color:
                _write_png(variant_dir / color)
            if depth:
                _write_png(variant_dir / depth)
            writer.writerow(
                {
                    "timestamp": camera.get("timestamp", record.get("timestamp")),
                    "camera_name": camera.get("camera_name", "camera"),
                    "color_image": color,
                    "depth_image": depth,
                }
            )


def _write_calibration(variant_dir: Path) -> None:
    payload = {
        "schema": "agentic_sim2real.golden_mutation_calibration.v1",
        "camera_name": "golden_mutation_camera",
        "robot_base_frame": "base",
        "camera_frame": "camera_color_optical_frame",
        "T_base_camera": [0.42, -0.31, 0.72, 0.0, 0.0, 0.7071068, 0.7071068],
        "quality": "synthetic_mutation_fixture",
    }
    (variant_dir / "calibration" / "calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_1X1)


def _drop_all_camera_links(records: list[dict[str, Any]], variant_dir: Path) -> None:
    for record in records:
        record.pop("camera", None)


def _drop_heldout_camera_links(records: list[dict[str, Any]], variant_dir: Path) -> None:
    for record in records:
        if str(record.get("split", "")).lower() in {"heldout", "holdout", "test", "validation", "val"}:
            record.pop("camera", None)


def _add_orphan_sbl_frames(records: list[dict[str, Any]], variant_dir: Path) -> None:
    orphan_dir = variant_dir / "frames_full" / "SBL"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        _write_png(orphan_dir / f"orphan_{idx:04d}.png")


def _lower_sample_rate(records: list[dict[str, Any]], variant_dir: Path) -> None:
    by_episode: dict[int, int] = {}
    for record in records:
        episode = int(record.get("episode_index", 0) or 0)
        idx = by_episode.get(episode, 0)
        by_episode[episode] = idx + 1
        record["timestamp"] = round(episode * 10.0 + idx * 0.1, 4)
        camera = record.get("camera")
        if isinstance(camera, dict):
            camera["timestamp"] = record["timestamp"]


def _make_auto_positive_labels(records: list[dict[str, Any]], variant_dir: Path) -> None:
    for record in records:
        record["success"] = True
        record["success_label_source"] = "tracking_error_threshold"
        record["tracking_error_rad"] = 0.0004


def _make_fk_proxy_pose(records: list[dict[str, Any]], variant_dir: Path) -> None:
    for record in records:
        pose = list(record.get("object_pose_reference") or _pose_for_episode(int(record.get("episode_index", 0) or 0)))
        record["object_pose_reference"] = pose
        record["object_pose_estimate"] = pose
        record["object_pose_reference_source"] = "fk_proxy"
        record["object_pose_estimate_source"] = "fk_proxy"
        record["pose_validation_source"] = "fk_proxy_only"


def _pose_for_episode(episode: int) -> list[float]:
    return [round(0.55 + 0.005 * episode, 6), round(-0.12 + 0.003 * episode, 6), 0.17, 0.0, 0.0, 0.0, 1.0]


def _vector_like(values: Any, scale: float) -> list[float]:
    if isinstance(values, list) and values:
        return [round(scale * math.cos(idx), 6) for idx, _ in enumerate(values)]
    return [round(scale * math.cos(idx), 6) for idx in range(6)]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "scenario"
