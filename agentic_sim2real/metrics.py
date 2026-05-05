from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from .dataset import Record


def summarize_records(records: list[Record]) -> dict:
    episodes = sorted({r.episode_index for r in records})
    timestamps_by_episode: dict[int, list[float]] = defaultdict(list)
    successes: dict[int, bool] = {}
    failures: Counter[str] = Counter()
    contact_forces = []
    for rec in records:
        timestamps_by_episode[rec.episode_index].append(rec.timestamp)
        if rec.success is not None:
            successes[rec.episode_index] = rec.success
        if rec.failure_mode:
            failures[rec.failure_mode] += 1
        if rec.contact_force is not None:
            contact_forces.append(float(rec.contact_force))

    dts = []
    for values in timestamps_by_episode.values():
        values = sorted(values)
        dts.extend(b - a for a, b in zip(values, values[1:]) if b > a)

    median_dt = statistics.median(dts) if dts else None
    success_rate = None
    if successes:
        success_rate = sum(1 for ok in successes.values() if ok) / len(successes)

    return {
        "records": len(records),
        "episodes": len(episodes),
        "episode_indices": episodes,
        "action_dim": len(records[0].action),
        "joint_dim": len(records[0].joint_state),
        "median_dt_s": round(median_dt, 5) if median_dt else None,
        "estimated_rate_hz": round(1.0 / median_dt, 3) if median_dt else None,
        "success_labeled_episodes": len(successes),
        "success_rate": round(success_rate, 3) if success_rate is not None else None,
        "failure_modes": dict(failures),
        "contact_force_mean_n": round(statistics.mean(contact_forces), 3) if contact_forces else None,
        "contact_force_peak_n": round(max(contact_forces), 3) if contact_forces else None,
    }


def estimate_delay_steps(records: list[Record], max_lag: int = 8) -> dict:
    episodes = _group_by_episode(records)
    lag_scores: dict[int, list[float]] = defaultdict(list)
    for ep_records in episodes.values():
        if len(ep_records) < max_lag + 4:
            continue
        action_series = [r.action for r in ep_records]
        joint_delta = _deltas([r.joint_state for r in ep_records])
        dims = min(len(action_series[0]), len(joint_delta[0]) if joint_delta else 0)
        for lag in range(0, max_lag + 1):
            scores = []
            for dim in range(dims):
                xs = [a[dim] for a in action_series[:-1 - lag or None]]
                ys = [d[dim] for d in joint_delta[lag:]]
                n = min(len(xs), len(ys))
                if n >= 3:
                    scores.append(abs(_corr(xs[:n], ys[:n])))
            if scores:
                lag_scores[lag].append(statistics.mean(scores))

    if not lag_scores:
        return {"delay_steps": 0, "confidence": 0.0, "lag_scores": {}}

    mean_scores = {lag: statistics.mean(vals) for lag, vals in lag_scores.items()}
    best_lag = max(mean_scores, key=mean_scores.get)
    sorted_scores = sorted(mean_scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    confidence = max(0.0, min(1.0, margin * 2.0))
    return {
        "delay_steps": int(best_lag),
        "confidence": round(confidence, 3),
        "lag_scores": {str(k): round(v, 4) for k, v in sorted(mean_scores.items())},
    }


def estimate_deadband(records: list[Record]) -> dict:
    episodes = _group_by_episode(records)
    command_magnitudes = []
    moving_command_magnitudes = []
    swallowed = 0
    total = 0
    for ep_records in episodes.values():
        if len(ep_records) < 3:
            continue
        deltas = _deltas([r.joint_state for r in ep_records])
        actions = [r.action for r in ep_records[:-1]]
        move_norms = [_norm(d) for d in deltas]
        if not move_norms:
            continue
        movement_threshold = max(1e-6, statistics.median(move_norms) * 0.4)
        for action, move_norm in zip(actions, move_norms):
            command = _norm(action)
            command_magnitudes.append(command)
            total += 1
            if move_norm <= movement_threshold:
                swallowed += 1
            else:
                moving_command_magnitudes.append(command)

    if not command_magnitudes:
        return {"deadband_command_norm": 0.0, "swallowed_command_ratio": 0.0, "confidence": 0.0}

    if moving_command_magnitudes:
        deadband = _percentile(moving_command_magnitudes, 0.1)
    else:
        deadband = _percentile(command_magnitudes, 0.5)
    swallowed_ratio = swallowed / total if total else 0.0
    confidence = min(1.0, len(command_magnitudes) / 200.0)
    return {
        "deadband_command_norm": round(deadband, 6),
        "swallowed_command_ratio": round(swallowed_ratio, 4),
        "confidence": round(confidence, 3),
    }


def estimate_pose_noise(records: list[Record]) -> dict:
    pos_errors = []
    quat_errors_deg = []
    for rec in records:
        est = rec.shaft_pose_estimate
        ref = rec.shaft_pose_reference
        if len(est) >= 3 and len(ref) >= 3:
            pos_errors.append(_norm([a - b for a, b in zip(est[:3], ref[:3])]))
        if len(est) >= 7 and len(ref) >= 7:
            quat_errors_deg.append(_quat_angle_error_deg(est[3:7], ref[3:7]))

    return {
        "samples": len(pos_errors),
        "position_error_mean_m": round(statistics.mean(pos_errors), 6) if pos_errors else None,
        "position_error_p95_m": round(_percentile(pos_errors, 0.95), 6) if pos_errors else None,
        "orientation_error_mean_deg": round(statistics.mean(quat_errors_deg), 3) if quat_errors_deg else None,
        "orientation_error_p95_deg": round(_percentile(quat_errors_deg, 0.95), 3) if quat_errors_deg else None,
    }


def estimate_contact(records: list[Record], force_limit_n: float) -> dict:
    forces = [float(r.contact_force) for r in records if r.contact_force is not None]
    if not forces:
        return {
            "samples": 0,
            "mean_n": None,
            "peak_n": None,
            "over_limit_ratio": 0.0,
            "force_limit_n": force_limit_n,
        }
    over = sum(1 for force in forces if force > force_limit_n)
    return {
        "samples": len(forces),
        "mean_n": round(statistics.mean(forces), 3),
        "peak_n": round(max(forces), 3),
        "p95_n": round(_percentile(forces, 0.95), 3),
        "over_limit_ratio": round(over / len(forces), 4),
        "force_limit_n": force_limit_n,
    }


def estimate_reset_scatter(records: list[Record]) -> dict:
    by_episode = _group_by_episode(records)
    starts = []
    for ep_records in by_episode.values():
        first = ep_records[0]
        pose = first.shaft_pose_reference or first.shaft_pose_estimate
        if len(pose) >= 3:
            starts.append(pose[:3])
    if len(starts) < 2:
        return {"samples": len(starts), "x_range_m": None, "y_range_m": None, "z_range_m": None}
    xs = [p[0] for p in starts]
    ys = [p[1] for p in starts]
    zs = [p[2] for p in starts]
    return {
        "samples": len(starts),
        "x_range_m": round(max(xs) - min(xs), 5),
        "y_range_m": round(max(ys) - min(ys), 5),
        "z_range_m": round(max(zs) - min(zs), 5),
    }


def _group_by_episode(records: list[Record]) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = defaultdict(list)
    for rec in records:
        grouped[rec.episode_index].append(rec)
    for values in grouped.values():
        values.sort(key=lambda r: r.timestamp)
    return grouped


def _deltas(vectors: list[list[float]]) -> list[list[float]]:
    return [
        [b_i - a_i for a_i, b_i in zip(a, b)]
        for a, b in zip(vectors, vectors[1:])
    ]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    return num / den if den > 1e-12 else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
    return values[idx]


def _quat_angle_error_deg(q_a: list[float], q_b: list[float]) -> float:
    a = _normalize_quat(q_a)
    b = _normalize_quat(q_b)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _normalize_quat(q: list[float]) -> list[float]:
    norm = _norm(q)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / norm for v in q]
