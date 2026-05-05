from __future__ import annotations

import os

from .config import PipelineConfig


def require_real_robot_gate(config: PipelineConfig) -> None:
    if not bool(config.safety.get("require_human_gate", True)):
        return
    key = str(config.safety.get("real_robot_gate_env", "I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK"))
    expected = str(config.safety.get("real_robot_gate_value", "yes"))
    actual = os.environ.get(key)
    if actual != expected:
        raise SystemExit(
            f"Real robot gate blocked. Set {key}={expected!r} only after a human has checked "
            "calibration, workspace, teach pendant state, gripper Tool I/O, and emergency stop."
        )
