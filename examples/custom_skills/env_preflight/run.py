from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    input_path = Path(os.environ["UR_SKILL_INPUT_JSON"])
    output_path = Path(os.environ["UR_SKILL_OUTPUT_JSON"])
    skill_out = Path(os.environ["UR_SKILL_OUT_DIR"])
    payload = json.loads(input_path.read_text())

    evidence_path = skill_out / "custom_preflight.json"
    evidence_path.write_text(
        json.dumps(
            {
                "message": "This is an example external replacement skill.",
                "skill_id": payload["skill_id"],
                "dataset": payload["dataset"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "quality_score": 0.95,
                "confidence": 0.9,
                "blocking_failures": [],
                "warnings": ["example custom env_preflight override was used"],
                "evidence_files": [str(evidence_path)],
                "metrics": {"custom_skill_used": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
