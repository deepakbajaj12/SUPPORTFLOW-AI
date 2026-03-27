from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.baseline import run_baseline


if __name__ == "__main__":
    results = run_baseline()
    average_score = round(sum(r.score for r in results) / len(results), 3)
    failure_average = round(sum(r.failure_case_score for r in results) / len(results), 3)
    robustness_margin = round(average_score - failure_average, 3)
    payload = {
        "average_score": average_score,
        "failure_case_average": failure_average,
        "robustness_margin": robustness_margin,
        "results": [r.model_dump() for r in results],
    }
    print(json.dumps(payload, indent=2))
