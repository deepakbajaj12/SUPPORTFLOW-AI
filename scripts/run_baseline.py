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
    payload = {
        "average_score": round(sum(r.score for r in results) / len(results), 3),
        "results": [r.model_dump() for r in results],
    }
    print(json.dumps(payload, indent=2))
