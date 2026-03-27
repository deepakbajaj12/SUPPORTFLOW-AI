from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:7860"


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        body = res.read().decode("utf-8")
        return json.loads(body) if body else {}


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    try:
        health = _request("GET", "/health")
        checks.append(("health endpoint", health.get("status") == "ok", str(health)))

        tasks = _request("GET", "/tasks")
        task_list = tasks.get("tasks", [])
        checks.append(("tasks endpoint", len(task_list) >= 3, f"count={len(task_list)}"))

        # Validate explicit error handling for invalid task id.
        invalid_failed_as_expected = False
        try:
            _request("POST", "/reset", {"task_id": "invalid_task"})
        except urllib.error.HTTPError as exc:
            invalid_failed_as_expected = exc.code == 400
        checks.append(("invalid task rejection", invalid_failed_as_expected, "expects HTTP 400"))

        reset = _request("POST", "/reset", {"task_id": "hard_refund_multistep"})
        checks.append(
            (
                "reset endpoint",
                reset.get("observation", {}).get("task_id") == "hard_refund_multistep",
                json.dumps(reset),
            )
        )

        step_1 = _request(
            "POST",
            "/step",
            {
                "classify_issue": "refund",
                "message": "Please share your order ID and a photo of the damaged package.",
                "ask_followup": True,
                "propose_resolution": False,
                "close_ticket": False,
            },
        )
        checks.append(("step endpoint #1", isinstance(step_1.get("reward"), (int, float)), str(step_1.get("reward"))))

        step_2 = _request(
            "POST",
            "/step",
            {
                "classify_issue": "refund",
                "message": "Refund approved for the damaged order.",
                "ask_followup": False,
                "propose_resolution": True,
                "close_ticket": True,
            },
        )
        checks.append(("step endpoint #2 done", step_2.get("done") is True, json.dumps(step_2)))

        state = _request("GET", "/state")
        checks.append(("state endpoint", state.get("done") is True, json.dumps(state)))

        grader = _request("GET", "/grader")
        grader_score = grader.get("score", -1)
        checks.append(("grader endpoint", isinstance(grader_score, (int, float)) and 0.0 <= grader_score <= 1.0, str(grader_score)))

        baseline = _request("GET", "/baseline")
        baseline_avg = baseline.get("average_score", -1)
        baseline_ok = isinstance(baseline_avg, (int, float)) and 0.0 <= baseline_avg <= 1.0
        checks.append(("baseline endpoint", baseline_ok, str(baseline_avg)))

    except Exception as exc:  # pragma: no cover
        print(f"FATAL: smoke test execution failed: {exc}")
        return 2

    failed = [item for item in checks if not item[1]]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} :: {detail}")

    if failed:
        print(f"\nSmoke test failed: {len(failed)} check(s) did not pass.")
        return 1

    print("\nSmoke test passed: all checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
