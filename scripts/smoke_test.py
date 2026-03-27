from __future__ import annotations

import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:7860"
LOG_DIR = PROJECT_ROOT / "logs"


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

        # Negative case: step called before reset should fail only when no active task exists.
        pre_state = _request("GET", "/state")
        if pre_state.get("task_id") is None:
            step_before_reset_failed = False
            try:
                _request(
                    "POST",
                    "/step",
                    {
                        "classify_issue": "billing",
                        "message": "test",
                        "ask_followup": False,
                        "propose_resolution": False,
                        "close_ticket": False,
                    },
                )
            except urllib.error.HTTPError as exc:
                step_before_reset_failed = exc.code == 400
            checks.append(("step before reset rejected", step_before_reset_failed, "expects HTTP 400"))
        else:
            checks.append(
                (
                    "step before reset rejected",
                    True,
                    f"skipped: server already had active task {pre_state.get('task_id')}",
                )
            )

        # Validate explicit error handling for invalid task id.
        invalid_failed_as_expected = False
        try:
            _request("POST", "/reset", {"task_id": "invalid_task"})
        except urllib.error.HTTPError as exc:
            invalid_failed_as_expected = exc.code == 400
        checks.append(("invalid task rejection", invalid_failed_as_expected, "expects HTTP 400"))

        # Validate reset works even with no JSON body.
        reset_default = _request("POST", "/reset")
        checks.append(
            (
                "reset empty body",
                bool(reset_default.get("observation", {}).get("task_id")),
                json.dumps(reset_default),
            )
        )

        reset = _request("POST", "/reset", {"task_id": "hard_refund_damaged_multistep"})
        checks.append(
            (
                "reset endpoint",
                reset.get("observation", {}).get("task_id") == "hard_refund_damaged_multistep",
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

        # Negative case: episode already done should return done=true and reward=0.
        step_after_done = _request(
            "POST",
            "/step",
            {
                "classify_issue": "refund",
                "message": "extra message",
                "ask_followup": False,
                "propose_resolution": False,
                "close_ticket": False,
            },
        )
        checks.append(
            (
                "step after done behavior",
                step_after_done.get("done") is True and float(step_after_done.get("reward", 1)) == 0.0,
                json.dumps(step_after_done),
            )
        )

        # Negative case: premature close should be penalized.
        _request("POST", "/reset", {"task_id": "hard_technical_outage_escalation"})
        premature = _request(
            "POST",
            "/step",
            {
                "classify_issue": "technical",
                "message": "Closing immediately without details.",
                "ask_followup": False,
                "propose_resolution": False,
                "close_ticket": True,
            },
        )
        checks.append(
            (
                "premature close penalty",
                float(premature.get("reward", 0)) < 0,
                json.dumps(premature),
            )
        )

        baseline = _request("GET", "/baseline")
        baseline_avg = baseline.get("average_score", -1)
        baseline_ok = isinstance(baseline_avg, (int, float)) and 0.0 <= baseline_avg <= 1.0
        checks.append(("baseline endpoint", baseline_ok, str(baseline_avg)))
        baseline_rows = baseline.get("results", [])
        has_breakdown = bool(baseline_rows) and isinstance(baseline_rows[0].get("breakdown"), dict)
        has_failure_score = bool(baseline_rows) and isinstance(
            baseline_rows[0].get("failure_case_score"),
            (int, float),
        )
        checks.append(("baseline breakdown field", has_breakdown, json.dumps(baseline_rows[0] if baseline_rows else {})))
        checks.append(("baseline failure analysis", has_failure_score, json.dumps(baseline_rows[0] if baseline_rows else {})))

    except Exception as exc:  # pragma: no cover
        print(f"FATAL: smoke test execution failed: {exc}")
        return 2

    failed = [item for item in checks if not item[1]]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} :: {detail}")

    if failed:
        print(f"\nSmoke test failed: {len(failed)} check(s) did not pass.")
        _write_validator_log(checks)
        return 1

    print("\nSmoke test passed: all checks succeeded.")
    _write_validator_log(checks)
    return 0


def _write_validator_log(checks: list[tuple[str, bool, str]]) -> None:
    target_dir = LOG_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        target_dir = Path(tempfile.gettempdir()) / "supportflow_logs"
        target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp_utc": timestamp,
        "base_url": BASE_URL,
        "log_directory": str(target_dir),
        "checks": [
            {
                "name": name,
                "passed": passed,
                "detail": detail,
            }
            for name, passed, detail in checks
        ],
        "summary": {
            "total": len(checks),
            "passed": sum(1 for _, passed, _ in checks if passed),
            "failed": sum(1 for _, passed, _ in checks if not passed),
        },
    }
    latest_path = target_dir / "validator_latest.json"
    timestamped_path = target_dir / f"validator_{timestamp}.json"
    try:
        latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        timestamped_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "supportflow_logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        payload["log_directory"] = str(fallback_dir)
        (fallback_dir / "validator_latest.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        (fallback_dir / f"validator_{timestamp}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    sys.exit(main())
