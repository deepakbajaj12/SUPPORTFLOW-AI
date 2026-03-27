from __future__ import annotations

from app.environment import SupportFlowEnvironment
from app.models import Action, BaselineResult


def _baseline_policy(task_id: str) -> list[Action]:
    if task_id == "easy_billing_duplicate_charge":
        return [
            Action(
                classify_issue="billing",
                message=(
                    "Sorry for the duplicate charge. I confirmed it and started a refund "
                    "for the extra charge."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    if task_id == "easy_refund_window_policy":
        return [
            Action(
                classify_issue="refund",
                message=(
                    "Thanks for checking. Based on policy refund eligibility depends on "
                    "timeline window, and I can process a partial refund now."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    if task_id == "medium_technical_login_loop":
        return [
            Action(
                classify_issue="technical",
                message=(
                    "I understand the frustration. Please reset your password again and use "
                    "the newest token link. "
                    "If token errors continue, we will issue a manual reset."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    if task_id == "medium_billing_invoice_mismatch":
        return [
            Action(
                classify_issue="billing",
                message=(
                    "I checked the invoice and tax line-items. We will apply an adjustment "
                    "and send an updated invoice within one business day."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    if task_id == "hard_refund_damaged_multistep":
        return [
            Action(
                classify_issue="refund",
                message=(
                    "Sorry this arrived damaged. Please share order ID and a photo so I can "
                    "complete your refund."
                ),
                ask_followup=True,
                propose_resolution=False,
                close_ticket=False,
            ),
            Action(
                classify_issue="refund",
                message=(
                    "Thanks, I initiated the refund for the damaged order and escalated "
                    "priority handling."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            ),
        ]

    return [
        Action(
            classify_issue="technical",
            message=(
                "I understand the outage impact. Please share your region and account id so "
                "I can confirm incident scope."
            ),
            ask_followup=True,
            propose_resolution=False,
            close_ticket=False,
        ),
        Action(
            classify_issue="technical",
            message=(
                "Incident confirmed. We applied mitigation and escalated to on-call. "
                "I will keep this ticket open for updates."
            ),
            ask_followup=False,
            propose_resolution=True,
            close_ticket=False,
        ),
        Action(
            classify_issue="technical",
            message=(
                "Issue stabilized and monitoring is green. We escalated root cause review and "
                "can close this ticket."
            ),
            ask_followup=False,
            propose_resolution=True,
            close_ticket=True,
        ),
    ]


def _failure_policy(task_id: str) -> list[Action]:
    if task_id.startswith("easy_"):
        return [
            Action(
                classify_issue="technical",
                message="Whatever. Closing this now.",
                ask_followup=False,
                propose_resolution=False,
                close_ticket=True,
            )
        ]

    if task_id.startswith("medium_"):
        return [
            Action(
                classify_issue="refund",
                message="Not my problem. Try again later.",
                ask_followup=False,
                propose_resolution=False,
                close_ticket=True,
            )
        ]

    return [
        Action(
            classify_issue="billing",
            message="Closing ticket. Cannot help.",
            ask_followup=False,
            propose_resolution=False,
            close_ticket=True,
        )
    ]


def _run_episode(env: SupportFlowEnvironment, task_id: str, policy: list[Action]) -> tuple[dict, dict]:
    env.reset(task_id)
    for action in policy:
        _, _, done, _ = env.step(action)
        if done:
            break
    return env.grader(), env.state()


def run_baseline(env: SupportFlowEnvironment | None = None) -> list[BaselineResult]:
    runtime_env = env or SupportFlowEnvironment()
    results: list[BaselineResult] = []

    for task in runtime_env.list_tasks():
        grader, state = _run_episode(runtime_env, task.task_id, _baseline_policy(task.task_id))
        failure_grader, _ = _run_episode(runtime_env, task.task_id, _failure_policy(task.task_id))

        failure_reason = (
            "Intentional negative run with wrong classification, unsafe tone, and premature closure "
            "to measure robustness margin."
        )
        results.append(
            BaselineResult(
                task_id=task.task_id,
                score=grader["score"],
                steps=state["step_count"],
                breakdown={key: float(value) for key, value in grader.items() if key != "score"},
                failure_case_score=failure_grader["score"],
                failure_case_reason=failure_reason,
            )
        )

    return results
