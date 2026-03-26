from __future__ import annotations

from app.environment import SupportFlowEnvironment
from app.models import Action, BaselineResult


def _baseline_policy(task_id: str) -> list[Action]:
    if task_id == "easy_billing_classification":
        return [
            Action(
                classify_issue="billing",
                message=(
                    "I confirmed a duplicate charge and started a refund for the extra charge."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    if task_id == "medium_technical_response":
        return [
            Action(
                classify_issue="technical",
                message=(
                    "Please reset your password again and use the newest token link. "
                    "If token errors continue, we will issue a manual reset."
                ),
                ask_followup=False,
                propose_resolution=True,
                close_ticket=True,
            )
        ]

    return [
        Action(
            classify_issue="refund",
            message="Please share your order ID and a photo of the damaged package.",
            ask_followup=True,
            propose_resolution=False,
            close_ticket=False,
        ),
        Action(
            classify_issue="refund",
            message=(
                "Thanks. I initiated the refund for the damaged order and emailed confirmation."
            ),
            ask_followup=False,
            propose_resolution=True,
            close_ticket=True,
        ),
    ]


def run_baseline(env: SupportFlowEnvironment | None = None) -> list[BaselineResult]:
    runtime_env = env or SupportFlowEnvironment()
    results: list[BaselineResult] = []

    for task in runtime_env.list_tasks():
        runtime_env.reset(task.task_id)
        policy_steps = _baseline_policy(task.task_id)

        for action in policy_steps:
            _, _, done, _ = runtime_env.step(action)
            if done:
                break

        grader = runtime_env.grader()
        state = runtime_env.state()
        results.append(
            BaselineResult(
                task_id=task.task_id,
                score=grader["score"],
                steps=state["step_count"],
            )
        )

    return results
