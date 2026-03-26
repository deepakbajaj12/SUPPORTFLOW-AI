from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.models import Action, Difficulty, Observation, TaskInfo


@dataclass(frozen=True)
class TaskScenario:
    task_id: str
    title: str
    difficulty: Difficulty
    description: str
    issue_type: str
    initial_customer_message: str
    required_keywords: tuple[str, ...]


class SupportFlowEnvironment:
    def __init__(self) -> None:
        self.tasks = self._build_tasks()
        self.max_steps = 5
        self.current_task: TaskScenario | None = None
        self.episode_id: str | None = None
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.history: list[str] = []
        self.signals = {
            "classification_correct": False,
            "followup_done": False,
            "resolution_proposed": False,
            "ticket_closed": False,
        }

    @staticmethod
    def _build_tasks() -> dict[str, TaskScenario]:
        return {
            "easy_billing_classification": TaskScenario(
                task_id="easy_billing_classification",
                title="Billing category + basic resolution",
                difficulty=Difficulty.EASY,
                description="Classify billing complaint and resolve incorrect charge.",
                issue_type="billing",
                initial_customer_message=(
                    "I was charged twice for my monthly plan. Can you fix this?"
                ),
                required_keywords=("refund", "duplicate", "charge"),
            ),
            "medium_technical_response": TaskScenario(
                task_id="medium_technical_response",
                title="Technical issue response",
                difficulty=Difficulty.MEDIUM,
                description=(
                    "Handle account access issue with a useful troubleshooting response."
                ),
                issue_type="technical",
                initial_customer_message=(
                    "I cannot log in after resetting my password. It says token expired."
                ),
                required_keywords=("password", "reset", "token"),
            ),
            "hard_refund_multistep": TaskScenario(
                task_id="hard_refund_multistep",
                title="Multi-step refund resolution",
                difficulty=Difficulty.HARD,
                description=(
                    "Ask follow-up, provide refund path, then close ticket in sequence."
                ),
                issue_type="refund",
                initial_customer_message=(
                    "My order arrived damaged. I need a refund but I do not know the process."
                ),
                required_keywords=("refund", "damaged", "order"),
            ),
        }

    def reset(self, task_id: str | None = None) -> Observation:
        selected_task_id = task_id or next(iter(self.tasks))
        if selected_task_id not in self.tasks:
            raise ValueError(f"Unknown task_id: {selected_task_id}")

        self.current_task = self.tasks[selected_task_id]
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.history = [f"customer: {self.current_task.initial_customer_message}"]
        self.signals = {
            "classification_correct": False,
            "followup_done": False,
            "resolution_proposed": False,
            "ticket_closed": False,
        }
        return self._observation()

    def state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_id": self.current_task.task_id if self.current_task else None,
            "step_count": self.step_count,
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "signals": dict(self.signals),
        }

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            return self._observation(), 0.0, True, {"message": "Episode already finished."}

        self.step_count += 1
        reward = 0.0
        info = {
            "step": self.step_count,
            "events": [],
        }

        if action.classify_issue:
            if action.classify_issue.strip().lower() == self.current_task.issue_type:
                if not self.signals["classification_correct"]:
                    reward += 0.25
                    self.signals["classification_correct"] = True
                    info["events"].append("correct_classification")
            else:
                reward -= 0.1
                info["events"].append("wrong_classification")

        if action.ask_followup:
            if self.current_task.difficulty == Difficulty.HARD:
                if not self.signals["followup_done"]:
                    reward += 0.2
                    self.signals["followup_done"] = True
                    info["events"].append("useful_followup")
            else:
                reward += 0.05
                info["events"].append("followup_not_required")

        if action.propose_resolution:
            message_lower = action.message.lower()
            matched = sum(1 for word in self.current_task.required_keywords if word in message_lower)
            if matched > 0:
                reward += min(0.35, 0.12 * matched)
                self.signals["resolution_proposed"] = True
                info["events"].append("resolution_attempt")

        can_close = self.signals["resolution_proposed"]
        if self.current_task.difficulty == Difficulty.HARD:
            can_close = can_close and self.signals["followup_done"]

        if action.close_ticket:
            if can_close:
                reward += 0.2
                self.signals["ticket_closed"] = True
                self.done = True
                info["events"].append("ticket_closed")
            else:
                reward -= 0.15
                info["events"].append("premature_close")

        if action.message.strip():
            self.history.append(f"agent: {action.message.strip()}")

        if self.step_count >= self.max_steps:
            self.done = True
            info["events"].append("max_steps_reached")

        reward = max(-1.0, min(1.0, reward))
        self.cumulative_reward = max(0.0, min(1.0, self.cumulative_reward + reward))
        info["grader_preview"] = self.grader()

        return self._observation(), round(reward, 3), self.done, info

    def grader(self) -> dict[str, float]:
        if self.current_task is None:
            return {
                "score": 0.0,
                "classification": 0.0,
                "followup": 0.0,
                "resolution": 0.0,
                "closure": 0.0,
            }

        classification = 1.0 if self.signals["classification_correct"] else 0.0
        followup_weight = 1.0 if self.current_task.difficulty == Difficulty.HARD else 0.0
        followup = 1.0 if self.signals["followup_done"] else 0.0
        resolution = 1.0 if self.signals["resolution_proposed"] else 0.0
        closure = 1.0 if self.signals["ticket_closed"] else 0.0

        if self.current_task.difficulty == Difficulty.EASY:
            score = 0.4 * classification + 0.4 * resolution + 0.2 * closure
        elif self.current_task.difficulty == Difficulty.MEDIUM:
            score = 0.3 * classification + 0.5 * resolution + 0.2 * closure
        else:
            score = 0.2 * classification + 0.2 * followup + 0.4 * resolution + 0.2 * closure

        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "classification": classification,
            "followup": followup if followup_weight else 0.0,
            "resolution": resolution,
            "closure": closure,
        }

    def list_tasks(self) -> list[TaskInfo]:
        schema = {
            "classify_issue": "string | null (billing|technical|refund)",
            "message": "string",
            "ask_followup": "boolean",
            "propose_resolution": "boolean",
            "close_ticket": "boolean",
        }
        return [
            TaskInfo(
                task_id=task.task_id,
                title=task.title,
                difficulty=task.difficulty,
                description=task.description,
                initial_customer_message=task.initial_customer_message,
                action_schema=schema,
            )
            for task in self.tasks.values()
        ]

    def _observation(self) -> Observation:
        if self.current_task is None:
            raise RuntimeError("Environment has not been reset yet.")

        return Observation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            customer_message=self.current_task.initial_customer_message,
            ticket_history=list(self.history),
            available_actions={
                "classify_issue": "billing|technical|refund",
                "message": "free text response",
                "ask_followup": "true|false",
                "propose_resolution": "true|false",
                "close_ticket": "true|false",
            },
        )
