from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
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
    followup_keywords: tuple[str, ...] = ()
    requires_followup: bool = False
    policy_keywords: tuple[str, ...] = ()


class SupportFlowEnvironment:
    def __init__(self) -> None:
        self.tasks = self._build_tasks()
        self.max_steps = 6
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
            "response_quality_good": False,
            "tone_safe": True,
        }
        self.quality_journal: list[dict[str, float]] = []

    @staticmethod
    def _build_tasks() -> dict[str, TaskScenario]:
        return {
            "easy_billing_duplicate_charge": TaskScenario(
                task_id="easy_billing_duplicate_charge",
                title="Billing category + basic resolution",
                difficulty=Difficulty.EASY,
                description="Classify billing complaint and resolve incorrect charge.",
                issue_type="billing",
                initial_customer_message=(
                    "I was charged twice for my monthly plan. Can you fix this?"
                ),
                required_keywords=("refund", "duplicate", "charge"),
            ),
            "easy_refund_window_policy": TaskScenario(
                task_id="easy_refund_window_policy",
                title="Refund eligibility explanation",
                difficulty=Difficulty.EASY,
                description="Classify refund request and explain policy eligibility clearly.",
                issue_type="refund",
                initial_customer_message=(
                    "I canceled after 35 days. Can I still get a full refund?"
                ),
                required_keywords=("refund", "eligibility", "timeline"),
                policy_keywords=("policy", "window", "eligible", "timeline"),
            ),
            "medium_technical_login_loop": TaskScenario(
                task_id="medium_technical_login_loop",
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
            "medium_billing_invoice_mismatch": TaskScenario(
                task_id="medium_billing_invoice_mismatch",
                title="Invoice mismatch with tax confusion",
                difficulty=Difficulty.MEDIUM,
                description=(
                    "Resolve billing mismatch by explaining invoice components and next actions."
                ),
                issue_type="billing",
                initial_customer_message=(
                    "My invoice tax looks wrong and total doesn't match checkout amount."
                ),
                required_keywords=("invoice", "tax", "adjustment"),
            ),
            "hard_refund_damaged_multistep": TaskScenario(
                task_id="hard_refund_damaged_multistep",
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
                followup_keywords=("order id", "photo", "attachment", "proof"),
                requires_followup=True,
            ),
            "hard_technical_outage_escalation": TaskScenario(
                task_id="hard_technical_outage_escalation",
                title="Service outage with escalation workflow",
                difficulty=Difficulty.HARD,
                description=(
                    "Confirm incident details, provide mitigation, then escalate and close carefully."
                ),
                issue_type="technical",
                initial_customer_message=(
                    "Our dashboard has been down for 40 minutes and we are losing sales."
                ),
                required_keywords=("incident", "mitigation", "escalate"),
                followup_keywords=("region", "account id", "screenshot", "impact"),
                requires_followup=True,
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
            "response_quality_good": False,
            "tone_safe": True,
        }
        self.quality_journal = []
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
            normalized_label = action.classify_issue.strip().lower()
            if normalized_label not in {"billing", "technical", "refund"}:
                reward -= 0.08
                info["events"].append("invalid_classification_label")
            elif normalized_label == self.current_task.issue_type:
                if not self.signals["classification_correct"]:
                    reward += 0.18
                    self.signals["classification_correct"] = True
                    info["events"].append("correct_classification")
            else:
                reward -= 0.12
                info["events"].append("wrong_classification")

        quality = self._score_message_quality(action.message)
        if action.message.strip():
            self.quality_journal.append(quality)
        if quality["tone_safety"] < 1.0:
            self.signals["tone_safe"] = False
            reward -= 0.18
            info["events"].append("unsafe_tone_penalty")

        if action.ask_followup:
            if self.current_task.requires_followup:
                if not self.signals["followup_done"]:
                    has_followup_signal = (
                        "?" in action.message
                        or any(
                            word in action.message.lower()
                            for word in self.current_task.followup_keywords
                        )
                    )
                    if has_followup_signal:
                        reward += 0.17
                        self.signals["followup_done"] = True
                        info["events"].append("useful_followup")
                    else:
                        reward += 0.05
                        info["events"].append("weak_followup")
                else:
                    reward -= 0.03
                    info["events"].append("repeated_followup")
            else:
                reward -= 0.02
                info["events"].append("unnecessary_followup")

        if action.propose_resolution:
            if quality["keyword_coverage"] >= 0.34 and quality["actionability"] >= 0.5:
                reward += min(0.36, 0.18 + (0.14 * quality["keyword_coverage"]))
                self.signals["resolution_proposed"] = True
                info["events"].append("resolution_attempt")
            elif quality["keyword_coverage"] > 0:
                reward += 0.06
                self.signals["resolution_proposed"] = True
                info["events"].append("partial_resolution_attempt")
            else:
                reward -= 0.05
                info["events"].append("weak_resolution")

        if quality["overall"] >= 0.65 and quality["empathy"] >= 0.4:
            self.signals["response_quality_good"] = True
            reward += 0.06
            info["events"].append("quality_bonus")

        can_close = self.signals["resolution_proposed"]
        if self.current_task.requires_followup:
            can_close = can_close and self.signals["followup_done"]

        if action.close_ticket:
            if can_close:
                reward += 0.18
                self.signals["ticket_closed"] = True
                self.done = True
                info["events"].append("ticket_closed")
            else:
                reward -= 0.2
                info["events"].append("premature_close")

        if action.message.strip():
            self.history.append(f"agent: {action.message.strip()}")

        if self.step_count >= self.max_steps:
            self.done = True
            info["events"].append("max_steps_reached")

        reward = max(-1.0, min(1.0, reward))
        self.cumulative_reward = max(0.0, min(1.0, self.cumulative_reward + reward))
        info["quality"] = quality
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
                "response_quality": 0.0,
                "empathy": 0.0,
                "actionability": 0.0,
                "policy": 0.0,
                "safety": 0.0,
                "efficiency": 0.0,
            }

        classification = 1.0 if self.signals["classification_correct"] else 0.0
        followup_weight = 1.0 if self.current_task.requires_followup else 0.0
        followup = 1.0 if self.signals["followup_done"] else 0.0
        resolution = 1.0 if self.signals["resolution_proposed"] else 0.0
        closure = 1.0 if self.signals["ticket_closed"] else 0.0

        if self.quality_journal:
            response_quality = mean(item["overall"] for item in self.quality_journal)
            empathy = mean(item["empathy"] for item in self.quality_journal)
            actionability = mean(item["actionability"] for item in self.quality_journal)
            policy = mean(item["policy"] for item in self.quality_journal)
            safety = min(item["tone_safety"] for item in self.quality_journal)
        else:
            response_quality = 0.0
            empathy = 0.0
            actionability = 0.0
            policy = 0.0
            safety = 1.0

        if closure:
            efficiency = max(0.0, 1 - ((self.step_count - 1) / max(1, self.max_steps - 1)))
        else:
            efficiency = 0.2 if resolution else 0.0

        if self.current_task.difficulty == Difficulty.EASY:
            score = (
                0.30 * classification
                + 0.25 * resolution
                + 0.15 * closure
                + 0.15 * response_quality
                + 0.10 * policy
                + 0.05 * safety
            )
        elif self.current_task.difficulty == Difficulty.MEDIUM:
            score = (
                0.20 * classification
                + 0.25 * resolution
                + 0.15 * closure
                + 0.15 * response_quality
                + 0.10 * empathy
                + 0.10 * actionability
                + 0.05 * safety
            )
        else:
            score = (
                0.15 * classification
                + 0.15 * followup
                + 0.20 * resolution
                + 0.15 * closure
                + 0.15 * response_quality
                + 0.10 * policy
                + 0.10 * safety
            )

        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "classification": classification,
            "followup": followup if followup_weight else 0.0,
            "resolution": resolution,
            "closure": closure,
            "response_quality": round(response_quality, 3),
            "empathy": round(empathy, 3),
            "actionability": round(actionability, 3),
            "policy": round(policy, 3),
            "safety": round(safety, 3),
            "efficiency": round(efficiency, 3),
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

    def _score_message_quality(self, message: str) -> dict[str, float]:
        if self.current_task is None:
            return {
                "keyword_coverage": 0.0,
                "empathy": 0.0,
                "actionability": 0.0,
                "policy": 0.0,
                "tone_safety": 1.0,
                "overall": 0.0,
            }

        normalized = message.lower().strip()
        if not normalized:
            return {
                "keyword_coverage": 0.0,
                "empathy": 0.0,
                "actionability": 0.0,
                "policy": 0.0,
                "tone_safety": 1.0,
                "overall": 0.0,
            }

        empathy_tokens = ("sorry", "understand", "apolog", "thanks", "appreciate")
        action_tokens = (
            "please",
            "step",
            "check",
            "update",
            "reset",
            "refund",
            "invoice",
            "escalate",
            "confirm",
        )
        harmful_tokens = (
            "not my problem",
            "can't help",
            "whatever",
            "stupid",
            "useless",
        )

        keyword_hits = sum(1 for token in self.current_task.required_keywords if token in normalized)
        keyword_coverage = keyword_hits / max(1, len(self.current_task.required_keywords))

        empathy_hits = sum(1 for token in empathy_tokens if token in normalized)
        empathy = min(1.0, empathy_hits / 2)

        action_hits = sum(1 for token in action_tokens if token in normalized)
        actionability = min(1.0, action_hits / 3)

        harmful_hit = any(token in normalized for token in harmful_tokens)
        tone_safety = 0.0 if harmful_hit else 1.0

        if self.current_task.policy_keywords:
            policy_hits = sum(1 for token in self.current_task.policy_keywords if token in normalized)
            policy = min(1.0, policy_hits / max(1, len(self.current_task.policy_keywords) / 2))
        else:
            policy = 1.0

        overall = (
            0.35 * keyword_coverage
            + 0.20 * empathy
            + 0.25 * actionability
            + 0.10 * policy
            + 0.10 * tone_safety
        )

        return {
            "keyword_coverage": round(keyword_coverage, 3),
            "empathy": round(empathy, 3),
            "actionability": round(actionability, 3),
            "policy": round(policy, 3),
            "tone_safety": round(tone_safety, 3),
            "overall": round(overall, 3),
        }
