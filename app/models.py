from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Action(BaseModel):
    classify_issue: str | None = Field(
        default=None,
        description="Issue category prediction: billing, technical, or refund.",
    )
    message: str = Field(
        default="",
        description="Agent response sent to the customer.",
    )
    ask_followup: bool = Field(
        default=False,
        description="Whether the agent asks for missing details before resolving.",
    )
    propose_resolution: bool = Field(
        default=False,
        description="Whether the message attempts to solve the issue.",
    )
    close_ticket: bool = Field(
        default=False,
        description="Whether the agent marks the ticket as closed.",
    )


class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    customer_message: str
    ticket_history: list[str]
    available_actions: dict[str, str]


class ResetRequest(BaseModel):
    task_id: str | None = None


class ResetResponse(BaseModel):
    observation: Observation


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    episode_id: str | None
    task_id: str | None
    step_count: int
    done: bool
    cumulative_reward: float
    signals: dict[str, bool]


class TaskInfo(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    description: str
    initial_customer_message: str
    action_schema: dict[str, str]


class TasksResponse(BaseModel):
    tasks: list[TaskInfo]


class GraderResponse(BaseModel):
    task_id: str | None
    score: float
    breakdown: dict[str, float]


class BaselineResult(BaseModel):
    task_id: str
    score: float
    steps: int
    breakdown: dict[str, float]
    failure_case_score: float
    failure_case_reason: str


class BaselineResponse(BaseModel):
    average_score: float
    results: list[BaselineResult]
