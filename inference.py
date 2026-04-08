from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from app.environment import SupportFlowEnvironment
from app.models import Action


HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("SUPPORTFLOW_BENCHMARK", "supportflow-ai")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))


SYSTEM_PROMPT = (
    "You are an expert customer support agent. Return ONLY valid JSON for the next action "
    "with fields: classify_issue (billing|technical|refund|null), message (string), "
    "ask_followup (bool), propose_resolution (bool), close_ticket (bool)."
)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _reward_str(value: float) -> str:
    return f"{value:.2f}"


def _format_action(action: Action) -> str:
    payload = {
        "classify_issue": action.classify_issue,
        "message": action.message,
        "ask_followup": action.ask_followup,
        "propose_resolution": action.propose_resolution,
        "close_ticket": action.close_ticket,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _make_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _llm_action(client: OpenAI, task_id: str, history: list[str]) -> Action:
    user_prompt = (
        f"Task: {task_id}\\n"
        "Conversation history:\\n"
        + "\\n".join(history[-8:])
        + "\\nReturn JSON only."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = _extract_json_object(content)

    classify_issue = payload.get("classify_issue")
    if isinstance(classify_issue, str):
        classify_issue = classify_issue.strip().lower()
        if classify_issue == "null":
            classify_issue = None
    elif classify_issue is not None:
        classify_issue = None

    return Action(
        classify_issue=classify_issue,
        message=str(payload.get("message") or ""),
        ask_followup=bool(payload.get("ask_followup", False)),
        propose_resolution=bool(payload.get("propose_resolution", False)),
        close_ticket=bool(payload.get("close_ticket", False)),
    )


def run_episode(env: SupportFlowEnvironment, client: OpenAI, task_id: str) -> None:
    rewards: list[float] = []
    done = False
    steps = 0
    success = False
    last_error: str | None = None

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    try:
        env.reset(task_id)
        while steps < MAX_STEPS and not done:
            action = _llm_action(client, task_id, env.history)
            _, reward, done, _ = env.step(action)
            steps += 1
            rewards.append(reward)
            error_field = last_error if last_error is not None else "null"
            print(
                "[STEP] "
                f"step={steps} "
                f"action={_format_action(action)} "
                f"reward={_reward_str(reward)} "
                f"done={_bool_str(done)} "
                f"error={error_field}"
            )
            if done:
                grade = env.grader()
                success = bool(grade.get("score", 0.0) >= 0.7)
    except Exception as exc:
        last_error = str(exc)
    finally:
        if not success and last_error is None:
            grade = env.grader()
            success = bool(grade.get("score", 0.0) >= 0.7)
        rewards_str = ",".join(_reward_str(item) for item in rewards)
        print(
            "[END] "
            f"success={_bool_str(success)} "
            f"steps={steps} "
            f"rewards={rewards_str}"
        )


if __name__ == "__main__":
    environment = SupportFlowEnvironment()
    llm_client = _make_client()
    for task in environment.list_tasks():
        run_episode(environment, llm_client, task.task_id)
