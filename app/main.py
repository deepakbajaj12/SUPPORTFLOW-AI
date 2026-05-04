from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from app.baseline import run_baseline
from app.environment import SupportFlowEnvironment
from app.models import (
    Action,
    BaselineResponse,
    GraderResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
    TasksResponse,
    InfoResponse,
)

app = FastAPI(title="SupportFlow AI OpenEnv", version="0.1.0")
env = SupportFlowEnvironment()


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "SupportFlow AI", "status": "ready"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        name="SupportFlow AI",
        version="0.1.0",
        description="A real-world customer support resolution environment for OpenEnv."
    )


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = Body(default=ResetRequest())) -> ResetResponse:
    try:
        observation = env.reset(request.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    snapshot = env.state()
    return StateResponse(**snapshot)


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return TasksResponse(tasks=env.list_tasks())


@app.get("/grader", response_model=GraderResponse)
def grader() -> GraderResponse:
    details = env.grader()
    snapshot = env.state()
    return GraderResponse(
        task_id=snapshot["task_id"],
        score=details["score"],
        breakdown={key: float(value) for key, value in details.items() if key != "score"},
    )


@app.get("/baseline", response_model=BaselineResponse)
@app.post("/baseline", response_model=BaselineResponse)
def baseline() -> BaselineResponse:
    results = run_baseline()
    avg = sum(item.score for item in results) / len(results)
    return BaselineResponse(average_score=round(avg, 3), results=results)
