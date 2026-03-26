# SupportFlow AI (OpenEnv Round 1 Project)

SupportFlow AI is a real-world customer support resolution environment for OpenEnv.
An AI agent must classify issues, respond appropriately, and resolve tickets using the standard API.

## Why this environment

- Realistic domain: customer support workflows used in SaaS and e-commerce teams.
- Incremental difficulty: easy -> medium -> hard tasks.
- Reward shaping: partial progress signals encourage useful intermediate behavior.

## Tasks

1. `easy_billing_classification` (easy)
- Goal: classify a billing issue and resolve duplicate charge complaint.

2. `medium_technical_response` (medium)
- Goal: classify a technical issue and provide a high-quality troubleshooting response.

3. `hard_refund_multistep` (hard)
- Goal: ask follow-up details, propose refund resolution, and close ticket in correct sequence.

All graders return scores in `[0.0, 1.0]`.

## Action Space

`POST /step` body:

```json
{
  "classify_issue": "billing | technical | refund | null",
  "message": "string",
  "ask_followup": false,
  "propose_resolution": false,
  "close_ticket": false
}
```

## Observation Space

`/reset` and `/step` responses include:

- `task_id`
- `difficulty`
- `customer_message`
- `ticket_history`
- `available_actions`

## API Endpoints

Required OpenEnv endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

Additional required endpoints:

- `GET /tasks`
- `GET /grader`
- `GET /baseline`
- `POST /baseline`

Utility endpoint:

- `GET /health`

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Test quickly:

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
curl http://localhost:7860/tasks
curl http://localhost:7860/baseline
```

## Baseline inference

Deterministic baseline script:

```bash
python scripts/run_baseline.py
```

Expected output format:

```json
{
  "average_score": 0.9,
  "results": [
    {"task_id": "easy_billing_classification", "score": 1.0, "steps": 1},
    {"task_id": "medium_technical_response", "score": 1.0, "steps": 1},
    {"task_id": "hard_refund_multistep", "score": 0.8, "steps": 2}
  ]
}
```

Exact scores may vary if reward logic is changed.

## Docker

Build and run:

```bash
docker build -t supportflow-ai .
docker run -p 7860:7860 supportflow-ai
```

## Hugging Face Spaces deployment

Use Docker Space and include this repository files as-is:

- `Dockerfile`
- `requirements.txt`
- `app/`
- `app.py`
- `openenv.yaml`

Set Space to expose port `7860`.

## File map

- `app/environment.py`: task definitions, reward logic, grader, state machine.
- `app/models.py`: typed request/response models.
- `app/main.py`: FastAPI endpoints.
- `app/baseline.py`: reproducible baseline policy.
- `scripts/run_baseline.py`: CLI script for baseline scoring.
- `openenv.yaml`: OpenEnv metadata and endpoint/model declarations.
