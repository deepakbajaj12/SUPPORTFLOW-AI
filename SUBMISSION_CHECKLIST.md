# Submission Checklist (Round 1)

## 1. Local environment

- [ ] Use Python `3.10+` (recommended `3.11`)
- [ ] Create and activate virtual environment
- [ ] Install dependencies

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2. Baseline reproducibility

- [ ] Baseline script runs without error
- [ ] Output contains all 3 tasks
- [ ] Scores are within `0.0-1.0`

```bash
python scripts/run_baseline.py
```

## 3. Local API validation

Start server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

In another terminal run:

```bash
python scripts/smoke_test.py
```

- [ ] `/reset` works
- [ ] `/step` works
- [ ] `/state` works
- [ ] `/tasks` returns at least 3 tasks
- [ ] `/grader` score is within `0.0-1.0`
- [ ] `/baseline` returns all tasks and average score

## 4. Docker validation

- [ ] Image builds successfully
- [ ] Container starts and serves API

```bash
docker build -t supportflow-ai .
docker run -p 7860:7860 supportflow-ai
```

## 5. Hugging Face Space validation

- [ ] `huggingface-cli login` completed
- [ ] Repo pushed to Docker Space
- [ ] Space URL returns HTTP `200`
- [ ] `POST /reset` responds correctly
- [ ] `GET /baseline` works

## 6. Submission package sanity

- [ ] `openenv.yaml` present
- [ ] `Dockerfile` present
- [ ] `README.md` has setup + endpoints + action/observation info
- [ ] `app/` includes typed models and `step/reset/state` implementation
- [ ] No local-only secrets committed

## 7. Final pre-submit command set

```bash
git status --short --branch
python scripts/run_baseline.py
python scripts/smoke_test.py
```

Only submit when all checks pass.
