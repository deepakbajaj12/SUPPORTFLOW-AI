 # Submission Checklist (Round 1)

## 1. Local environment

- [x] Use Python `3.10+` (recommended `3.11`)
- [x] Create and activate virtual environment
- [x] Install dependencies

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2. Baseline reproducibility

- [x] Baseline script runs without error
- [x] Output contains all 6 tasks
- [x] Scores are within `0.0-1.0`
- [x] `failure_case_average` and `robustness_margin` are present

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
- [ ] `/tasks` returns at least 6 tasks
- [ ] `/grader` score is within `0.0-1.0`
- [ ] `/baseline` returns all tasks and average score
- [ ] negative checks pass (`step before reset`, `invalid task`, `premature close penalty`)
- [ ] validator logs generated in `logs/validator_latest.json`
- [x] `/reset` works
- [x] `/step` works
- [x] `/state` works
- [x] `/tasks` returns at least 6 tasks
- [x] `/grader` score is within `0.0-1.0`
- [x] `/baseline` returns all tasks and average score
- [x] negative checks pass (`step before reset`, `invalid task`, `premature close penalty`)
- [x] validator logs generated in `logs/validator_latest.json`

## 4. Docker validation

- [x] Image builds successfully
- [x] Container starts and serves API

```bash
docker build -t supportflow-ai .
docker run -p 7860:7860 supportflow-ai
```

## 5. Hugging Face Space validation

- [ ] `huggingface-cli login` completed (requires your account token)
- [ ] Repo pushed to Docker Space (requires your Space URL/permissions)
- [ ] Space URL returns HTTP `200`
- [ ] `POST /reset` responds correctly
- [ ] `GET /baseline` works

## 6. Submission package sanity

- [x] `openenv.yaml` present
- [x] `Dockerfile` present
- [x] `README.md` has setup + endpoints + action/observation info
- [x] `app/` includes typed models and `step/reset/state` implementation
- [x] No local-only secrets committed

## 7. Final pre-submit command set

```bash
git status --short --branch
python scripts/run_baseline.py
python scripts/smoke_test.py
```

Only submit when all checks pass.
