# OpenEnv-PaperBench

**Environment ID:** `paper_review_env_v1`  
**Domain:** Research paper screening / review  
**Submission type:** OpenEnv Round 1 — solo

---

## Overview

OpenEnv-PaperBench is a sequential budget-constrained literature screening environment. An AI agent acts as a constrained reviewer: given a batch of synthetic research paper abstracts and a fixed step budget, it must assign relevance labels, quality scores, and/or ranked justifications before the budget runs out.

The environment presents four tasks of increasing difficulty, from binary relevance classification to ranked shortlisting with justification under a structural budget deficit. All grading is deterministic and rule-based — no LLM is used in evaluation.

---

## Tasks

| Task | Objective | Budget | Papers | Grader |
|------|-----------|--------|--------|--------|
| task1 | Binary relevance (RELEVANT / NOT_RELEVANT) | 12 | 10 | Binary F1 |
| task2 | Relevance + quality score (1–4) | 14 | 10 | 0.6×F1 + 0.4×quality_acc |
| task3 | Adversarial screening (INCLUDE / EXCLUDE / DEFER) | 15 | 15 | 0.85×F1 + 0.15×efficiency |
| task4 | Ranked top-5 with justification | 18 | 20 | 0.5×nDCG@5 + 0.35×F1 + 0.15×justification |

---

## Action Schema

All tasks use a single unified action type:

```json
{
  "action_type": "review",
  "paper_id": "p001",
  "label": "RELEVANT",
  "quality_score": 3,
  "rank": 1,
  "justification": "Strong diagnostic ML model with evaluation baselines."
}
```

Or to end the episode early:

```json
{"action_type": "submit"}
```

Field requirements by task:
- **task1**: `label` required (RELEVANT / NOT_RELEVANT)
- **task2**: `label` + `quality_score` (1–4) required
- **task3**: `label` required (INCLUDE / EXCLUDE / DEFER)
- **task4**: `label` required; INCLUDE also requires `rank` (1–5) and `justification` (≤200 chars)

---

## Running the Server Locally

```bash
# Install server dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Verify health
curl http://localhost:7860/health
# {"status":"ok","env":"paper_review_env_v1"}

# List tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1","instance_id":"instance_001"}'

# Take a step (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","action":{"action_type":"review","paper_id":"p001","label":"RELEVANT"}}'
```

---

## Running with Docker

```bash
docker build -t paperbench .
docker run -p 7860:7860 paperbench
```

---

## Running the Baseline

```bash
# Install inference dependencies
pip install -r requirements-inference.txt

# Set required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<your-api-key>"
export ENV_BASE_URL="http://localhost:7860"  # or HF Space URL

# Run
python inference.py
```

Expected stdout format:

```
[START] task_id=task1 episode=1
[STEP] step=1 action={"action_type":"review","paper_id":"p001","label":"RELEVANT"} reward=null
[STEP] step=2 action={"action_type":"review","paper_id":"p002","label":"NOT_RELEVANT"} reward=null
...
[END] task_id=task1 episode=1 final_score=0.8000
[START] task_id=task2 episode=1
...
```

---

## Running Tests

```bash
# Install test dependencies (requires pydantic, pytest)
pip install -r requirements.txt pytest

# Run all tests
pytest tests/ -v
```

---

## Repository Structure

```
openenv-paperbench/
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # Container for HF Space
├── requirements.txt          # Server deps only
├── requirements-inference.txt
├── inference.py              # Baseline agent
├── env/
│   ├── environment.py        # PaperReviewEnv: reset/step/state
│   ├── models.py             # Pydantic models
│   ├── reward.py             # Grade dispatcher
│   └── utils.py             # Constants and helpers
├── tasks/
│   ├── task_base.py          # Abstract base + fixture loader
│   ├── task1.py … task4.py   # Per-task config + validation
├── graders/
│   └── graders.py            # Four deterministic graders
├── fixtures/
│   └── task{1-4}/instance_00{1-5}.json  # 20 fixtures total
├── server/
│   ├── app.py                # FastAPI: /health /tasks /reset /step
│   └── session.py            # In-memory session store
├── scripts/
│   └── validate_fixtures.py  # Dev-only fixture health check
└── tests/
    ├── test_env.py
    └── test_graders.py
```

---

## Scoring

All scores are in [0.0, 1.0]. Graders are pure functions with static fixture ground truth — no LLM is ever called during evaluation.

| Degenerate strategy | task1 | task2 | task3 | task4 |
|--------------------|-------|-------|-------|-------|
| All positive       | ~0.67 | ~0.67 | ~0.59 | ~0.14 |
| All negative       | 0.00  | ~0.13 | 0.00  | 0.00  |
| Empty episode      | 0.00  | ~0.13 | 0.00  | 0.00  |
| Perfect agent      | 1.00  | 1.00  | 0.85  | ~1.00 |
