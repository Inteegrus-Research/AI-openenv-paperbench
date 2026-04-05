"""
server/app.py
FastAPI application for OpenEnv-PaperBench HF Space deployment.

Exposes four endpoints required by the OpenEnv validator:
  GET  /health    — liveness check
  GET  /tasks     — enumerate all four tasks
  POST /reset     — start a new episode, returns Observation + session_id
  POST /step      — advance one step, returns next Observation

Phase 2: routes are importable and correctly structured.
Phase 7: full request/response wiring is completed for Docker deployment.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from server.session import session_store

app = FastAPI(
    title="OpenEnv-PaperBench",
    description="Research paper screening RL environment — paper_review_env_v1",
    version="0.1.0",
)

# ── Request / response bodies ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str
    instance_id: str = "instance_001"


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


# ── Task descriptors (static) ─────────────────────────────────────────────────

_TASK_DESCRIPTORS = [
    {
        "task_id": "task1",
        "description": "Binary relevance classification of paper abstracts.",
        "budget": 12,
        "paper_count": 10,
        "grader_type": "binary_f1",
    },
    {
        "task_id": "task2",
        "description": "Relevance + methodological quality scoring.",
        "budget": 14,
        "paper_count": 10,
        "grader_type": "weighted_f1_quality",
    },
    {
        "task_id": "task3",
        "description": "Adversarial batch screening with DEFER support.",
        "budget": 15,
        "paper_count": 15,
        "grader_type": "f1_budget_efficiency",
    },
    {
        "task_id": "task4",
        "description": "Ranking and justification under structural budget deficit.",
        "budget": 18,
        "paper_count": 20,
        "grader_type": "ndcg_f1_justification",
    },
]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "paper_review_env_v1"}


@app.get("/tasks")
def list_tasks():
    return _TASK_DESCRIPTORS


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode for the requested task.
    Returns the initial Observation and a session_id for subsequent /step calls.
    """
    from env.environment import PaperReviewEnv
    from env.models import PaperAction

    env = PaperReviewEnv()
    try:
        obs = env.reset(req.task_id, req.instance_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session_id = session_store.create(env)
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    """
    Advance the episode by one action.
    Returns the updated Observation.
    Invalid actions return a valid Observation with obs.error set (HTTP 200).
    """
    from env.models import PaperAction
    from pydantic import ValidationError

    env = session_store.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found or expired.",
        )

    try:
        action = PaperAction(**req.action)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action schema: {exc}",
        )

    try:
        obs = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return obs.model_dump()

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
