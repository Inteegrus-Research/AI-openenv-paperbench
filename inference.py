#!/usr/bin/env python3
"""
OpenEnv-PaperBench baseline inference script.

Stdout contract:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

All debug output goes to stderr only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("API_KEY", ""))
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

BENCHMARK = os.getenv("BENCHMARK", "paper_review_env_v1")
TASKS = ["task1", "task2", "task3", "task4"]
INSTANCE_ID = os.getenv("INSTANCE_ID", "instance_001")

TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.10


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _health_check() -> None:
    resp = requests.get(f"{ENV_BASE_URL}/health", timeout=15)
    resp.raise_for_status()


def _env_reset(task_id: str, instance_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "instance_id": instance_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _env_step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("observation"), dict):
        return payload["observation"]
    return payload


def _extract_reward(payload: Dict[str, Any]) -> float:
    if isinstance(payload.get("reward"), (int, float)):
        return float(payload["reward"])
    obs = _extract_observation(payload)
    if isinstance(obs.get("reward"), (int, float)):
        return float(obs["reward"])
    return 0.0


def _extract_done(payload: Dict[str, Any]) -> bool:
    if isinstance(payload.get("done"), bool):
        return payload["done"]
    obs = _extract_observation(payload)
    if isinstance(obs.get("episode_complete"), bool):
        return obs["episode_complete"]
    return False


def _extract_score(payload: Dict[str, Any]) -> float:
    if isinstance(payload.get("score"), (int, float)):
        return float(payload["score"])
    obs = _extract_observation(payload)
    if isinstance(obs.get("final_score"), (int, float)):
        return float(obs["final_score"])
    return 0.0


def _extract_error(payload: Dict[str, Any]) -> Optional[str]:
    if isinstance(payload.get("error"), str):
        return payload["error"]
    obs = _extract_observation(payload)
    if isinstance(obs.get("error"), str):
        return obs["error"]
    return None


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _build_prompt(task_id: str, obs: Dict[str, Any]) -> str:
    task_desc = obs.get("task_description", "")
    step = obs.get("step", 0)
    budget_remaining = obs.get("budget_remaining", 0)
    papers = obs.get("papers", [])
    decisions = obs.get("decisions_so_far", {})
    error = obs.get("error")

    lines: List[str] = [
        f"Task: {task_id}",
        f"Step: {step}",
        f"Budget remaining: {budget_remaining}",
        "",
        task_desc,
        "",
        "Papers:",
    ]

    for p in papers:
        pid = p.get("id", "")
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        topic_hint = p.get("topic_hint", "")
        methodology_hint = p.get("methodology_hint", "")
        claimed_contribution = p.get("claimed_contribution", "")
        lines.append(f"- id={pid}")
        lines.append(f"  title={title}")
        lines.append(f"  abstract={abstract}")
        if topic_hint:
            lines.append(f"  topic_hint={topic_hint}")
        if methodology_hint:
            lines.append(f"  methodology_hint={methodology_hint}")
        if claimed_contribution:
            lines.append(f"  claimed_contribution={claimed_contribution}")

    if decisions:
        lines.append("")
        lines.append("Decisions so far:")
        for pid in sorted(decisions.keys()):
            lines.append(f"- {pid}: {decisions[pid]}")

    if error:
        lines.append("")
        lines.append(f"Last error: {error}")

    lines.append("")
    lines.append("Return exactly one JSON object as the next action.")
    lines.append("Do not include markdown fences or explanation.")

    return "\n".join(lines)


def _parse_action(text: str) -> Dict[str, Any]:
    raw = text.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {"action_type": "submit"}


def _llm_action(client: OpenAI, prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agent for a text-based paper screening environment. "
                    "Respond with only valid JSON for the next action."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return {"action_type": "submit"}
    return _parse_action(content)


def _run_task(client: OpenAI, task_id: str) -> Tuple[float, int, List[float]]:
    reset_payload = _env_reset(task_id, INSTANCE_ID)
    session_id = reset_payload["session_id"]
    obs = _extract_observation(reset_payload)

    rewards: List[float] = []
    steps = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    while True:
        if obs.get("episode_complete", False):
            break

        prompt = _build_prompt(task_id, obs)
        action = _llm_action(client, prompt)
        action_str = _compact_json(action)

        step_payload = _env_step(session_id, action)
        reward = _extract_reward(step_payload)
        done = _extract_done(step_payload)
        error = _extract_error(step_payload)
        obs = _extract_observation(step_payload)

        steps += 1
        rewards.append(reward)

        log_step(step=steps, action=action_str, reward=reward, done=done, error=error)

        if done or obs.get("episode_complete", False):
            break

    score = _extract_score(obs)
    if score == 0.0 and rewards:
        score = max(0.0, min(1.0, sum(rewards) / max(len(rewards), 1)))

    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return score, steps, rewards


def main() -> None:
    try:
        _health_check()
    except Exception as exc:
        _eprint(f"[ERROR] environment health check failed: {exc}")
        raise SystemExit(1)

    if not HF_TOKEN:
        _eprint("[ERROR] HF_TOKEN is required")
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    total_scores: List[float] = []
    for task_id in TASKS:
        try:
            score, _, _ = _run_task(client, task_id)
            total_scores.append(score)
        except Exception as exc:
            _eprint(f"[ERROR] task {task_id} failed: {exc}")
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            total_scores.append(0.0)

    if total_scores:
        _eprint(f"[SUMMARY] mean_score={sum(total_scores) / len(total_scores):.4f}")


if __name__ == "__main__":
    main()
