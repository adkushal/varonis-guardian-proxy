"""Guardian HTTP service: mock (default) or vLLM-backed Granite Guardian."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("guardian_service")
logging.basicConfig(level=logging.INFO)

MODE = os.getenv("GUARDIAN_MODE", "mock").lower()
MODEL_PATH = os.getenv("GUARDIAN_MODEL", "ibm-granite/granite-guardian-4.1-8b")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")

app = FastAPI(title="Granite Guardian Judge", version="1.0.0")


class JudgeRequest(BaseModel):
    text: str
    role: Literal["user", "assistant"] = "user"
    criterion_id: str = "harm"
    criteria: str
    think: bool = False


class JudgeResponse(BaseModel):
    score: str = Field(description="yes if risk present, else no")
    criterion_id: str
    mode: str
    raw: str | None = None


# Copied verbatim from the granite-guardian-4.1-8b model card; the model is
# trained on these exact strings, so do not paraphrase them.
GUARDIAN_JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the "
    "given judging criteria. Use all available information, including conversations, "
    "documents, and tools. Reason through the evidence step by step before providing "
    "your score. Wrap your reasoning in <think></think> tags, then provide your "
    "score in <score></score> tags."
)
GUARDIAN_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)

PROMPT_ONLY_SCHEMA = (
    "If the last user's text meets the criteria, return 'yes'; otherwise, return 'no'."
)
PROMPT_RESPONSE_SCHEMA = (
    "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
)


def build_guardian_block(criteria: str, role: str = "user", think: bool = False) -> str:
    judge = GUARDIAN_JUDGE_THINK if think else GUARDIAN_JUDGE_NOTHINK
    schema = PROMPT_RESPONSE_SCHEMA if role == "assistant" else PROMPT_ONLY_SCHEMA
    return (
        f"{judge}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: {schema}"
    )


def build_guardian_messages(req: JudgeRequest) -> list[dict[str, Any]]:
    block = build_guardian_block(req.criteria, role=req.role, think=req.think)
    return [
        {"role": req.role, "content": req.text},
        {"role": "user", "content": block},
    ]


def parse_score(text: str) -> str | None:
    """Model-card parser: drop the think trace, then read the first <score> tag."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.findall(r"<score>\s*(.*?)\s*</score>", cleaned, flags=re.DOTALL)
    if not match:
        return None
    score = match[0].strip().lower()
    return score if score in ("yes", "no") else None


_MOCK_PATTERNS: dict[str, re.Pattern[str]] = {
    "harm": re.compile(
        r"\b(hate|kill|idiot|stupid|worthless|trash\s+human|you\s+suck)\b",
        re.I,
    ),
    "profanity": re.compile(
        r"\b(fuck|shit|bitch|asshole|damn\s+you)\b",
        re.I,
    ),
    "violence": re.compile(r"\b(kill|murder|stab|shoot|bomb|torture)\b", re.I),
    "sexual_content": re.compile(r"\b(porn|explicit\s+sex|nude)\b", re.I),
    "unethical_behavior": re.compile(r"\b(steal|fraud|hack\s+into|launder)\b", re.I),
    "social_bias": re.compile(r"\b(all\s+\w+\s+are\s+(stupid|inferior|criminals))\b", re.I),
    "jailbreaking": re.compile(r"\b(ignore\s+(all\s+)?(previous\s+)?instructions|dan\s+mode)\b", re.I),
}


def mock_judge(text: str, criterion_id: str) -> str:
    pattern = _MOCK_PATTERNS.get(criterion_id)
    if pattern and pattern.search(text):
        return "yes"
    if criterion_id == "harm" and re.search(r"\b(toxic|abusive|harass)\b", text, re.I):
        return "yes"
    return "no"


_http: httpx.Client | None = None


def _vllm_http() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(base_url=VLLM_BASE_URL.rstrip("/") + "/", timeout=120.0)
    return _http


def vllm_judge(req: JudgeRequest) -> tuple[str, str]:
    """Judge via vLLM's chat API, which applies the model's own chat template."""
    payload = {
        "model": MODEL_PATH,
        "messages": build_guardian_messages(req),
        "temperature": 0.0,
        "max_tokens": 64 if not req.think else 2048,
    }
    try:
        resp = _vllm_http().post("chat/completions", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM unreachable: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"vLLM error: {resp.text}")

    raw = resp.json()["choices"][0]["message"]["content"] or ""
    score = parse_score(raw)
    if score is None:
        # Surface as an error so the proxy's fail-closed policy applies,
        # instead of silently treating an unreadable verdict as "safe".
        raise HTTPException(status_code=502, detail=f"Unparseable Guardian output: {raw[:200]}")
    return score, raw


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": MODE, "model": MODEL_PATH}


@app.post("/v1/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest) -> JudgeResponse:
    if not req.text.strip():
        return JudgeResponse(score="no", criterion_id=req.criterion_id, mode=MODE)

    if MODE == "mock":
        score = mock_judge(req.text, req.criterion_id)
        return JudgeResponse(score=score, criterion_id=req.criterion_id, mode=MODE)

    if MODE == "vllm":
        score, raw = vllm_judge(req)
        return JudgeResponse(
            score=score,
            criterion_id=req.criterion_id,
            mode=MODE,
            raw=raw,
        )

    raise HTTPException(status_code=500, detail=f"Unknown GUARDIAN_MODE={MODE}")
