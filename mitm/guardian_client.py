"""Guardian criteria registry + HTTP client (mock or vLLM-backed service)."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import httpx

logger = logging.getLogger("guardian_client")

# Pre-baked criteria definitions from the granite-guardian-4.1-8b model card.
# sexual_content has no published definition there, so it uses our own wording.
CRITERIA_REGISTRY: dict[str, str] = {
    "harm": (
        "Texts deemed universally harmful are those that pose harm across all contexts "
        "and audiences."
    ),
    "social_bias": (
        "Social bias reflects systemic prejudice against groups defined by shared identity "
        "or characteristics. This often stems from stereotypes or cultural influences, "
        "manifesting as thoughts, attitudes, or behaviors that unfairly favor or disfavor "
        "certain groups."
    ),
    "jailbreaking": (
        "A jailbreak text refers to the deliberate evasion of an AI system's built-in "
        "safeguards or ethical boundaries. This practice often involves crafting specific "
        "prompts or scenarios intended to manipulate the AI into producing restricted or "
        "inappropriate content."
    ),
    "violence": (
        "Violence refers to content that promotes or depicts physical harm to individuals "
        "or groups. This includes depictions of assault, self-harm, or the creation of a "
        "hostile environment through threats or intimidation."
    ),
    "profanity": (
        "Profanity refers to the use of language deemed offensive or socially unacceptable "
        "in a given context. This primarily includes curse words, swear words, and crude "
        "expressions, though it typically excludes slurs or derogatory terms specifically "
        "targeting certain groups."
    ),
    "unethical_behavior": (
        "Unethical behavior refers to actions that violate moral or professional standards, "
        "particularly those involving exploitation for personal gain or a disregard for "
        "others' well-being. This category excludes physical violence or explicit bias and "
        "instead encompasses activities like fraud, theft, or the abuse of power."
    ),
    "sexual_content": (
        "Sexual content refers to explicit or suggestive material of a sexual nature."
    ),
}


@dataclass(frozen=True)
class CriterionResult:
    criterion_id: str
    risk: bool  # True if Guardian score is "yes" (risk present)
    raw_score: str | None = None
    error: bool = False  # Guardian could not produce a verdict


def enabled_criteria_from_env(env_value: str | None = None) -> list[str]:
    raw = env_value if env_value is not None else os.getenv("GUARDIAN_CRITERIA", "harm,profanity")
    ids = [c.strip().lower() for c in raw.split(",") if c.strip()]
    unknown = [c for c in ids if c not in CRITERIA_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown Guardian criteria: {unknown}. Known: {sorted(CRITERIA_REGISTRY)}")
    return ids


def fail_closed_from_env() -> bool:
    return os.getenv("GUARDIAN_FAIL_CLOSED", "1") == "1"


class GuardianClient:
    """Calls the Guardian HTTP service for one or more criteria in parallel."""

    def __init__(
        self,
        base_url: str | None = None,
        criteria: Iterable[str] | None = None,
        timeout: float = 60.0,
        fail_closed: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GUARDIAN_URL", "http://guardian:8000")).rstrip("/")
        self.criteria = list(criteria) if criteria is not None else enabled_criteria_from_env()
        self.fail_closed = fail_closed if fail_closed is not None else fail_closed_from_env()
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        # Created lazily so it binds to the event loop that actually serves requests.
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evaluate_text(self, text: str, role: str = "user") -> list[CriterionResult]:
        """Evaluate all enabled criteria concurrently."""
        if not self.criteria:
            return []
        client = self._http()
        tasks = [
            self._evaluate_one(client, text=text, role=role, criterion_id=cid)
            for cid in self.criteria
        ]
        return list(await asyncio.gather(*tasks))

    async def _evaluate_one(
        self,
        client: httpx.AsyncClient,
        text: str,
        role: str,
        criterion_id: str,
    ) -> CriterionResult:
        payload = {
            "text": text,
            "role": role,
            "criterion_id": criterion_id,
            "criteria": CRITERIA_REGISTRY[criterion_id],
            "think": False,
        }
        try:
            resp = await client.post("/v1/judge", json=payload)
            resp.raise_for_status()
            score = str(resp.json().get("score", "")).strip().lower()
        except Exception as exc:  # noqa: BLE001 — any failure means "no verdict"
            logger.warning("Guardian call failed for %s: %s", criterion_id, exc)
            return CriterionResult(criterion_id=criterion_id, risk=False, raw_score=None, error=True)

        if score not in ("yes", "no"):
            logger.warning("Guardian returned unexpected score %r for %s", score, criterion_id)
            return CriterionResult(criterion_id=criterion_id, risk=False, raw_score=score, error=True)
        return CriterionResult(criterion_id=criterion_id, risk=score == "yes", raw_score=score)
