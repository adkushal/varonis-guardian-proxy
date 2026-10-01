"""Tests for GuardianClient and GuardianToxicityStrategy against a mocked judge API."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from guardian_client import CRITERIA_REGISTRY, GuardianClient, enabled_criteria_from_env
from strategies import GuardianToxicityStrategy, PromptContext


def judge_transport(scores: dict[str, str | int], seen: list[dict] | None = None) -> httpx.MockTransport:
    """scores maps criterion_id → 'yes'/'no' for a verdict, or an HTTP status int for an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if seen is not None:
            seen.append(payload)
        outcome = scores[payload["criterion_id"]]
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"detail": "boom"})
        return httpx.Response(200, json={"score": outcome, "criterion_id": payload["criterion_id"]})

    return httpx.MockTransport(handler)


def make_client(scores, *, fail_closed=True, seen=None) -> GuardianClient:
    return GuardianClient(
        base_url="http://guardian.test",
        criteria=list(scores),
        fail_closed=fail_closed,
        transport=judge_transport(scores, seen),
    )


def test_evaluates_every_criterion_with_registry_text():
    seen: list[dict] = []
    client = make_client({"harm": "no", "profanity": "yes"}, seen=seen)
    results = asyncio.run(client.evaluate_text("hello", role="user"))

    assert {r.criterion_id: r.risk for r in results} == {"harm": False, "profanity": True}
    assert {p["criterion_id"] for p in seen} == {"harm", "profanity"}
    for payload in seen:
        assert payload["criteria"] == CRITERIA_REGISTRY[payload["criterion_id"]]
        assert payload["think"] is False
        assert payload["role"] == "user"


def test_http_error_is_reported_as_error_not_risk():
    client = make_client({"harm": 502})
    [result] = asyncio.run(client.evaluate_text("hello"))
    assert result.error is True
    assert result.risk is False


def test_unexpected_score_is_reported_as_error():
    client = make_client({"harm": "maybe"})
    [result] = asyncio.run(client.evaluate_text("hello"))
    assert result.error is True


def test_strategy_blocks_toxic_with_403():
    strategy = GuardianToxicityStrategy(make_client({"harm": "yes", "profanity": "no"}))
    decision = asyncio.run(strategy.evaluate(PromptContext(text="x")))
    assert decision is not None
    assert decision.status == 403
    assert decision.code == "guardian_toxic"
    assert decision.message == "The prompt is considered toxic."
    assert decision.detail == "harm"


def test_strategy_fail_closed_blocks_with_503_when_guardian_errors():
    strategy = GuardianToxicityStrategy(make_client({"harm": 500}, fail_closed=True))
    decision = asyncio.run(strategy.evaluate(PromptContext(text="x")))
    assert decision is not None
    assert decision.status == 503
    assert decision.code == "guardian_unavailable"


def test_strategy_fail_open_allows_when_guardian_errors():
    strategy = GuardianToxicityStrategy(make_client({"harm": 500}, fail_closed=False))
    assert asyncio.run(strategy.evaluate(PromptContext(text="x"))) is None


def test_risk_wins_over_partial_error():
    strategy = GuardianToxicityStrategy(make_client({"harm": "yes", "profanity": 500}))
    decision = asyncio.run(strategy.evaluate(PromptContext(text="x")))
    assert decision is not None
    assert decision.code == "guardian_toxic"


def test_unknown_criteria_rejected():
    with pytest.raises(ValueError):
        enabled_criteria_from_env("harm,not_a_criterion")
