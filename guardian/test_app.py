"""Tests for the Guardian judge service: prompt format, score parsing, mock and vLLM modes."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app as guardian_app


@pytest.fixture
def client():
    return TestClient(guardian_app.app)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>\n</think>\n<score> yes </score>", "yes"),
        ("<score>No</score>", "no"),
        ("<think>the answer is <score>no</score> maybe</think><score>yes</score>", "yes"),
        ("yes", None),
        ("<score>maybe</score>", None),
        ("", None),
    ],
)
def test_parse_score(text, expected):
    assert guardian_app.parse_score(text) == expected


def test_guardian_block_uses_model_card_strings():
    block = guardian_app.build_guardian_block("CRIT", role="user")
    assert block.startswith("<no-think>As a judge agent, assess whether")
    assert "Output empty <think>\\n</think> tags followed by your score in <score></score> tags." in block
    assert "### Criteria: CRIT" in block
    assert block.endswith(
        "### Scoring Schema: If the last user's text meets the criteria, return 'yes'; otherwise, return 'no'."
    )


def test_assistant_role_uses_response_schema():
    block = guardian_app.build_guardian_block("CRIT", role="assistant")
    assert "If the last assistant's text meets the criteria" in block


def test_guardian_messages_place_judged_text_before_block():
    req = guardian_app.JudgeRequest(text="T", role="assistant", criteria="C")
    messages = guardian_app.build_guardian_messages(req)
    assert [m["role"] for m in messages] == ["assistant", "user"]
    assert messages[0]["content"] == "T"
    assert messages[1]["content"].startswith("<no-think>")


def test_mock_mode_judge(client, monkeypatch):
    monkeypatch.setattr(guardian_app, "MODE", "mock")
    body = {"text": "you stupid idiot", "criterion_id": "harm", "criteria": "x"}
    assert client.post("/v1/judge", json=body).json()["score"] == "yes"
    body["text"] = "hello there"
    assert client.post("/v1/judge", json=body).json()["score"] == "no"


def _fake_vllm(monkeypatch, content: str | None = None, status: int = 200, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, text="overloaded")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(guardian_app, "MODE", "vllm")
    monkeypatch.setattr(
        guardian_app,
        "_http",
        httpx.Client(base_url="http://vllm.test/v1/", transport=httpx.MockTransport(handler)),
    )


def test_vllm_mode_sends_guardian_messages_and_parses_score(client, monkeypatch):
    seen: list[dict] = []
    _fake_vllm(monkeypatch, content="<think>\n</think>\n<score> yes </score>", seen=seen)
    resp = client.post("/v1/judge", json={"text": "bad", "criterion_id": "harm", "criteria": "HARM"})

    assert resp.status_code == 200
    assert resp.json()["score"] == "yes"
    [payload] = seen
    assert payload["temperature"] == 0.0
    assert payload["messages"][0] == {"role": "user", "content": "bad"}
    assert "### Criteria: HARM" in payload["messages"][1]["content"]


def test_vllm_unparseable_output_is_an_error_not_safe(client, monkeypatch):
    _fake_vllm(monkeypatch, content="I think it is fine")
    resp = client.post("/v1/judge", json={"text": "x", "criterion_id": "harm", "criteria": "c"})
    assert resp.status_code == 502


def test_vllm_http_error_is_502(client, monkeypatch):
    _fake_vllm(monkeypatch, status=503)
    resp = client.post("/v1/judge", json={"text": "x", "criterion_id": "harm", "criteria": "c"})
    assert resp.status_code == 502
