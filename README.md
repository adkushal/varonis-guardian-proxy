# IBM Granite Guardian Exercise — OpenAI reverse proxy with risk blocking

TLS-terminated reverse proxy in front of OpenAI Chat Completions. Traffic path:

```text
Python client  --HTTPS (verified, local CA)-->  NGINX  --HTTPS (verified, local CA)-->  mitmproxy addon
                                                                                          |       |
                                                                                          |       +--> OpenAI (if allowed)
                                                                                          +--> Granite Guardian judge API
```

## Features

- **NGINX** terminates TLS with a certificate issued by a local CA. The client uses `verify=True` + `SSL_CERT_FILE` (never `verify=False`).
- **NGINX → mitmproxy is re-encrypted**: mitmproxy listens with its own CA-signed cert (`mitm.pem`) and NGINX verifies it (`proxy_ssl_verify on`, name `mitm`).
- **mitmproxy** reverse-proxies `POST /v1/chat/completions`, screens **every message role** (system, user, assistant, tool), forwards or blocks.
- **Simple heuristics** classify violence / illegal how-tos / sexual content → HTTP **403** with the assignment message text.
- **Granite Guardian** multi-criteria toxicity (default `harm,profanity`): any `yes` → `The prompt is considered toxic.`
- **Fail-closed by default**: if Guardian can't give a verdict, the request is rejected with HTTP **503**.
- **Responses** are monitored in the background (logged, never blocked or delayed).
- **Guardian modes**: `mock` (default, no GPU) or `vllm` (Compose profile `gpu`).

## Quick start (no GPU)

### 1. Prerequisites

- Docker + Docker Compose
- Python 3.10+ with `cryptography` (for cert generation)
- An OpenAI API key

### 2. Certificates

```bash
pip install cryptography
python scripts/gen_certs.py
```

Creates in `certs/`:

| File | Used by |
|------|---------|
| `ca.crt` / `ca.key` | Local CA (the key never leaves the host) |
| `server.crt` / `server.key` | NGINX (SAN: `localhost`, `nginx`, `127.0.0.1`) |
| `mitm.pem` | mitmproxy listener key + cert (SAN: `mitm`) |

Only missing files are generated. `FORCE=1 python scripts/gen_certs.py` regenerates everything.

### 3. Environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=...
```

### 4. Start the proxy stack

```bash
docker compose up --build -d
```

Services: `nginx` (`:8443`), `mitm`, `guardian` (mock, `127.0.0.1:8000`). Internal ports are bound to localhost only.

### 5. Run demos

```bash
# Rebuild the client image after code changes (it lives in a profile, so `up` skips it)
docker compose --profile client build client

# All built-in demos (safe, violence, illegal, sexual, toxic)
docker compose --profile client run --rm client --demo all

# Single demo
docker compose --profile client run --rm client --demo violence

# TLS check: verify against the public CA bundle instead of the local CA → must fail
docker compose --profile client run --rm client --demo safe --system-ca
```

Or on the host:

```powershell
pip install -r client/requirements.txt
$env:SSL_CERT_FILE=".\certs\ca.crt"
$env:PROXY_BASE_URL="https://localhost:8443"
$env:OPENAI_API_KEY="sk-..."
python client/chat_client.py --demo all
```

Expected:

| Demo | Result |
|------|--------|
| safe | 200 + OpenAI answer |
| violence | 403 … contained Description of violent acts |
| illegal | 403 … contained Inquiries on how to perform an illegal activity |
| sexual | 403 … contained Any sexual content |
| toxic | 403 The prompt is considered toxic. |
| `--system-ca` | `TLS/CONNECT FAILED: CERTIFICATE_VERIFY_FAILED` |
| any, with `guardian` stopped | 503 `guardian_unavailable` |

### 6. Tests

```bash
pip install "mitmproxy==10.4.2" pytest fastapi httpx
cd mitm && python -m pytest -q
cd ../guardian && python -m pytest -q
```

They cover the classifier, chain/strategies/decorators, the Guardian client (mocked HTTP), the addon hooks (mitmproxy test flows), and the judge service (prompt format, score parsing, mock mode, mocked vLLM).

## Architecture details

### Prompt pipeline

1. Route allowlist: only `POST /v1/chat/completions` is accepted; anything else → 404. Bad JSON → 400.
2. Heuristic classifier (`mitm/simple_classifier.py`) — first match wins.
3. Guardian criteria (`mitm/guardian_client.py`) — parallel no-think judges; block if any `yes`. 503 if any judge errored and none said `yes` (fail-closed).
4. Otherwise forward to `https://api.openai.com`.

Error bodies follow the OpenAI error shape:

```json
{
  "error": {
    "message": "The prompt was blocked because it contained Description of violent acts",
    "type": "prompt_blocked",
    "code": "heuristic_violence",
    "param": null
  }
}
```

| Code | Status | Meaning |
|------|--------|---------|
| `heuristic_violence` / `heuristic_illegal` / `heuristic_sexual` | 403 | Heuristic category match |
| `guardian_toxic` | 403 | A Guardian criterion returned `yes` |
| `guardian_unavailable` | 503 | No Guardian verdict (fail-closed) |
| `unsupported_route` | 404 | Not `POST /v1/chat/completions` |
| `invalid_json` | 400 | Body isn't a JSON object |

### Extending Guardian criteria

Set `GUARDIAN_CRITERIA` to a comma-separated list of registry IDs:

```bash
GUARDIAN_CRITERIA=harm,profanity,jailbreaking,social_bias
```

Known IDs: `harm`, `profanity`, `violence`, `sexual_content`, `unethical_behavior`, `social_bias`, `jailbreaking`. Definitions follow the Granite Guardian 4.1 model card (except `sexual_content`). Add new entries to `CRITERIA_REGISTRY` in `mitm/guardian_client.py`. Each criterion is a separate, concurrent judge call, so adding criteria adds little latency.

### OOP patterns in the mitm gate

- **Strategy** (`mitm/strategies.py`): `HeuristicBlockStrategy` and `GuardianToxicityStrategy` implement `PromptEvaluationStrategy`.
- **Chain of Responsibility** (`mitm/chain.py`): `PromptHandler` nodes implement the `PromptGate` interface; first `BlockDecision` wins. Default chain: heuristic → Guardian.
- **Decorator** (`mitm/decorators.py`): `LoggingPromptGate` and `TimingPromptGate` wrap any `PromptGate`; `decorate_prompt_gate()` composes them.

To add a check, implement a strategy and insert it in `build_default_prompt_chain()` / `build_chain([...])`.
To add cross-cutting behavior (metrics, audit), subclass `PromptGateDecorator`.

### Real Granite Guardian (GPU)

Requires an NVIDIA GPU, the NVIDIA Container Toolkit, and enough VRAM for the 8B model.

```bash
# .env
GUARDIAN_MODE=vllm
# optional: HUGGING_FACE_HUB_TOKEN=...

docker compose --profile gpu up --build -d
```

This starts `vllm/vllm-openai:v0.11.0` serving `ibm-granite/granite-guardian-4.1-8b`, and the judge service waits for it to be healthy. The judge sends the model-card prompt exactly: the judged text as its own message, followed by a user message holding the `<no-think>` judge instruction, the criteria, and the scoring schema; the first `<score>yes|no</score>` is the verdict. Unparseable model output is treated as an error (and therefore fail-closed), never as safe.

See the [Granite Guardian model card](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) and [vLLM](https://github.com/vllm-project/vllm).

## Project layout

```text
client/          Step-1 OpenAI client (via NGINX)
nginx/           TLS terminator + re-encryption to mitmproxy
mitm/            mitmproxy addon, classifier, Guardian client, chain/strategies/decorators, tests
guardian/        Judge HTTP API (mock | vllm), tests
scripts/         Certificate generation (gen_certs.py)
certs/           Generated CA + certs (local, gitignored — includes private keys)
docker-compose.yml
```

## Notes for reviewers

- `GUARDIAN_MODE=mock` implements the same `/v1/judge` contract as vLLM mode, so the full proxy path runs without a GPU. vLLM mode is unit-tested against a mocked vLLM API but hasn't been run on real hardware.
- Set `GUARDIAN_FAIL_CLOSED=0` to fail open (forward when Guardian is unreachable).
- The mitm → guardian hop is plain HTTP on the internal Docker network; only NGINX's port is meant to be exposed.
- nginx gets only `server.crt`/`server.key` + `ca.crt`, mitm gets only `mitm.pem`, and the client gets only `ca.crt`; the CA private key stays on the host.
