# QuotaDrift

QuotaDrift is a contract-driven AI reliability layer for FastAPI applications.

Instead of best-effort model routing, each request carries explicit latency, reliability, and cost constraints. QuotaDrift enforces those constraints at runtime, adapts strategy per request, and returns an auditable decision trace.

## Why QuotaDrift

Production AI systems drift under pressure:

- provider outages and rate limits
- latency spikes under load
- unpredictable cost behavior
- silent degradation with no accountability

QuotaDrift converts those failure modes into controlled outcomes with observable routing decisions.

## Core Behavior

For every request:

1. Validate the request contract.
2. Score provider options with confidence-adjusted reliability.
3. Choose strategy: `single`, `hedged`, or `reject`.
4. Execute with timeout-aware routing and fallback guards.
5. Persist outcome data for reliability learning.
6. Return response with full trace metadata.

If the contract is impossible, QuotaDrift rejects early with explicit reason codes.

## Request Contract

The contract schema is implemented in `quotadrift/contract_models.py` and supports:

- `max_latency_ms`
- `min_reliability`
- `max_cost_usd`
- `allow_degrade`

The decision engine is implemented in `quotadrift/contract_engine.py`.

## Example API Usage

`POST /api/chat`

```json
{
  "session_id": 42,
  "message": "Summarize today\'s incidents",
  "contract": {
    "max_latency_ms": 1800,
    "min_reliability": 0.95,
    "max_cost_usd": 0.01,
    "allow_degrade": true
  }
}
```

Example high-level response fields:

- `success`
- `status` (`fulfilled`, `rejected`, `degraded`, `failed`)
- `trace.strategy`
- `trace.selected_providers`
- `trace.rejected_providers`
- `trace.contract_met`
- `trace.attempts`

## Repository Organization

```text
.
|-- quotadrift/
|   |-- __init__.py
|   |-- main.py
|   |-- router.py
|   |-- model_manager.py
|   |-- contract_engine.py
|   |-- contract_models.py
|   |-- memory.py
|   |-- cache.py
|   |-- compiler.py
|   |-- agent_runner.py
|   |-- enhanced_agent_runner.py
|   `-- mcp_server.py
|-- tests/
|   `-- test_router.py
|-- static/
|   |-- index.html
|   `-- shared.html
|-- docs/
|   `-- architecture.md
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   `-- workflows/
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Quick Start

### Local (venv)

```bash
git clone <your-fork-or-repo-url>
cd QuotaDrift
python -m venv .venv
```

Activate environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install and run:

```bash
pip install -r requirements.txt
uvicorn quotadrift.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

### Docker

```bash
docker compose up --build
```

## Development Standards

- Lint: Ruff
- Format: Black + Ruff format check
- Type-aware models: Pydantic
- Tests: Pytest

CI workflows live under `.github/workflows`.

## Key Endpoints

- `GET /api/health`
- `GET /api/model-status`
- `GET /metrics`
- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/provider-test`

## Architecture and Design Notes

See `docs/architecture.md` for:

- component boundaries
- decision flow
- persistence and observability model
- reliability invariants

## Contributing

- Review `CONTRIBUTING.md`
- Follow issue templates under `.github/ISSUE_TEMPLATE`
- Keep changes contract-safe and traceable

## License

MIT (see `LICENSE`).
