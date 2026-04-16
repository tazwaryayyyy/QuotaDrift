# QuotaDrift

Contract-driven AI inference that enforces latency, reliability, and cost per request, then proves every routing decision with a trace.

## 1. Title + Tagline

**QuotaDrift: AI reliability with enforceable contracts, not best-effort routing.**

## 2. The Problem

AI features fail in production for predictable reasons:

- provider outages
- quota exhaustion
- latency spikes
- unpredictable cost drift

Most systems route and hope. When contracts are missed, users see failures and teams lose trust.

## 3. The Solution

QuotaDrift takes a request-level contract and enforces it at runtime.

For each request, it decides whether to:

- run a single provider
- hedge across providers
- degrade deliberately
- reject immediately when the contract is impossible

Every decision returns a trace explaining what was chosen and why alternatives were rejected.

## 4. Why This Is Different

This is not a wrapper over multiple models.

QuotaDrift is a policy-enforced control layer:

- hard constraints first
- adaptive strategy second
- explainability by default

If the contract cannot be met, it says so explicitly. No silent violations.

## 5. Core Concept

Each request carries a contract:

- `max_latency_ms`
- `min_reliability`
- `max_cost_usd`
- `allow_degrade`

The engine scores providers with confidence-adjusted reliability, predicts risk, and chooses the safest valid path.

## 6. How It Works

1. Receive request + contract
2. Fast-reject impossible constraints
3. Score available providers with confidence penalty
4. Choose strategy: `single`, `hedged`, or `reject`
5. Execute with timeout and fallback guards
6. Persist outcome and update provider stats
7. Return response with full decision trace

## 7. Key Capabilities

- **Contract Enforcement**: latency and cost limits are checked on every response.
- **Adaptive Routing**: strategy shifts between single and hedged based on risk.
- **Explainable Decisions**: trace includes selected and rejected providers with reasons.
- **Controlled Degradation**: degrade with explicit `degrade_reason`, never implicit failure.
- **Outcome Ledger**: every request is recorded for reliability and cost accountability.

## 8. Example Request + Response

### Request

```http
POST /api/chat
Content-Type: application/json
```

```json
{
    "session_id": 42,
    "message": "Generate a release summary for this sprint",
    "contract": {
        "max_latency_ms": 1800,
        "min_reliability": 0.95,
        "max_cost_usd": 0.01,
        "allow_degrade": true
    }
}
```

### Response

```json
{
    "success": true,
    "status": "fulfilled",
    "content": "Here is your sprint release summary...",
    "latency_ms": 1214,
    "contract": {
        "max_latency_ms": 1800,
        "min_reliability": 0.95,
        "max_cost_usd": 0.01,
        "allow_degrade": true
    },
    "trace": {
        "request_id": "5a1c7f3d-1d66-4a8e-ae37-4f8f8d6bc5ef",
        "strategy": "hedged",
        "selected_providers": ["primary", "secondary"],
        "rejected_providers": [
            {
                "provider": "fallback",
                "reason": "cost too high"
            }
        ],
        "reason": "Single route insufficient; hedged route fulfills contract",
        "risk_level": "medium",
        "degrade_reason": null,
        "contract_met": true,
        "fallback_triggered": false,
        "attempts": [
            {
                "provider": "primary",
                "success": true,
                "latency_ms": 1188,
                "error": null,
                "error_code": null
            }
        ]
    },
    "error": null,
    "error_code": null
}
```

## 9. What Makes It Powerful

- Protects user-facing AI flows from provider volatility
- Prevents cost surprises with request-level ceilings
- Turns reliability decisions into auditable artifacts
- Gives teams deterministic behavior under failure, not guesswork

## 10. Quick Start

```bash
git clone https://github.com/tazwaryayyyy/quotadrift.git
cd quotadrift

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# configure your provider keys
cp .env.example .env

python main.py
```

Open: `http://localhost:8000`

## 11. Tech Stack

- FastAPI (API runtime)
- LiteLLM (provider execution)
- SQLite (outcome and session persistence)
- Prometheus metrics (operational visibility)

## 12. Closing Line

QuotaDrift makes AI requests behave like production contracts, not optimistic API calls.
