# QuotaDrift Architecture

## Overview

QuotaDrift is a control plane around multi-provider LLM execution. It applies request-level contracts before and during inference, then records outcomes for adaptive reliability scoring.

## Components

- API Layer: `quotadrift/main.py`
- Contract Decision Engine: `quotadrift/contract_engine.py`
- Contract Schema: `quotadrift/contract_models.py`
- Provider Router: `quotadrift/router.py`
- Adaptive Health and Circuit Breakers: `quotadrift/model_manager.py`
- Persistence and Search Memory: `quotadrift/memory.py`
- Semantic Cache: `quotadrift/cache.py`
- Context Compiler: `quotadrift/compiler.py`

## Request Flow

1. Client calls `POST /api/chat` with message and optional contract.
2. `main.py` validates request and initializes request trace.
3. `contract_engine.py` computes strategy (`single`, `hedged`, `reject`).
4. `router.py` executes provider call(s) through LiteLLM.
5. `model_manager.py` records timing, errors, and provider health changes.
6. `memory.py` stores message and provider outcome records.
7. API returns response with contract status and execution trace.

## Reliability Invariants

- Impossible contracts are rejected explicitly.
- Contract fulfillment is not inferred from provider success alone.
- Hedged execution drains or cancels tasks safely to avoid leaks.
- Provider health windows use consistent epoch timestamps.
- Reject/degrade outcomes are persisted for unbiased scoring.

## Observability

- Prometheus metrics endpoint: `GET /metrics`
- Health endpoint: `GET /api/health`
- Model status snapshot: `GET /api/model-status`
- Structured logs include request IDs and error details.

## Persistence Model

SQLite stores:

- projects
- sessions
- messages
- indexed files metadata
- provider outcomes

Chroma + BM25 hybrid retrieval supports semantic context reconstruction per project/session.

## Extension Points

- Add provider slots in `quotadrift/config.py`
- Tune scoring/circuit thresholds in `quotadrift/model_manager.py`
- Extend contract semantics in `quotadrift/contract_models.py`
- Add decision policies in `quotadrift/contract_engine.py`
