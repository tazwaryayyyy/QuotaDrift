from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RequestContract(BaseModel):
    max_latency_ms: int = Field(default=3000, ge=200, le=30000)
    min_reliability: float = Field(default=0.9, ge=0.5, le=0.999)
    max_cost_usd: float = Field(default=0.02, gt=0, le=2.0)
    allow_degrade: bool = True


class DecisionResult(BaseModel):
    strategy: Literal["single", "hedged", "reject"]
    enforcement: Literal["fulfill", "degrade", "reject"]
    selected_providers: list[str] = Field(default_factory=list)
    fallback_providers: list[str] = Field(default_factory=list)
    rejected_providers: list[dict[str, str]] = Field(default_factory=list)
    reason: str
    risk_level: Literal["low", "medium", "high"] = "medium"
    degrade_reason: str | None = None
    expected_reliability: float = 0.0
    expected_latency_ms: int = 0
    estimated_cost_usd: float = 0.0


class OutcomeRecord(BaseModel):
    request_id: str
    session_id: int
    strategy: Literal["single", "hedged", "reject"]
    selected_providers: list[str]
    winner_provider: str | None = None
    success: bool
    latency_ms: int
    tokens: int = 0
    cost_usd: float = 0.0
    contract_met: bool
    fallback_triggered: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
