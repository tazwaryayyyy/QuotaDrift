import asyncio
import time
from typing import Any, cast

import config
import router as ai_router
from contract_models import DecisionResult, RequestContract


TOKENS_PER_REQUEST_ESTIMATE = 1200

# Conservative per-1k token prices for deterministic contract checks.
_PRICE_PER_1K = {
    "primary": 0.0005,
    "secondary": 0.0010,
    "tertiary": 0.0010,
    "quaternary": 0.0012,
    "siliconflow": 0.0008,
    "huggingface": 0.0006,
    "cloudflare": 0.0009,
    "fallback": 0.0014,
}


def _risk_level(
    min_reliability: float,
    expected_reliability: float,
    max_latency_ms: int,
    expected_latency_ms: int,
) -> str:
    reliability_gap = max(0.0, min_reliability - expected_reliability)
    latency_pressure = (
        expected_latency_ms /
        max(1, max_latency_ms) if max_latency_ms > 0 else 1.0
    )
    if reliability_gap > 0.08 or latency_pressure > 0.95:
        return "high"
    if reliability_gap > 0.03 or latency_pressure > 0.8:
        return "medium"
    return "low"


def estimate_cost_usd(provider_slot: str, tokens: int) -> float:
    price = _PRICE_PER_1K.get(provider_slot, 0.001)
    return round((tokens / 1000.0) * price, 6)


def _provider_latency_ms(provider: dict[str, Any]) -> int:
    latency_ms = int(provider.get("avg_latency_ms", 0) or 0)
    return latency_ms if latency_ms > 0 else 900


def _expected_model_id(provider_slot: str) -> str:
    for slot in config.MODEL_LIST:
        if slot["model_name"] == provider_slot:
            return slot["litellm_params"]["model"]
    return provider_slot


def _provider_score(
    provider: dict[str, Any],
    contract: RequestContract,
) -> tuple[float, float, int, float, int]:
    reliability = float(provider.get("success_rate", 0.0) or 0.0)
    request_count = int(provider.get("requests", 0) or 0)
    confidence = min(1.0, request_count / 50.0)
    adjusted_reliability = reliability * confidence

    latency_ms = int(provider.get("avg_latency_ms", 0) or 0)
    if latency_ms <= 0:
        latency_ms = 900

    est_cost = estimate_cost_usd(provider["id"], TOKENS_PER_REQUEST_ESTIMATE)

    latency_score = max(
        0.0, min(1.0, contract.max_latency_ms / max(latency_ms, 1)))
    cost_score = max(
        0.0, min(1.0, contract.max_cost_usd / max(est_cost, 1e-6)))

    total = (adjusted_reliability * 0.55) + \
        (latency_score * 0.30) + (cost_score * 0.15)
    return total, adjusted_reliability, latency_ms, est_cost, request_count


def _meets_contract(
    reliability: float,
    latency_ms: int,
    est_cost: float,
    contract: RequestContract,
) -> bool:
    return (
        reliability >= contract.min_reliability
        and latency_ms <= contract.max_latency_ms
        and est_cost <= contract.max_cost_usd
    )


def decide_strategy(
    request: RequestContract,
    providers_state: list[dict[str, Any]],
) -> DecisionResult:
    candidates: list[dict[str, Any]] = []
    rejected_providers: list[dict[str, str]] = []

    available_providers = [
        p
        for p in providers_state
        if p.get("status", "failed") not in {"failed", "cooling"}
    ]

    if not available_providers:
        return DecisionResult(
            strategy="reject",
            enforcement="reject",
            reason="No eligible providers are available",
            risk_level="high",
        )

    min_possible_latency = min(_provider_latency_ms(p)
                               for p in available_providers)
    min_possible_cost = min(
        estimate_cost_usd(str(p.get("id", "unknown")),
                          TOKENS_PER_REQUEST_ESTIMATE)
        for p in available_providers
    )

    if request.max_latency_ms < min_possible_latency:
        return DecisionResult(
            strategy="reject",
            enforcement="reject",
            reason=f"Latency constraint impossible: min achievable is {min_possible_latency}ms",
            rejected_providers=[
                {
                    "provider": str(p.get("id", "unknown")),
                    "reason": f"latency {_provider_latency_ms(p)}ms exceeds contract",
                }
                for p in available_providers
            ],
            risk_level="high",
        )

    if request.max_cost_usd < min_possible_cost:
        return DecisionResult(
            strategy="reject",
            enforcement="reject",
            reason=f"Cost constraint impossible: min achievable is ${min_possible_cost:.6f}",
            rejected_providers=[
                {
                    "provider": str(p.get("id", "unknown")),
                    "reason": f"estimated cost ${estimate_cost_usd(str(p.get('id', 'unknown')), TOKENS_PER_REQUEST_ESTIMATE):.6f} exceeds contract",
                }
                for p in available_providers
            ],
            risk_level="high",
        )

    for provider in providers_state:
        status = provider.get("status", "failed")
        if status in {"failed", "cooling"}:
            rejected_providers.append(
                {
                    "provider": str(provider.get("id", "unknown")),
                    "reason": f"provider unavailable ({status})",
                }
            )
            continue

        score, reliability, latency_ms, est_cost, request_count = _provider_score(
            provider, request)
        meets = _meets_contract(reliability, latency_ms, est_cost, request)

        if not meets:
            fail_reasons: list[str] = []
            if reliability < request.min_reliability:
                fail_reasons.append("reliability below threshold")
            if latency_ms > request.max_latency_ms:
                fail_reasons.append("latency too high")
            if est_cost > request.max_cost_usd:
                fail_reasons.append("cost too high")
            rejected_providers.append(
                {
                    "provider": str(provider.get("id", "unknown")),
                    "reason": ", ".join(fail_reasons) if fail_reasons else "contract mismatch",
                }
            )

        candidates.append(
            {
                "id": provider["id"],
                "score": score,
                "reliability": reliability,
                "latency_ms": latency_ms,
                "est_cost": est_cost,
                "meets": meets,
                "request_count": request_count,
            }
        )

    if not candidates:
        return DecisionResult(
            strategy="reject",
            enforcement="reject",
            reason="No eligible providers are available",
            rejected_providers=rejected_providers,
            risk_level="high",
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    second: dict[str, Any] | None = candidates[1] if len(
        candidates) > 1 else None

    # Fulfill with single provider when top candidate already satisfies contract.
    if best["meets"]:
        disable_hedging = False
        if second is not None and (
            request.min_reliability >= 0.97 or request.max_latency_ms <= 1200
        ):
            second_provider = cast(dict[str, Any], second)
            second_reliability = float(
                second_provider.get("reliability", 0.0) or 0.0)
            second_cost = float(second_provider.get("est_cost", 0.0) or 0.0)
            second_latency = int(second_provider.get("latency_ms", 0) or 0)
            second_id = str(second_provider.get("id", "unknown"))
            combined_rel = 1 - (1 - best["reliability"]) * (
                1 - second_reliability
            )
            hedged_cost = best["est_cost"] + second_cost
            if best["est_cost"] * 2 > request.max_cost_usd:
                disable_hedging = True

            if disable_hedging:
                rejected_providers.append(
                    {
                        "provider": second_id,
                        "reason": "hedging disabled: estimated_cost * 2 exceeds contract max_cost",
                    }
                )

            if combined_rel >= request.min_reliability and hedged_cost <= request.max_cost_usd:
                if not disable_hedging:
                    return DecisionResult(
                        strategy="hedged",
                        enforcement="fulfill",
                        selected_providers=[best["id"], second_id],
                        rejected_providers=rejected_providers,
                        reason="High criticality contract: hedged execution selected",
                        risk_level=_risk_level(
                            request.min_reliability,
                            combined_rel,
                            request.max_latency_ms,
                            min(best["latency_ms"], second_latency),
                        ),
                        expected_reliability=round(combined_rel, 4),
                        expected_latency_ms=min(
                            best["latency_ms"], second_latency),
                        estimated_cost_usd=round(hedged_cost, 6),
                    )

        return DecisionResult(
            strategy="single",
            enforcement="fulfill",
            selected_providers=[best["id"]],
            fallback_providers=[p["id"] for p in candidates[1:3]],
            rejected_providers=rejected_providers,
            reason="Top provider satisfies contract",
            risk_level=_risk_level(
                request.min_reliability,
                best["reliability"],
                request.max_latency_ms,
                best["latency_ms"],
            ),
            expected_reliability=round(best["reliability"], 4),
            expected_latency_ms=best["latency_ms"],
            estimated_cost_usd=round(best["est_cost"], 6),
        )

    # Try hedging to satisfy reliability when single route cannot.
    if second is not None:
        second_provider = cast(dict[str, Any], second)
        second_reliability = float(
            second_provider.get("reliability", 0.0) or 0.0)
        second_cost = float(second_provider.get("est_cost", 0.0) or 0.0)
        second_latency = int(second_provider.get("latency_ms", 0) or 0)
        second_id = str(second_provider.get("id", "unknown"))
        combined_rel = 1 - (1 - best["reliability"]) * (
            1 - second_reliability
        )
        hedged_cost = best["est_cost"] + second_cost
        hedged_latency = min(best["latency_ms"], second_latency)
        disable_hedging = best["est_cost"] * 2 > request.max_cost_usd
        if disable_hedging:
            rejected_providers.append(
                {
                    "provider": second_id,
                    "reason": "hedging disabled: estimated_cost * 2 exceeds contract max_cost",
                }
            )

        if (
            combined_rel >= request.min_reliability
            and hedged_latency <= request.max_latency_ms
            and hedged_cost <= request.max_cost_usd
            and not disable_hedging
        ):
            return DecisionResult(
                strategy="hedged",
                enforcement="fulfill",
                selected_providers=[best["id"], second_id],
                rejected_providers=rejected_providers,
                reason="Single route insufficient; hedged route fulfills contract",
                risk_level=_risk_level(
                    request.min_reliability,
                    combined_rel,
                    request.max_latency_ms,
                    hedged_latency,
                ),
                expected_reliability=round(combined_rel, 4),
                expected_latency_ms=hedged_latency,
                estimated_cost_usd=round(hedged_cost, 6),
            )

    if request.allow_degrade:
        degrade_reason_parts: list[str] = []
        if best["latency_ms"] > request.max_latency_ms:
            degrade_reason_parts.append(
                f"No provider met latency < {request.max_latency_ms}ms"
            )
        if best["est_cost"] > request.max_cost_usd:
            degrade_reason_parts.append(
                f"No provider met cost <= ${request.max_cost_usd:.6f}"
            )
        if best["reliability"] < request.min_reliability:
            degrade_reason_parts.append(
                f"No provider met reliability >= {request.min_reliability:.3f}"
            )
        degrade_reason = "; ".join(
            degrade_reason_parts) or "No provider met full contract"

        return DecisionResult(
            strategy="single",
            enforcement="degrade",
            selected_providers=[best["id"]],
            fallback_providers=[p["id"] for p in candidates[1:3]],
            rejected_providers=rejected_providers,
            reason="No route can fulfill contract; degrade mode allowed",
            risk_level="high",
            degrade_reason=degrade_reason,
            expected_reliability=round(best["reliability"], 4),
            expected_latency_ms=best["latency_ms"],
            estimated_cost_usd=round(best["est_cost"], 6),
        )

    return DecisionResult(
        strategy="reject",
        enforcement="reject",
        reason="Contract cannot be fulfilled by available providers",
        selected_providers=[best["id"]],
        rejected_providers=rejected_providers,
        risk_level="high",
        expected_reliability=round(best["reliability"], 4),
        expected_latency_ms=best["latency_ms"],
        estimated_cost_usd=round(best["est_cost"], 6),
    )


async def execute_single(
    messages: list[dict[str, str]],
    system: str | None,
    provider_slot: str,
    timeout_s: float = 30,
) -> dict[str, Any]:
    full_messages = messages if not system else [
        {"role": "system", "content": system}] + messages
    started = time.monotonic()

    try:
        response = await asyncio.wait_for(
            ai_router.get_router().acompletion(
                model=provider_slot,
                messages=full_messages,
            ),
            timeout=timeout_s,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        model_used = response.model or _expected_model_id(provider_slot)
        expected_model = _expected_model_id(provider_slot)
        tokens = response.usage.total_tokens if response.usage else 0

        return {
            "success": True,
            "content": response.choices[0].message.content,
            "provider_slot": provider_slot,
            "model_used": model_used,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "fallback_triggered": expected_model not in model_used,
            "error": None,
            "error_code": None,
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "content": "",
            "provider_slot": provider_slot,
            "model_used": _expected_model_id(provider_slot),
            "tokens": 0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "fallback_triggered": False,
            "error": f"Provider {provider_slot} timed out",
            "error_code": "provider_timeout",
        }
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        return {
            "success": False,
            "content": "",
            "provider_slot": provider_slot,
            "model_used": _expected_model_id(provider_slot),
            "tokens": 0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "fallback_triggered": False,
            "error": str(exc),
            "error_code": "provider_error",
        }


async def execute_hedged(
    messages: list[dict[str, str]],
    system: str | None,
    providers: list[str],
    timeout_s: float = 30,
) -> dict[str, Any]:
    if not providers:
        return {
            "success": False,
            "content": "",
            "provider_slot": "unknown",
            "model_used": "unknown",
            "tokens": 0,
            "latency_ms": 0,
            "fallback_triggered": False,
            "error": "hedged_no_providers",
            "error_code": "hedged_no_providers",
            "provider_errors": [],
        }

    hard_timeout = max(0.5, float(timeout_s))
    tasks = {
        provider: asyncio.create_task(execute_single(
            messages, system, provider, hard_timeout))
        for provider in providers
    }

    provider_errors: list[dict[str, str]] = []
    started = time.monotonic()
    try:
        done, pending = await asyncio.wait(
            set(tasks.values()),
            timeout=hard_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            for provider, task in tasks.items():
                provider_errors.append(
                    {
                        "provider": provider,
                        "error": "timeout",
                    }
                )
                if not task.done():
                    task.cancel()
            return {
                "success": False,
                "content": "",
                "provider_slot": providers[0],
                "model_used": "unknown",
                "tokens": 0,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "fallback_triggered": False,
                "error": "All hedged providers timed out",
                "error_code": "hedged_timeout",
                "provider_errors": provider_errors,
            }

        for finished in done:
            result = await finished
            if result["success"]:
                for task in pending:
                    task.cancel()
                return result

            provider_errors.append(
                {
                    "provider": str(result.get("provider_slot", "unknown")),
                    "error": str(result.get("error", "unknown_error")),
                }
            )

        # If the first completed task failed, wait for other tasks up to remaining timeout.
        remaining_timeout = max(
            0.1, hard_timeout - (time.monotonic() - started))
        if pending:
            done_late, still_pending = await asyncio.wait(
                pending,
                timeout=remaining_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            for finished in done_late:
                late_result = await finished
                if late_result["success"]:
                    for task in still_pending:
                        task.cancel()
                    return late_result
                provider_errors.append(
                    {
                        "provider": str(late_result.get("provider_slot", "unknown")),
                        "error": str(late_result.get("error", "unknown_error")),
                    }
                )
            for task in still_pending:
                provider = next(
                    (name for name, known_task in tasks.items() if known_task is task),
                    "unknown",
                )
                provider_errors.append(
                    {
                        "provider": provider,
                        "error": "timeout",
                    }
                )
                task.cancel()

        return {
            "success": False,
            "content": "",
            "provider_slot": providers[0],
            "model_used": "unknown",
            "tokens": 0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "fallback_triggered": False,
            "error": "All hedged providers failed",
            "error_code": "hedged_all_failed",
            "provider_errors": provider_errors,
        }
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        # Drain cancelled/failed tasks to avoid Task exception was never retrieved.
        await asyncio.gather(*tasks.values(), return_exceptions=True)
