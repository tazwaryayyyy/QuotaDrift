"""
LiteLLM router — the core of the switchboard.

Wraps the LiteLLM Router to add:
- Async streaming with SSE-ready async generators
- Per-model health tracking (marks used/error on each call)
- Clean model ID extraction from streamed chunks
"""

from functools import lru_cache

from dotenv import load_dotenv
from litellm import Router

from quotadrift import config, model_manager

load_dotenv()


@lru_cache(maxsize=1)
def get_router() -> Router:
    return Router(
        model_list=config.MODEL_LIST,
        fallbacks=config.FALLBACK_CHAIN,
        allowed_fails=2,
        cooldown_time=300,  # 5 min cooldown on rate limit
        retry_after=3,
        num_retries=1,
        set_verbose=False,
    )


async def chat(messages: list[dict], system: str | None = None) -> dict:
    """
    Non-streaming chat with circuit breaker and dynamic model selection.
    Returns: {"content": str, "model_used": str, "tokens": int}
    """
    request_id = model_manager.get_request_id()

    # Get best available model
    slot_name = model_manager.model_manager.get_best_model(request_id)
    if not slot_name:
        raise RuntimeError(
            "No models available - all circuits open or rate limited")

    # Start request tracking
    model_manager.model_manager.start_request(slot_name, request_id)

    full_messages = _prepend_system(messages, system)

    try:
        response = await get_router().acompletion(
            model=slot_name,
            messages=full_messages,
        )

        model_id = response.model or "unknown"
        tokens = response.usage.total_tokens if response.usage else 0

        # Record success
        model_manager.model_manager.record_success(
            slot_name, request_id, tokens)

        # Update rate limits if available
        if hasattr(response, "headers"):
            remaining = response.headers.get("x-ratelimit-remaining")
            reset = response.headers.get("x-ratelimit-reset")
            model_manager.model_manager.update_rate_limit(
                slot_name, remaining, reset)

        return {
            "content": response.choices[0].message.content,
            "model_used": model_id,
            "tokens": tokens,
        }
    except (RuntimeError, ValueError, TypeError, KeyError, OSError, TimeoutError) as exc:
        # Record failure
        model_manager.model_manager.record_failure(
            slot_name, request_id, str(exc))
        raise


async def stream_chat(messages: list[dict], system: str | None = None):
    """
    Streaming chat with circuit breaker and dynamic model selection.
    Async generator yielding dicts:
      {"type": "token",  "content": "..."}
      {"type": "done",   "model": "...", "tokens": 0}
      {"type": "error",  "message": "..."}
    """
    import time

    request_id = model_manager.get_request_id()

    # Get best available model
    slot_name = model_manager.model_manager.get_best_model(request_id)
    if not slot_name:
        yield {
            "type": "error",
            "message": "No models available - all circuits open or rate limited",
        }
        return

    # Start request tracking
    model_manager.model_manager.start_request(slot_name, request_id)

    full_messages = _prepend_system(messages, system)
    model_used = "unknown"
    token_count = 0
    start_time = time.monotonic()
    first_token_logged = False

    try:
        response = await get_router().acompletion(
            model=slot_name,
            messages=full_messages,
            stream=True,
        )

        async for chunk in response:
            # Capture model ID from first chunk that has it
            if hasattr(chunk, "model") and chunk.model:
                model_used = chunk.model

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta and delta.content:
                # TTFT calculation
                if not first_token_logged:
                    ttft_ms = (time.monotonic() - start_time) * 1000
                    first_token_logged = True
                    _update_ttft(slot_name, ttft_ms)

                token_count += 1
                yield {"type": "token", "content": delta.content}

        # Record success and update metrics
        model_manager.model_manager.record_success(
            slot_name, request_id, token_count)

        # Update rate limits if available
        try:
            response_obj = getattr(response, "_response_object", None)
            headers = getattr(response_obj, "headers", None)
            if headers is not None:
                remaining = headers.get(
                    "x-ratelimit-remaining-requests"
                ) or headers.get("x-ratelimit-remaining")
                reset = headers.get("x-ratelimit-reset-requests") or headers.get(
                    "x-ratelimit-reset"
                )
                model_manager.model_manager.update_rate_limit(
                    slot_name, remaining, reset
                )
        except (AttributeError, TypeError, ValueError, KeyError):
            pass

        yield {"type": "done", "model": model_used, "tokens": token_count}

    except (RuntimeError, ValueError, TypeError, KeyError, OSError, TimeoutError) as exc:
        # Record failure
        model_manager.model_manager.record_failure(
            slot_name, request_id, str(exc))
        yield {"type": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update_ttft(slot: str, ttft_ms: float):
    if slot in config.health:
        h = config.health[slot]
        h["ttft_samples"] += 1
        h["ttft_ms_avg"] = ((h["ttft_ms_avg"] * (h["ttft_samples"] - 1)) + ttft_ms) / h[
            "ttft_samples"
        ]


def _update_rate_limits(slot: str, headers: dict):
    # Common header patterns
    remaining = headers.get("x-ratelimit-remaining-requests") or headers.get(
        "x-ratelimit-remaining"
    )
    reset = headers.get("x-ratelimit-reset-requests") or headers.get(
        "x-ratelimit-reset"
    )

    if slot in config.health:
        if remaining is not None:
            config.health[slot]["rl_remaining"] = int(remaining)
        if reset is not None:
            from datetime import datetime, timedelta

            try:
                # reset is often seconds until reset
                config.health[slot]["rl_reset_at"] = (
                    datetime.utcnow() + timedelta(seconds=float(reset))
                ).isoformat()
            except (TypeError, ValueError, OverflowError):
                pass


def _prepend_system(messages: list[dict], system: str | None) -> list[dict]:
    if not system:
        return messages
    return [{"role": "system", "content": system}] + messages


def _model_to_slot(model_id: str) -> str:
    """Map a litellm model string back to our slot name."""
    for slot in config.MODEL_LIST:
        if slot["litellm_params"]["model"] in model_id:
            return slot["model_name"]
    return "primary"


def _try_mark_error(err_msg: str):
    """Best-effort: mark whichever slot looks rate-limited."""
    # LiteLLM surfaces 429s with the model name in the exception message
    for slot in config.MODEL_LIST:
        if slot["litellm_params"]["model"] in err_msg:
            config.mark_error(slot["model_name"])
            return
    config.mark_error("primary")
