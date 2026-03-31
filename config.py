"""
Model registry and health tracking for the Multi-AI Switchboard.

Priority chain (cloud-only, no GPU required):
  1. Groq          — fastest free inference (Llama 3.3 70B)
  2. GitHub Models  — GPT-4o mini (reliable, good quality)
  3. GitHub Models  — Llama 3.3 70B (same model, different provider)
  4. Mistral AI     — Mistral Small (1B tokens/month free)
  5. Silicon Flow   — Qwen2.5-7B (20M tokens free)
  6. Hugging Face   — Mistral-7B (free inference API)
  7. Cloudflare AI  — Llama 3.3 70B (10k requests/day)
  8. OpenRouter     — Mistral 7B free (last resort, always available)
"""

from datetime import datetime
from typing import Literal

# Model display names for the UI
MODEL_DISPLAY = {
    "groq/llama-3.3-70b-versatile":             "Groq · Llama 3.3 70B",
    "openai/gpt-4o-mini":                        "GitHub · GPT-4o mini",
    "openai/meta-llama-3.3-70b-instruct":        "GitHub · Llama 3.3 70B",
    "mistral/mistral-small-latest":              "Mistral · Mistral Small",
    "openai/Qwen2.5-7B-Instruct":                "Silicon Flow · Qwen2.5-7B",
    "huggingface/mistralai/Mistral-7B-Instruct-v0.3": "Hugging Face · Mistral-7B",
    "cloudflare/@cf/meta/llama-3.3-70b-instruct": "Cloudflare · Llama 3.3 70B",
    "openrouter/mistralai/mistral-7b-instruct:free": "OpenRouter · Mistral 7B",
}

MODEL_COLORS = {
    "groq/llama-3.3-70b-versatile":             "#f97316",  # orange
    "openai/gpt-4o-mini":                        "#38bdf8",  # sky
    "openai/meta-llama-3.3-70b-instruct":        "#818cf8",  # indigo
    "mistral/mistral-small-latest":              "#ea580c",  # amber
    "openai/Qwen2.5-7B-Instruct":                "#0d9488",  # teal
    "huggingface/mistralai/Mistral-7B-Instruct-v0.3": "#7c3aed",  # purple
    "cloudflare/@cf/meta/llama-3.3-70b-instruct": "#f59e0b",  # yellow
    "openrouter/mistralai/mistral-7b-instruct:free": "#a78bfa",  # violet
}

MODEL_LIST = [
    {
        "model_name": "primary",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile",
            "api_key": "os.environ/GROQ_API_KEY",
        },
    },
    {
        "model_name": "secondary",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_base": "https://models.inference.ai.azure.com",
            "api_key": "os.environ/GITHUB_TOKEN",
        },
    },
    {
        "model_name": "tertiary",
        "litellm_params": {
            "model": "openai/meta-llama-3.3-70b-instruct",
            "api_base": "https://models.inference.ai.azure.com",
            "api_key": "os.environ/GITHUB_TOKEN",
        },
    },
    {
        "model_name": "quaternary",
        "litellm_params": {
            "model": "mistral/mistral-small-latest",
            "api_key": "os.environ/MISTRAL_API_KEY",
        },
    },
    {
        "model_name": "siliconflow",
        "litellm_params": {
            "model": "openai/Qwen2.5-7B-Instruct",
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "os.environ/SILICONFLOW_API_KEY",
            "custom_llm_provider": "openai",
            "drop_params": True  # Drop unsupported params for OpenAI-compatible
        },
    },
    {
        "model_name": "huggingface",
        "litellm_params": {
            "model": "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
            "api_key": "os.environ/HUGGINGFACE_API_KEY",
            "api_base": "https://api-inference.huggingface.co/v1"
        },
    },
    {
        "model_name": "cloudflare",
        "litellm_params": {
            "model": "cloudflare/@cf/meta/llama-3.3-70b-instruct",
            "api_key": "os.environ/CLOUDFLARE_API_KEY",
            "api_base": "https://api.cloudflare.com/client/v4/accounts/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/ai/run"
        },
    },
    {
        "model_name": "fallback",
        "litellm_params": {
            "model": "openrouter/mistralai/mistral-7b-instruct:free",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
]

FALLBACK_CHAIN = [
    {"primary": ["secondary", "tertiary", "quaternary", "siliconflow", "huggingface", "cloudflare", "fallback"]}
]

# In-memory health state — reset on restart (fine for personal use)
ModelStatus = Literal["available", "cooling", "failed", "untested"]

health: dict[str, dict] = {
    slot["model_name"]: {
        "model_id": slot["litellm_params"]["model"],
        "display":  MODEL_DISPLAY.get(slot["litellm_params"]["model"], slot["litellm_params"]["model"]),
        "color":    MODEL_COLORS.get(slot["litellm_params"]["model"], "#94a3b8"),
        "status":   "untested",
        "requests": 0,
        "errors":   0,
        "last_used": None,
        "cooldown_until": None,
        # Observability
        "ttft_ms_avg": 0,
        "ttft_samples": 0,
        "rl_remaining": None,   # x-ratelimit-remaining
        "rl_reset_at":  None,   # iso timestamp
    }
    for slot in MODEL_LIST
}


def mark_used(model_name: str):
    if model_name in health:
        health[model_name]["status"]   = "available"
        health[model_name]["requests"] += 1
        health[model_name]["last_used"] = datetime.utcnow().isoformat()


def mark_error(model_name: str, cooldown_secs: int = 300):
    if model_name in health:
        health[model_name]["errors"] += 1
        health[model_name]["status"]  = "cooling"
        from datetime import timedelta
        health[model_name]["cooldown_until"] = (
            datetime.utcnow() + timedelta(seconds=cooldown_secs)
        ).isoformat()


def get_health_snapshot() -> list[dict]:
    now = datetime.utcnow()
    snapshot = []
    for name, info in health.items():
        entry = dict(info, slot=name)
        # Auto-recover from cooldown
        if info["cooldown_until"]:
            cd = datetime.fromisoformat(info["cooldown_until"])
            if now >= cd:
                entry["status"]         = "available"
                entry["cooldown_until"] = None
                health[name]["status"]         = "available"
                health[name]["cooldown_until"] = None
        snapshot.append(entry)
    return snapshot
