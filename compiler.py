"""
Context compiler for seamless model handoff.

When a model switch happens (quota exhausted, rate limited, etc.),
the compiler asks the current model to serialize the full session
state into a structured JSON artifact. That artifact is injected
into the new model's system prompt so it can continue mid-task
without the user needing to re-explain anything.
"""

import json
import re

COMPILE_SYSTEM = """You are a session state compiler. Your only job is to output a compact JSON object that captures the complete state of the coding session so a different AI model can continue it instantly.

Output ONLY valid JSON. No markdown, no explanation, no code fences.

Required fields:
{
  "project_goal":      "one sentence — what the user is ultimately building",
  "tech_stack":        ["list", "of", "technologies", "mentioned"],
  "current_task":      "what was being actively worked on at the moment of this call",
  "last_action":       "exact last thing discussed, generated, or decided",
  "key_decisions":     ["decision 1", "decision 2"],
  "known_constraints": ["constraint 1", "constraint 2"],
  "open_questions":    ["anything unresolved or pending"],
  "working_code":      "paste any important code snippets that were generated (or null)",
  "errors_seen":       ["any error messages encountered"],
  "next_steps":        ["what the user will likely ask next"]
}"""


COMPILE_TRIGGER = (
    "Compile the complete session state into the required JSON format. "
    "Be precise — the next model will have zero prior context."
)


HANDOFF_TEMPLATE = """You are continuing a coding session handed off from another AI model. The previous model compiled the following state before switching:

{state_json}

Instructions:
- Act as if you have been present in this conversation from the start
- Do NOT ask for context already captured above
- Continue exactly from "current_task" and "last_action"
- If "working_code" is present, treat it as code you already wrote
- Your first response should be natural continuation, not a summary of what you read"""


def build_handoff_system(state: dict) -> str:
    return HANDOFF_TEMPLATE.format(
        state_json=json.dumps(state, indent=2, ensure_ascii=False)
    )


async def compile_state(messages: list[dict], chat_fn) -> dict:
    """
    Ask the current LLM to serialize its session state.

    chat_fn: async callable that accepts (messages, system) and returns
             {"content": str, "model_used": str, ...}
    """
    result = await chat_fn(
        messages=messages + [{"role": "user", "content": COMPILE_TRIGGER}],
        system=COMPILE_SYSTEM,
    )

    raw = result.get("content", "")

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Graceful degradation — return whatever we got as a raw summary
        return {
            "project_goal":  "Could not parse structured state",
            "raw_summary":   raw,
            "current_task":  "Unknown — see raw_summary",
            "last_action":   "Unknown — see raw_summary",
        }
