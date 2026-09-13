"""
Thin wrapper around an LLM API call. Supports two modes:

- "live": calls Anthropic's Messages API using ANTHROPIC_API_KEY from env.
- "mock": deterministic, offline, keyword-driven stand-in used for the demo
  and for CI, so `python scripts/run_demo.py` works with zero API keys and
  under the 15-minute reproduction budget (see README).

Swap MODE in src/config.py once you have a key. The mock is NOT a toy for
show — it's simple keyword matching, and the eval harness's whole point is
to show you the *gap* between mock-level quality and live-LLM quality
(see REPORT.md, "Results vs. baselines").
"""

import os
import json
import time
from dataclasses import dataclass

from src import config


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None


def _mock_complete(system: str, prompt: str) -> LLMResponse:
    """
    Extremely small offline stand-in. It does NOT try to be a good model —
    it exists so the pipeline is runnable without network access / API keys.
    Real quality numbers in REPORT.md come from live-mode runs.
    """
    # --- classification-style prompts: return the first intent keyword seen
    if "respond with only the intent key" in system.lower() or "intent key" in prompt.lower():
        from src.intents import INTENT_KEYS

        # Only scan the actual customer message (last "Message: \"...\"" block),
        # not the few-shot examples baked into the prompt above it — otherwise
        # keywords from the examples leak into the match.
        marker = 'Message: "'
        last_idx = prompt.rfind(marker)
        text = prompt[last_idx:].lower() if last_idx != -1 else prompt.lower()

        keyword_map = {
            "account_login_issue": ["password", "log in", "login", "locked", "2fa", "verify"],
            "billing_payment_issue": ["charged", "charge", "invoice", "refund my card", "billing", "subscription price"],
            "device_technical_issue": ["won't turn on", "not working", "buffering", "crash", "error code", "won't connect", "freezes"],
            "shipping_delivery_issue": ["delivery", "shipping", "tracking", "arrived damaged", "hasn't arrived", "wrong item"],
            "product_information_request": ["does it support", "is it compatible", "spec", "when will", "available in"],
            "cancellation_refund_request": ["cancel my", "want a refund", "cancel subscription", "money back"],
            "complaint_general": ["unacceptable", "worst", "third time", "again", "never buying", "lawsuit", "regulator"],
            "positive_feedback": ["thank you", "thanks", "love my", "great support", "amazing"],
        }
        for intent, kws in keyword_map.items():
            if any(kw in text for kw in kws):
                return LLMResponse(text=intent)
        return LLMResponse(text="device_technical_issue")  # fallback default

    # --- judge-style prompts: return a plausible JSON score
    if "score" in prompt.lower() and "json" in system.lower():
        return LLMResponse(
            text=json.dumps(
                {
                    "grounded": True,
                    "on_topic": True,
                    "tone_appropriate": True,
                    "score_1_to_5": 3,
                    "rationale": "mock-judge: heuristic pass, not a real quality signal",
                }
            )
        )

    # --- reply drafting: template-y fallback
    return LLMResponse(
        text=(
            "Hi, thanks for reaching out — sorry for the trouble. "
            "We're looking into this for you now and will follow up "
            "shortly with next steps. [mock reply — enable live mode for a real draft]"
        )
    )


def complete(system: str, prompt: str, max_tokens: int = 500) -> LLMResponse:
    if config.LLM_MODE == "mock":
        time.sleep(0.01)  # simulate latency so timing code paths are exercised
        return _mock_complete(system, prompt)

    if config.LLM_MODE == "live":
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic` "
                "or set LLM_MODE=mock in src/config.py"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or switch LLM_MODE to 'mock'."
            )

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, raw=resp.model_dump())

    raise ValueError(f"Unknown LLM_MODE: {config.LLM_MODE}")
