"""
Decides auto-handle vs escalate-to-human, with a stated reason — required
by the assignment ("with a stated reason").

This is deliberately a small set of transparent, auditable rules layered on
top of the classifier output, NOT another LLM call. Reasoning in
DECISION_LOG.md #8: an opaque LLM "should I escalate?" call is harder to
audit and harder to tune than an explicit rule list, and escalation is a
high-stakes, low-volume decision where auditability beats marginal recall.
"""

from dataclasses import dataclass

from src.intents import INTENT_BY_KEY
from src import config

# Phrases that always force escalation regardless of intent — legal/safety
# language a human must see. Kept short and reviewed, not a giant blocklist.
HARD_ESCALATION_PHRASES = [
    "lawsuit", "legal action", "bbb", "regulator", "sue you",
    "attorney", "class action", "unsafe", "fire hazard", "injury",
]


@dataclass
class Decision:
    action: str  # "auto_handle" | "escalate"
    reason: str


class DecisionEngine:
    def decide(self, message: str, intent_key: str, confidence: float, repeat_contact: bool = False) -> Decision:
        text = message.lower()

        for phrase in HARD_ESCALATION_PHRASES:
            if phrase in text:
                return Decision("escalate", f"Message contains high-risk language ('{phrase}') requiring human review.")

        if confidence < config.LOW_CONFIDENCE_ESCALATION_THRESHOLD:
            return Decision(
                "escalate",
                f"Classifier confidence ({confidence:.2f}) below threshold "
                f"({config.LOW_CONFIDENCE_ESCALATION_THRESHOLD}); routing to human rather than guessing.",
            )

        if repeat_contact:
            return Decision(
                "escalate",
                "Customer has already followed up negatively on this thread once; "
                "a second automated reply risks compounding frustration.",
            )

        intent = INTENT_BY_KEY.get(intent_key)
        if intent and intent.default_escalate:
            return Decision(
                "escalate",
                f"Intent '{intent_key}' involves money leaving the company or "
                f"active customer frustration — requires human sign-off by policy.",
            )

        return Decision(
            "auto_handle",
            f"Intent '{intent_key}' classified with confidence {confidence:.2f}, "
            f"no high-risk language, no repeat contact — safe to auto-handle.",
        )
