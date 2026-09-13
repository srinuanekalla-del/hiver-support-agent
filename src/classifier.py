"""
Intent classification. Two implementations:

- LLMClassifier: prompts the model with the taxonomy + a few labeled
  examples (few-shot), asks for intent + a self-reported confidence.
  Self-reported LLM confidence is a known-weak signal (see REPORT.md,
  "What is misleading about my headline number?") — we only use it as a
  cheap escalation trigger, not as a calibrated probability.

- KeywordBaseline: simple, transparent, "simple baseline" for the eval
  harness comparison required by the assignment.
"""

import json
import re
from dataclasses import dataclass

from src import llm_client
from src.intents import INTENT_KEYS, intent_list_for_prompt

FEW_SHOT_EXAMPLES = """
Example 1:
Message: "@PixelWaveHelp cant log in, reset link just 404s"
Intent: account_login_issue

Example 2:
Message: "@PixelWaveHelp charged me twice this month, fix it"
Intent: billing_payment_issue

Example 3:
Message: "@PixelWaveHelp this is the third time i've tweeted about this, unacceptable"
Intent: complaint_general

Example 4:
Message: "@PixelWaveHelp does the mini support dolby atmos?"
Intent: product_information_request
"""

SYSTEM_PROMPT = f"""You are an intent classifier for a customer support system.
Classify the customer's message into exactly one of these intents:

{intent_list_for_prompt()}

Respond with ONLY the intent key on the first line, and a confidence score
from 0.0 to 1.0 on the second line, nothing else. Example output:
device_technical_issue
0.87
"""


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    raw_output: str


class LLMClassifier:
    def classify(self, message: str) -> ClassificationResult:
        prompt = f"{FEW_SHOT_EXAMPLES}\n\nNow classify this message:\nMessage: \"{message}\"\nIntent key:"
        resp = llm_client.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=20)
        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]

        intent = lines[0] if lines else ""
        intent = re.sub(r"[^a-z_]", "", intent.lower())
        if intent not in INTENT_KEYS:
            # fall back: substring match against known keys
            intent = next((k for k in INTENT_KEYS if k in intent), "device_technical_issue")

        confidence = 0.6  # default if model didn't give one (mock mode)
        if len(lines) > 1:
            try:
                confidence = float(re.sub(r"[^0-9.]", "", lines[1]))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        return ClassificationResult(intent=intent, confidence=confidence, raw_output=resp.text)


class KeywordBaseline:
    """Simple baseline for comparison — required by the assignment as
    "a simple one" (see REPORT.md, Results vs. baselines)."""

    KEYWORD_MAP = {
        "account_login_issue": ["password", "log in", "login", "locked", "2fa", "verify"],
        "billing_payment_issue": ["charged", "charge", "invoice", "billing", "subscription"],
        "device_technical_issue": ["turn on", "not working", "buffering", "crash", "error code", "connect", "freeze"],
        "shipping_delivery_issue": ["delivery", "shipping", "tracking", "damaged", "wrong item", "arrived"],
        "product_information_request": ["support", "compatible", "spec", "when will", "available", "does it"],
        "cancellation_refund_request": ["cancel", "refund", "money back"],
        "complaint_general": ["unacceptable", "worst", "third time", "ridiculous", "lawsuit", "bbb"],
        "positive_feedback": ["thank", "thanks", "love", "amazing", "great"],
    }

    def classify(self, message: str) -> ClassificationResult:
        text = message.lower()
        for intent, kws in self.KEYWORD_MAP.items():
            if any(kw in text for kw in kws):
                return ClassificationResult(intent=intent, confidence=0.5, raw_output="keyword-match")
        return ClassificationResult(intent="device_technical_issue", confidence=0.2, raw_output="default-fallback")


class TrivialBaseline:
    """Always predicts the single most common class in the training data.
    This is the "trivial baseline" the assignment explicitly asks for."""

    def __init__(self, most_common_intent: str = "device_technical_issue"):
        self.most_common_intent = most_common_intent

    def classify(self, message: str) -> ClassificationResult:
        return ClassificationResult(intent=self.most_common_intent, confidence=1.0, raw_output="trivial")
