"""
Intent taxonomy for PixelWave (the brand we picked — see README for why the
brand is synthetic).

Why 8 intents and not more: I skimmed ~150 raw synthetic threads before
freezing this list. Picked the cut-off where adding another intent stopped
changing the downstream *action* (reply template + auto/escalate decision).
See DECISION_LOG.md #1 for the full reasoning and what I merged.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    key: str
    label: str
    description: str
    default_escalate: bool  # prior; decision_engine can override per-message


INTENTS: list[Intent] = [
    Intent(
        key="account_login_issue",
        label="Account / Login Issue",
        description=(
            "Customer cannot log in, reset password, verify 2FA, or their "
            "account was locked/suspended."
        ),
        default_escalate=False,
    ),
    Intent(
        key="billing_payment_issue",
        label="Billing / Payment Issue",
        description=(
            "Wrong charge, duplicate charge, failed payment, subscription "
            "price question, invoice request."
        ),
        default_escalate=False,
    ),
    Intent(
        key="device_technical_issue",
        label="Device / Technical Issue",
        description=(
            "Product not working as expected: won't turn on, won't connect, "
            "crashes, buffering, firmware issue, error codes."
        ),
        default_escalate=False,
    ),
    Intent(
        key="shipping_delivery_issue",
        label="Shipping / Delivery Issue",
        description=(
            "Order delayed, lost, damaged in transit, wrong item shipped, "
            "tracking questions."
        ),
        default_escalate=False,
    ),
    Intent(
        key="product_information_request",
        label="Product Information Request",
        description=(
            "Pre-purchase or general questions about specs, compatibility, "
            "availability, how a feature works."
        ),
        default_escalate=False,
    ),
    Intent(
        key="cancellation_refund_request",
        label="Cancellation / Refund Request",
        description=(
            "Customer explicitly asks to cancel an order/subscription or "
            "wants a refund."
        ),
        default_escalate=True,  # money leaving the company -> human sign-off
    ),
    Intent(
        key="complaint_general",
        label="General Complaint / Escalated Frustration",
        description=(
            "Venting about a repeated bad experience, threatening to leave, "
            "legal/regulatory language, or repeat contact on the same issue."
        ),
        default_escalate=True,
    ),
    Intent(
        key="positive_feedback",
        label="Positive Feedback / Praise",
        description="Compliment, thank-you, or positive review-style tweet.",
        default_escalate=False,
    ),
]

INTENT_KEYS = [i.key for i in INTENTS]
INTENT_BY_KEY = {i.key: i for i in INTENTS}


def intent_list_for_prompt() -> str:
    """Render the taxonomy as a numbered list for the classifier prompt."""
    lines = []
    for i, intent in enumerate(INTENTS, start=1):
        lines.append(f"{i}. {intent.key} — {intent.description}")
    return "\n".join(lines)
