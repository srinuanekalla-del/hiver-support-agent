"""
Drafts a reply grounded in retrieved historical resolutions for the same
brand. The prompt explicitly instructs the model to only use info present in
the retrieved examples or to ask a clarifying question rather than invent
policy (refund amounts, delivery timelines, etc.) it wasn't shown.
"""

from dataclasses import dataclass

from src import llm_client
from src.retriever import RetrievedResolution

SYSTEM_PROMPT = """You are a customer support agent for PixelWave, a
consumer electronics brand. Write a short (1-3 sentence) public reply to the
customer's tweet, in a warm but efficient tone matching the brand voice
shown in the examples.

Rules:
- Ground your reply in the example resolutions provided. Do not invent
  specific policies, refund amounts, or timelines that aren't implied by
  the examples.
- If the retrieved examples don't clearly cover this case, write a reply
  that acknowledges the issue and asks the customer to DM order/account
  details, rather than guessing at a fix.
- Never promise something (like a full refund) unless the examples show
  that's the standard resolution for this type of issue.
"""


@dataclass
class DraftReply:
    text: str
    grounding_used: list[RetrievedResolution]


class ReplyGenerator:
    def generate(self, customer_message: str, retrieved: list[RetrievedResolution]) -> DraftReply:
        if retrieved:
            examples_block = "\n\n".join(
                f"Past customer message: \"{r.customer_text}\"\nPast agent reply: \"{r.agent_reply}\""
                for r in retrieved
            )
        else:
            examples_block = "(No similar past resolutions found — acknowledge and ask for details.)"

        prompt = (
            f"Similar past resolutions:\n{examples_block}\n\n"
            f"New customer message: \"{customer_message}\"\n\n"
            f"Write the reply:"
        )
        resp = llm_client.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=200)
        return DraftReply(text=resp.text.strip(), grounding_used=retrieved)
