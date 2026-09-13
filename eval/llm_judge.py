"""
LLM-as-judge for reply quality, plus the human-agreement check the
assignment requires ("including evidence of how well your judge agrees
with a human").

Judge rubric (1-5): grounded in retrieved history, on-topic, appropriate
tone, would-you-send-this-as-is. Kept small on purpose — a sprawling rubric
is harder for a human to re-score consistently, which defeats the point of
checking agreement.
"""

import json
from dataclasses import dataclass

from src import llm_client

JUDGE_SYSTEM_PROMPT = """You are grading a customer support reply for
quality. Score it 1-5 where:
1 = irrelevant or would upset the customer further
3 = acceptable but generic, doesn't fully address the message
5 = specific, grounded, appropriate tone, ready to send as-is

Respond with ONLY a JSON object (no markdown fences) with keys:
grounded (bool), on_topic (bool), tone_appropriate (bool),
score_1_to_5 (int), rationale (string, one sentence).
"""


@dataclass
class JudgeResult:
    score: int
    grounded: bool
    on_topic: bool
    tone_appropriate: bool
    rationale: str


def judge_reply(customer_message: str, draft_reply: str) -> JudgeResult:
    prompt = f'Customer message: "{customer_message}"\nDraft reply: "{draft_reply}"\n\nJSON score:'
    resp = llm_client.complete(system=JUDGE_SYSTEM_PROMPT, prompt=prompt, max_tokens=200)
    try:
        data = json.loads(resp.text.strip().strip("`").replace("json\n", ""))
    except json.JSONDecodeError:
        data = {"grounded": False, "on_topic": False, "tone_appropriate": False,
                "score_1_to_5": 1, "rationale": "judge output failed to parse"}
    return JudgeResult(
        score=int(data.get("score_1_to_5", 1)),
        grounded=bool(data.get("grounded", False)),
        on_topic=bool(data.get("on_topic", False)),
        tone_appropriate=bool(data.get("tone_appropriate", False)),
        rationale=data.get("rationale", ""),
    )


def judge_human_agreement(judge_scores: list[int], human_scores: list[int]) -> dict:
    """
    Reports exact-match rate and within-1-point agreement rate between the
    LLM judge and a human rater on the same set of replies.

    IMPORTANT (see REPORT.md): in mock mode, `human_scores` below are stand-
    ins I entered myself while reviewing a sample of drafts, not a separate
    blind rater — a real submission should use a second, independent human
    rater blind to the judge's scores to avoid confirmation bias.
    """
    n = len(judge_scores)
    exact = sum(1 for j, h in zip(judge_scores, human_scores) if j == h) / n if n else 0.0
    within_one = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 1) / n if n else 0.0
    return {"n": n, "exact_match_rate": exact, "within_one_point_rate": within_one}
