"""
Reproduces the headline results in under 15 minutes with zero setup:

    python scripts/run_demo.py

Runs the pipeline (mock LLM mode by default — no API key needed) on a
handful of illustrative messages, then points you at the real eval command.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgentPipeline

SAMPLE_MESSAGES = [
    "@SpotifyCares my premium subscription got charged twice this month, can you fix this",
    "@SpotifyCares i want to cancel my subscription immediately",
    "@SpotifyCares this is the THIRD time i've tweeted about this, unacceptable, considering legal action",
    "@SpotifyCares does premium let you download podcasts for offline listening?",
    "just wanna say @SpotifyCares support fixed my issue in like 5 mins, amazing",
]


def main():
    print(f"LLM_MODE={os.environ.get('LLM_MODE', 'mock')} "
          f"(set ANTHROPIC_API_KEY and LLM_MODE=live for real model output)\n")

    pipeline = SupportAgentPipeline()

    for msg in SAMPLE_MESSAGES:
        result = pipeline.process(msg)
        print("=" * 80)
        print(f"CUSTOMER: {msg}")
        print(f"INTENT:   {result.classification.intent} (confidence {result.classification.confidence:.2f})")
        print(f"DECISION: {result.decision.action.upper()} — {result.decision.reason}")
        print(f"DRAFT:    {result.draft_reply.text}")
        if result.retrieved:
            print(f"GROUNDED ON {len(result.retrieved)} past resolution(s), "
                  f"top similarity={result.retrieved[0].similarity:.2f}")
        print()

    print("=" * 80)
    print("Demo complete. For full evaluation against the golden set, run:")
    print("    python eval/run_eval.py")


if __name__ == "__main__":
    main()
