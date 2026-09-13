"""
Central config. Flip LLM_MODE to "live" once you export ANTHROPIC_API_KEY
(or adapt llm_client.py for a different provider — OpenAI, local model, etc.
The classifier/reply-gen/judge code doesn't care which provider, it only
calls llm_client.complete()).
"""

import os

# "mock" (default, offline, deterministic) or "live" (real API calls)
LLM_MODE = os.environ.get("LLM_MODE", "mock")

# Only used when LLM_MODE == "live"
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

# Confidence threshold below which we escalate regardless of intent
# (see DECISION_LOG.md #7 for how this number was picked)
LOW_CONFIDENCE_ESCALATION_THRESHOLD = 0.55

# How many similar historical resolutions to retrieve for grounding
RETRIEVAL_TOP_K = 3

BRAND_NAME = "SpotifyCares"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_TWEETS_PATH = os.path.join(DATA_DIR, "raw", "spotifycares_support_tweets.csv")
HISTORICAL_RESOLUTIONS_PATH = os.path.join(DATA_DIR, "historical_resolutions.json")

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval")
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.csv")
