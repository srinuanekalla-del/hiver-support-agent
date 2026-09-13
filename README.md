# PixelWave AI Support Agent — Hiver Take-Home

An AI support agent for **PixelWave** (a consumer electronics brand) that
classifies incoming customer messages into intents, drafts a reply grounded
in how PixelWave has historically resolved similar issues, and decides
whether the message can be auto-handled or needs a human — with a stated
reason for that decision.

## ⚠️ Read this first: about the dataset

This environment has no network access to `kaggle.com` or
`huggingface.co`, so I could not download the real
`thoughtvector/customer-support-on-twitter` dataset specified in the brief.

Rather than submit a pipeline that only works against a file nobody can
fetch, I generated a **synthetic dataset in the exact same schema** the real
Kaggle dataset uses (`tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id`) for one fictional brand,
**PixelWave** (see `data/generate_synthetic_dataset.py` — the generation
logic and injected messiness, e.g. unresolved threads, repeat contacts,
typos, are documented inline).

**This is a substitution of the data source, not of the engineering
problem.** Every component — classifier, retriever, decision engine, eval
harness, golden set — is built against the real dataset's schema and would
run unmodified against the real Kaggle CSV filtered to one brand; you'd
just point `src/config.py:RAW_TWEETS_PATH` at it. See `DECISION_LOG.md #2`.

If you'd like me to re-run this against the real dataset live (e.g. on a
call), I'm glad to — it's a data-loading change, not an architecture change.

## What "good" means for this system (short version — full version in REPORT.md)

A reply is good if it's (1) grounded in what PixelWave actually told past
customers, not invented policy, and (2) the routing decision doesn't
auto-handle something that needed a human. I weight **false auto-handles**
(escalate-worthy message wrongly auto-handled) far more heavily than
over-escalation, because the failure mode of "bot promised something it
shouldn't have" or "bot ignored a furious repeat customer" is reputationally
expensive in a way that "human reviewed something routine" is not.

## Project layout

```
data/
  generate_synthetic_dataset.py   # builds the synthetic Kaggle-schema CSV
  raw/pixelwave_support_tweets.csv
src/
  intents.py           # the 8-intent taxonomy + rationale
  data_loader.py        # reconstructs threads from the raw tweet CSV
  classifier.py         # LLMClassifier + KeywordBaseline + TrivialBaseline
  retriever.py           # TF-IDF retrieval over resolved historical threads
  reply_generator.py     # drafts a reply grounded in retrieved history
  decision_engine.py     # auto-handle vs escalate, with a stated reason
  pipeline.py             # wires the above into one SupportAgentPipeline
  llm_client.py           # Anthropic API wrapper + offline mock mode
  config.py
eval/
  golden_set.csv          # 60 hand-labeled examples (see note below)
  build_golden_set.py     # stratified-sampling helper for hand-labeling more
  metrics.py               # classification + decision metrics
  llm_judge.py              # LLM-as-judge for reply quality + human-agreement check
  run_eval.py                # runs everything end-to-end, prints the report
baselines/
  baselines.py               # re-exports the two required baselines
scripts/
  run_demo.py                 # quick illustrative run, no API key needed
tests/
  test_pipeline.py
REPORT.md
DECISION_LOG.md
```

## Setup & reproduction (under 15 minutes, no API key required)

```bash
git clone <this repo>
cd hiver-support-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (dataset is already generated and checked in at data/raw/; to regenerate:)
python data/generate_synthetic_dataset.py

# Quick demo — runs in "mock" LLM mode (deterministic, offline, no API key)
python scripts/run_demo.py

# Full evaluation harness — classification + decision + reply-quality metrics
python eval/run_eval.py

# Tests
pip install pytest
python -m pytest tests/ -q
```

Total wall time for all of the above: under a minute, since mock mode has
no network calls. This is intentional (see `DECISION_LOG.md #6`) — the
assignment's 15-minute reproduction budget shouldn't be spent waiting on
API rate limits.

### Running with a real LLM ("live" mode)

```bash
export LLM_MODE=live
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_demo.py
python eval/run_eval.py
```

Every module (`classifier.py`, `reply_generator.py`, `llm_judge.py`) calls
the same `src/llm_client.complete()` function, so switching modes doesn't
touch any other code. **The numbers printed in REPORT.md were produced in
mock mode** (since I have no API key provisioned in this environment) —
REPORT.md is explicit about which numbers would change under live mode and
why (short answer: intent classification and reply grounding would improve
substantially; the decision-engine's rule logic wouldn't change since it's
not itself an LLM call).

## What I chose *not* to build (with one more week, I would)

- A real embeddings-based retriever instead of TF-IDF (see `DECISION_LOG.md #5`).
- Multi-turn context beyond one customer message + one prior agent reply —
  real Twitter threads can run longer.
- A calibrated confidence score instead of self-reported LLM confidence
  (flagged as a known weak signal in `REPORT.md`).
- A second, blind human rater for the LLM-judge agreement check, rather
  than my own single-pass review (flagged inline in `eval/llm_judge.py`).
- Scaling the golden set from 60 to the requested 150–250 — reasonable for
  a demonstration harness on synthetic data, not for a real launch decision
  (see `DECISION_LOG.md #3`).

Full write-up, including the two required baselines, failure analysis, and
the mandatory "what's misleading about my headline number" section, is in
**REPORT.md**.
