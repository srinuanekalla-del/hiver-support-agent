# SpotifyCares AI Support Agent — Hiver Take-Home

An AI support agent for **SpotifyCares** (Spotify's real Twitter support
handle) that classifies incoming customer messages into intents, drafts a
reply grounded in how SpotifyCares has historically resolved similar
issues, and decides whether the message can be auto-handled or needs a
human — with a stated reason for that decision.

## About the dataset

This project uses the real **`thoughtvector/customer-support-on-twitter`**
Kaggle dataset (~2.8M tweets, dozens of brands), filtered down to just
**SpotifyCares** using `data/filter_brand.py` (74,618 rows).

The raw dataset files (`twcs.csv` and the filtered
`spotifycares_support_tweets.csv`) are **not checked into this repo** —
they're large (500MB+) and excluded via `.gitignore`. To reproduce:

```bash
# 1. Download the dataset from Kaggle (free account required):
#    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# 2. Extract it, place twcs.csv at: data/raw/twcs.csv
# 3. Filter it to SpotifyCares:
python data/filter_brand.py --brand SpotifyCares
```

This regenerates `data/raw/spotifycares_support_tweets.csv`, which the
pipeline reads by default (see `src/config.py:RAW_TWEETS_PATH`).

**Note on process:** development started against a synthetic dataset (see
`data/generate_synthetic_dataset.py`, still present, still runnable) built
in the same schema, because the original dev environment had no network
access to Kaggle. Once real network access was available, the project was
switched over to the real dataset above — the architecture didn't need to
change, only the data source (see `DECISION_LOG.md #2`).

**A real data-quality issue was found and fixed along the way:** some
`tweet_id` values in the real Kaggle file are comma-formatted (e.g.
`861,858`), which breaks naive `int()` parsing. `src/data_loader.py`
handles this defensively (`_safe_int`) rather than crashing or silently
dropping rows — a good example of the "messy real-world dataset" the
assignment brief calls out.

## What "good" means for this system (short version — full version in REPORT.md)

A reply is good if it's (1) grounded in what SpotifyCares actually told
past customers, not invented policy, and (2) the routing decision doesn't
auto-handle something that needed a human. I weight **false auto-handles**
(escalate-worthy message wrongly auto-handled) far more heavily than
over-escalation, because the failure mode of "bot promised something it
shouldn't have" or "bot ignored a furious repeat customer" is reputationally
expensive in a way that "human reviewed something routine" is not.

## Project layout

```
data/
  generate_synthetic_dataset.py   # original synthetic-data generator (see note above)
  scan_brands.py                   # ranks brands in the real dataset by volume
  filter_brand.py                  # filters the real twcs.csv down to one brand
  raw/                              # twcs.csv + filtered CSVs (gitignored, regenerate per above)
src/
  intents.py           # the 8-intent taxonomy + rationale
  data_loader.py         # reconstructs threads from the raw tweet CSV (handles messy IDs)
  classifier.py           # LLMClassifier + KeywordBaseline + TrivialBaseline
  retriever.py             # TF-IDF retrieval over resolved historical threads
  reply_generator.py       # drafts a reply grounded in retrieved history
  decision_engine.py       # auto-handle vs escalate, with a stated reason
  pipeline.py               # wires the above into one SupportAgentPipeline
  llm_client.py             # Anthropic API wrapper + offline mock mode
  config.py
eval/
  golden_set.csv            # hand-labeled examples (see note below on current status)
  build_golden_set.py        # samples real messages from the filtered CSV for hand-labeling
  metrics.py                  # classification + decision metrics
  llm_judge.py                 # LLM-as-judge for reply quality + human-agreement check
  run_eval.py                   # runs everything end-to-end, prints the report
baselines/
  baselines.py                   # re-exports the two required baselines
scripts/
  run_demo.py                     # quick illustrative run, no API key needed
tests/
  test_pipeline.py
REPORT.md
DECISION_LOG.md
```

## Setup & reproduction (under 15 minutes, no API key required)

**Windows (Command Prompt):**
```bat
git clone https://github.com/srinuanekalla-del/hiver-support-agent.git
cd hiver-support-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

:: Quick demo — runs in "mock" LLM mode (deterministic, offline, no API key)
python scripts\run_demo.py

:: Full evaluation harness — classification + decision + reply-quality metrics
python eval\run_eval.py

:: Tests
pip install pytest
python -m pytest tests -q
```

**Mac/Linux:**
```bash
git clone https://github.com/srinuanekalla-del/hiver-support-agent.git
cd hiver-support-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/run_demo.py
python eval/run_eval.py

pip install pytest
python -m pytest tests/ -q
```

Total wall time for all of the above: under a minute, since mock mode has
no network calls (see `DECISION_LOG.md #6`). To run against the real
SpotifyCares data, first follow the "About the dataset" steps above to
download and filter `twcs.csv` — the eval scripts will otherwise fall back
to whatever `src/config.py:RAW_TWEETS_PATH` points at.

### Running with a real LLM ("live" mode)

```bash
# Windows
set LLM_MODE=live
set ANTHROPIC_API_KEY=sk-ant-...
python scripts\run_demo.py

# Mac/Linux
export LLM_MODE=live
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_demo.py
```

Every module (`classifier.py`, `reply_generator.py`, `llm_judge.py`) calls
the same `src/llm_client.complete()` function, so switching modes doesn't
touch any other code. **The numbers currently in REPORT.md were produced in
mock mode** — REPORT.md explains exactly why the mock numbers should not be
read as real model-quality evidence, and what would change under live mode.

## Current status / known gaps (read before reviewing results)

- **Golden set (`eval/golden_set.csv`) needs to be rebuilt from real
  SpotifyCares messages.** It currently still reflects the earlier
  synthetic-brand phase of development. `eval/build_golden_set.py` samples
  real, de-duplicated messages straight from
  `data/raw/spotifycares_support_tweets.csv` with a rough keyword-based
  suggested label for a human to review and correct — this is the
  in-progress step. Numbers in REPORT.md should be read as "the harness
  works correctly" evidence, not final accuracy against real SpotifyCares
  data, until this is complete.
- **Live-mode (real Claude API) results have not yet been captured** —
  see REPORT.md for what's expected to change once they are.

## What I chose *not* to build (with one more week, I would)

- A real embeddings-based retriever instead of TF-IDF (see `DECISION_LOG.md #5`).
- Multi-turn context beyond one customer message + one prior agent reply —
  real Twitter threads can run longer.
- A calibrated confidence score instead of self-reported LLM confidence
  (flagged as a known weak signal in `REPORT.md`).
- A second, blind human rater for the LLM-judge agreement check, rather
  than my own single-pass review (flagged inline in `eval/llm_judge.py`).
- Finish scaling the golden set to the requested 150–250 real, hand-labeled
  SpotifyCares examples (see "Current status" above and `DECISION_LOG.md #3`).

Full write-up, including the two required baselines, failure analysis, and
the mandatory "what's misleading about my headline number" section, is in
**REPORT.md**.
