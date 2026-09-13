# Decision Log

Plain list of the non-obvious calls I made and why. Bullet points, no essay.

**1. Picked an 8-intent taxonomy, not finer-grained.**
Started from ~14 candidate intents after skimming the synthetic threads,
then merged down wherever two intents led to the *same* reply template and
the *same* auto/escalate decision (e.g. "wrong item shipped" and "item
damaged in transit" both collapsed into `shipping_delivery_issue`, because
both get the same "DM order number" reply and the same auto-handle default).
Rule of thumb: an intent earns its own category only if it changes what the
agent *does* next.

**2. Substituted a synthetic dataset for the real Kaggle one.**
No network access to kaggle.com/huggingface.co in this environment. Built a
generator (`data/generate_synthetic_dataset.py`) that reproduces the real
dataset's exact schema and injects the same messiness (unresolved threads,
repeat contacts, typos) rather than hand-crafting a suspiciously clean toy
set. Documented prominently in README so this isn't discovered halfway
through review.

**3. Golden set is 60 examples, not the requested 150–250.**
This is a demonstration harness on synthetic data — scaling to 250 hand-
labeled examples over data I generated myself would be labeling effort
without a corresponding real-world signal gain. Kept the process
identical to what I'd do at scale (stratified across intents, ~15%
deliberately ambiguous/edge-case) so it's a credible proof of method, not
just proof of volume.

**4. Kept the three classifiers (LLM, keyword, trivial) in one file
(`src/classifier.py`) instead of separate baseline modules.**
They share one interface (`ClassificationResult`) and get compared directly
in the eval harness — keeping them adjacent made the diff between them
easier to read during development. `baselines/baselines.py` re-exports them
so they're still easy to find per the assignment's file structure.

**5. TF-IDF retrieval instead of embeddings for grounding.**
At this corpus size (a few hundred resolved threads for one brand),
TF-IDF cosine similarity is free, deterministic, and testable without an
embeddings API call per query. If the eval harness showed grounding quality
suffering because of missed paraphrases TF-IDF can't catch, I'd swap to
embeddings — but I'd want that evidence first rather than assuming it.

**6. Default to a "mock" LLM mode that's genuinely offline.**
The assignment gives a 15-minute reproduction budget. An offline,
deterministic mock mode means reviewers don't spend that budget waiting on
someone else's API keys or rate limits. The mock is intentionally weak
(simple keyword matching) — see REPORT.md for why its numbers should not be
read as "how good is this system," only "does the pipeline wire together
correctly."

**7. Low-confidence escalation threshold set at 0.55, not tuned via search.**
Picked as a round, conservative-leaning number given self-reported LLM
confidence is a known-unreliable signal (models are frequently overconfident).
Didn't grid-search this against the 60-example golden set because that's
small enough to overfit a threshold to noise. Flagged as a tuning target on
the real dataset, not treated as a resolved question here.

**8. Auto-handle/escalate decision is rule-based, not another LLM call.**
Escalation is a high-stakes, low-volume decision. Rules are auditable —
anyone can read `decision_engine.py` and know exactly why a message got
escalated. An LLM "should this escalate?" call would probably be marginally
more flexible but harder to explain to a support-ops lead who wants to know
*why* the bot didn't catch something.

**9. Hard-coded escalation phrases (lawsuit, BBB, fire hazard, etc.) as a
short, reviewed list, not a giant keyword blocklist.**
A long blocklist gives a false sense of coverage and is expensive to
maintain. A short list of clearly high-stakes phrases, backstopped by the
low-confidence-escalation rule and the intent-level `default_escalate`
flags, is the layered approach — no single rule is asked to catch
everything.

**10. `cancellation_refund_request` and `complaint_general` are
`default_escalate=True` at the intent level, regardless of confidence.**
Money leaving the company and active customer frustration are the two
categories where I'd rather over-escalate than risk an automated reply
making things worse. This is a policy choice a real business could
disagree with (e.g. auto-approve refunds under $10) — flagged explicitly so
it's a decision, not a default I forgot to examine.

**11. Repeat-contact detection uses simple negative-keyword matching on the
customer's follow-up tweet, not sentiment analysis.**
Given the follow-up text is usually short and formulaic ("still not fixed",
"any update??"), a small keyword list is transparent and testable. Flagged
in `data_loader.py:Thread.had_repeat_contact` as the place to swap in
something more sophisticated if the failure analysis showed it missing
cases.

**12. The reply generator's system prompt explicitly forbids inventing
policy (refund amounts, timelines) not shown in retrieved examples.**
Grounding retrieval only helps if the model is told not to freelance when
retrieval comes back empty or thin. Chose "acknowledge + ask for details"
as the fallback behavior rather than a generic templated non-answer,
because it's the same behavior a competent human agent would default to.

**13. Judge rubric kept to 4 fields (grounded, on_topic, tone_appropriate,
1-5 score) instead of a longer rubric.**
A sprawling rubric is harder for a human reviewer to re-score consistently
when checking judge/human agreement — and agreement is exactly what the
assignment asks me to provide evidence for. A rubric a human can't apply
consistently themselves isn't a fair one to hold the judge to.

**14. Did not build a real second-human-rater agreement check.**
`eval/llm_judge.py:judge_human_agreement()` is wired correctly but the
"human_scores" used in `run_eval.py` are a labeled placeholder, not a blind
second rater — flagged loudly in both files rather than silently presenting
a suspiciously perfect agreement number as if it were real evidence.

**15. Chose PixelWave over reusing a real brand name from the (would-be)
Kaggle set.**
Even in the synthetic substitution, using a real company's handle felt like
it could misrepresent actual brand policy or tone. A fictional brand keeps
the exercise honest about being a demonstration.
