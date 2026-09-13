# Report

All numbers below are from `python eval/run_eval.py` in **mock LLM mode**
(no API key available in this environment — see README). Mock mode uses a
deliberately simple keyword-based stand-in for the LLM calls, not a real
model. I've kept the numbers exactly as they printed, including the
unflattering ones, because they're honest about what mock mode is (a
plumbing check) versus what it isn't (a model-quality signal). Numbers that
would materially change under a real model are called out inline.

## 1. Problem framing: what "good" means for PixelWave, and what I chose not to build

**Good, for this system, means three things in priority order:**

1. **Never auto-handle something that needed a human.** A wrong intent
   label is annoying; an auto-sent reply that ignores a furious repeat
   customer, promises a refund the company wouldn't actually give, or
   glosses over a safety complaint (e.g. "device started smoking") is a
   real cost — reputational and sometimes legal. This is why the decision
   engine treats `false_auto_handle_rate` as the metric to optimize first,
   ahead of raw intent accuracy.
2. **Ground replies in what PixelWave actually does, not invented policy.**
   A reply that sounds confident but invents a refund timeline is worse
   than a reply that honestly says "let us look into this" — the first
   creates a promise the company has to either honor or walk back.
3. **Intent accuracy, as a means to the above two, not an end in itself.**
   Getting `device_technical_issue` vs `account_login_issue` wrong matters
   mainly because it changes the reply template and (for a few intents)
   the escalation default — not because intent labels are inherently
   valuable.

**What I chose not to build** (would with one more week — see REPORT §5):
a real embeddings retriever, multi-turn context beyond one prior exchange,
a calibrated (rather than self-reported) confidence score, and a blind
second human rater for judge agreement. Full list with reasoning in
`DECISION_LOG.md`.

## 2. Results vs. baselines

Three classifiers evaluated on the 60-example golden set:

| Classifier | Accuracy | Notes |
|---|---|---|
| **Trivial** (always predict most common class) | 18.3% | Required trivial baseline. |
| **Keyword** (simple keyword-match rules) | 63.3% | Required simple baseline. |
| **LLM classifier (mock mode)** | 55.0% | See caveat below — **this number is not a real model-quality signal.** |

**Why the "LLM classifier" scored *below* the keyword baseline here, and
why that's expected, not alarming:** in mock mode, `LLMClassifier` is
backed by the same style of keyword heuristic as the baseline (see
`src/llm_client.py:_mock_complete`), but with less coverage, because it's
standing in for a real model's judgment on ambiguous cases rather than
trying to out-heuristic the dedicated keyword baseline. Both mock outputs
are, definitionally, keyword matching wearing different hats — comparing
them tells you the harness works, not which approach is better. **The real
comparison this report can't yet make** is mock-heuristic vs. actual
Claude output, which is why `README.md` gives exact steps to re-run this
in `LLM_MODE=live` mode.

Decision engine (auto-handle vs. escalate) vs. gold labels:

| Metric | Value |
|---|---|
| Overall accuracy | 78.3% |
| False auto-handle rate (escalate-worthy, wrongly auto-handled) | **65.0%** (13/20) |
| Over-escalation rate (auto-worthy, wrongly escalated) | 0.0% (0/40) |

Reply quality (LLM-as-judge, n=10 sample): average score **3.00 / 5**.

## 3. "What is misleading about my headline number?" (mandatory section)

**The headline number to distrust here is "78.3% decision accuracy."** On
its own it sounds like a B+. It is not.

That 78.3% is almost entirely propped up by the model being *correctly
conservative on the easy 40 auto-handle cases* (0% over-escalation) while
being *badly wrong on the 20 cases that actually mattered* (65% false
auto-handle rate). A support-ops lead who reads "78% accuracy" and ships
this would be shipping a system that mishandles two out of every three
messages that should have gone to a human — including, in this run, a
message about a device that "started smoking."

Two specific mechanisms are driving that false-auto-handle rate, and
neither is really about model quality:

1. **Confidence is hardcoded in mock mode.** `LLMClassifier` in mock mode
   always returns confidence 0.60, regardless of how genuinely ambiguous
   the message is (see `src/llm_client.py`). That means the low-confidence
   escalation trigger (`< 0.55`) never fires in this run — it's structurally
   disabled by the mock, not proven ineffective. In live mode, a real
   model's self-reported confidence would vary and this trigger would
   actually do work (though see caveat below — self-reported LLM confidence
   is itself a known-weak calibration signal, so this wouldn't fully close
   the gap either).
2. **The eval harness doesn't wire `repeat_contact` through.** Several of
   the false auto-handles in this run (e.g. "still not fixed, this is
   ridiculous, did that already") are exactly the case the
   `repeat_contact` escalation rule exists for — but `run_eval.py`
   currently calls `decide(..., repeat_contact=False)` for every golden
   example rather than detecting it from thread history. That's a harness
   gap, not a decision-engine design gap; fixing it (thread the golden
   set's own prior-turn context through `Thread.had_repeat_contact`) would
   likely resolve several of these on its own.

So: **78.3% is real, but it is measuring "the system is appropriately
cautious in the majority-easy case," not "the system is safe to deploy."**
The number to actually watch, report to a stakeholder, and gate a launch
decision on is the 65% false-auto-handle rate — and even that number is
inflated by two fixable harness issues above, not a verdict on the
underlying rule design.

## 4. Failure analysis: top 5 modes, with real examples

**1. Confidence never varies (mock-mode artifact, see §3).**
Every classification returns 0.60 confidence in mock mode, so the
low-confidence-escalation safety net never engages.
*Hypothesis:* fixed once real model calls are used (`LLM_MODE=live`), since
a real model's confidence output — imperfect as it is — will actually vary
by message ambiguity.

**2. Repeat-contact signal not wired into the eval harness.**
Example: `"@PixelWaveHelp still not fixed, this is ridiculous, did that
already"` — this is a textbook repeat-contact case, but `run_eval.py`
never passes the prior-turn context needed for `DecisionEngine` to see it.
*Hypothesis:* fixed by having `run_eval.py` reconstruct
`Thread.had_repeat_contact` from the golden set's source thread, not by
changing the decision engine itself.

**3. Ambiguous-safety and ambiguous-scope messages default to auto-handle.**
Example: `"@PixelWaveHelp device got so hot it started smoking, kinda
scared to touch it"` — no hard-escalation keyword ("fire hazard", "unsafe")
was hit, and the message classified as an ordinary `device_technical_issue`
with the (mock's flat) 0.60 confidence, so it auto-handled.
*Hypothesis:* the hard-escalation phrase list is too literal — it needs
semantic coverage of "safety-adjacent" language (heat, smoke, sparks,
shock), not just legal-threat phrases. This is a rule-design gap, not a
mock-mode artifact, and the one failure mode in this list I'd fix even
without a real LLM.

**4. Product-information intent is systematically under-predicted.**
Per-class recall for `product_information_request` was 0.0 in this run —
every example in that class got misclassified as something else.
*Hypothesis:* the mock's keyword list for this intent
(`"does it support"`, `"is it compatible"`) is too narrowly phrased and
misses natural variants like `"does the mini support..."`. This is squarely
a mock-mode limitation; a real model doesn't need exact-phrase matches.

**5. Multi-intent messages get force-fit into one label.**
Example: `"@PixelWaveHelp my kid ordered the pro without asking, can we
cancel and also is it kid-safe re small parts"` combines a cancellation
request and a safety question. The pipeline (correctly, by luck of the
cancel keyword) escalated it, but the reply-drafting step only ever sees
one intent label and would draft around whichever one won classification —
potentially ignoring the safety question entirely.
*Hypothesis:* needs either multi-label classification or an explicit
"does this message contain more than one distinct request?" pre-check
before drafting a reply, so nothing gets silently dropped.

## 5. What I'd do with one more week

- Wire `repeat_contact` through the eval harness properly (failure mode #2
  — this alone should meaningfully move the false-auto-handle number).
- Expand the hard-escalation phrase list to cover safety-adjacent language
  semantically, not just literal legal-threat phrases (failure mode #3).
- Re-run the full eval in `LLM_MODE=live` against real Claude output and
  replace every mock-mode number in this report with the real one —
  everything in §2 and §3 is scaffolding for that comparison, not a
  substitute for it.
- Add multi-label intent detection for messages that clearly contain more
  than one request (failure mode #5).
- Replace the placeholder "human scores" in `judge_human_agreement()` with
  a real second, blind rater's scores on a larger sample (n=30+).
- Scale the golden set from 60 to the requested 150–250 once running
  against the real Kaggle dataset (see `DECISION_LOG.md #3`), stratified
  the same way.
- Swap TF-IDF retrieval for embeddings and re-check whether grounding
  quality (per the judge's `grounded` field) actually improves enough to
  justify the added complexity and cost.
