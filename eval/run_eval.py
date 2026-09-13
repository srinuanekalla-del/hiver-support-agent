"""
Runs the full evaluation:
1. Intent classification: LLMClassifier vs KeywordBaseline vs TrivialBaseline
   on the golden set.
2. Decision (auto-handle/escalate): pipeline's DecisionEngine vs golden
   labels, reporting false-auto-handle rate as the headline safety metric.
3. Reply quality: LLM-as-judge scores on a sample, with a human-agreement
   spot-check.

Usage:
    python eval/run_eval.py
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.classifier import LLMClassifier, KeywordBaseline, TrivialBaseline
from src.decision_engine import DecisionEngine
from src.retriever import ResolutionRetriever
from src.reply_generator import ReplyGenerator
from src.data_loader import load_threads
from eval.metrics import classification_report, decision_report, print_report
from eval.llm_judge import judge_reply, judge_human_agreement


def load_golden_set(path=None):
    path = path or config.GOLDEN_SET_PATH
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_classification_eval(golden):
    messages = [row["message"] for row in golden]
    gold_intents = [row["gold_intent"] for row in golden]

    llm_clf = LLMClassifier()
    kw_clf = KeywordBaseline()
    trivial_clf = TrivialBaseline(most_common_intent="device_technical_issue")

    llm_preds = [llm_clf.classify(m).intent for m in messages]
    kw_preds = [kw_clf.classify(m).intent for m in messages]
    trivial_preds = [trivial_clf.classify(m).intent for m in messages]

    print_report("Trivial baseline (always predict most common class)",
                  classification_report(gold_intents, trivial_preds))
    print_report("Keyword baseline", classification_report(gold_intents, kw_preds))
    print_report("LLM classifier (this system)", classification_report(gold_intents, llm_preds))

    return llm_preds  # used downstream for the decision eval


def run_decision_eval(golden, llm_intent_preds):
    messages = [row["message"] for row in golden]
    gold_actions = [row["gold_action"] for row in golden]

    engine = DecisionEngine()
    clf = LLMClassifier()

    pred_actions = []
    for msg, intent in zip(messages, llm_intent_preds):
        conf = clf.classify(msg).confidence  # re-fetch confidence (cheap in mock mode)
        decision = engine.decide(msg, intent, conf, repeat_contact=False)
        pred_actions.append(decision.action)

    report = decision_report(gold_actions, pred_actions)
    print_report("Decision engine (auto-handle vs escalate)", report)

    if report["false_auto_handle_count"] > 0:
        print(f"\n⚠ {report['false_auto_handle_count']} escalate-worthy message(s) were "
              f"auto-handled — see these in REPORT.md failure analysis:")
        for msg, gold, pred in zip(messages, gold_actions, pred_actions):
            if gold == "escalate" and pred == "auto_handle":
                print(f"    - \"{msg}\"")


def run_reply_quality_eval(golden, sample_size=10):
    threads = load_threads()
    retriever = ResolutionRetriever(threads)
    generator = ReplyGenerator()

    sample = golden[:sample_size]
    judge_scores = []
    for row in sample:
        msg = row["message"]
        retrieved = retriever.retrieve(msg)
        draft = generator.generate(msg, retrieved)
        result = judge_reply(msg, draft.text)
        judge_scores.append(result.score)

    avg_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    print(f"\n--- Reply quality (LLM-as-judge, n={len(sample)}) ---")
    print(f"average score (1-5): {avg_score:.2f}")

    # Human-agreement spot check. In mock mode these are illustrative
    # placeholder human scores — see docstring in llm_judge.py. In live mode,
    # replace `human_scores` with actual second-rater scores on the same
    # sample before citing this number in a report.
    human_scores_placeholder = [3] * len(judge_scores)
    agreement = judge_human_agreement(judge_scores, human_scores_placeholder)
    print(f"judge/human agreement (PLACEHOLDER human scores, replace before citing): {agreement}")


def main():
    print(f"LLM_MODE={config.LLM_MODE}")
    golden = load_golden_set()
    print(f"Loaded golden set: {len(golden)} examples")

    llm_intent_preds = run_classification_eval(golden)
    run_decision_eval(golden, llm_intent_preds)
    run_reply_quality_eval(golden)


if __name__ == "__main__":
    main()
