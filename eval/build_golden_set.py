"""
Pulls a random sample of REAL inbound customer messages from the filtered
brand CSV (e.g. data/raw/spotifycares_support_tweets.csv) and writes them
to a CSV with a suggested_intent column pre-filled by the keyword baseline,
plus empty gold_intent / gold_action / notes columns for YOU to fill in by
hand.

This does NOT auto-label the golden set — the suggested_intent column is a
head start to review and correct, not a substitute for your own judgment.
The assignment requires a golden set "you built yourself" with "a short
note on how you sampled and labelled them" — this script is the sampling
half; the labeling still has to be done by a human (you) reading each row.

Usage:
    python eval/build_golden_set.py --input data/raw/spotifycares_support_tweets.csv --n 150 --out eval/golden_set_TO_LABEL.csv
"""

import argparse
import csv
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import KeywordBaseline

random.seed(42)


def load_inbound_messages(path: str) -> list[str]:
    messages = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["inbound"] == "True" and row["text"].strip():
                messages.append(row["text"].strip())
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="e.g. data/raw/spotifycares_support_tweets.csv")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--out", default="eval/golden_set_TO_LABEL.csv")
    args = parser.parse_args()

    print(f"Loading inbound customer messages from {args.input} ...")
    messages = load_inbound_messages(args.input)
    print(f"Found {len(messages):,} inbound messages total.")

    # Dedupe near-identical messages (real Twitter data has a lot of exact
    # or near-exact repeats, e.g. copy-pasted complaints) so the sample
    # isn't dominated by duplicates.
    seen = set()
    unique_messages = []
    for m in messages:
        key = m.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_messages.append(m)
    print(f"{len(unique_messages):,} unique messages after de-duplication.")

    sample_size = min(args.n, len(unique_messages))
    sample = random.sample(unique_messages, sample_size)

    baseline = KeywordBaseline()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message", "suggested_intent", "gold_intent", "gold_action", "notes"])
        for msg in sample:
            suggestion = baseline.classify(msg).intent
            writer.writerow([msg, suggestion, "", "", ""])

    print(f"\nWrote {sample_size} unlabeled rows to {args.out}")
    print("Next: open this file (Excel/Google Sheets/Notepad) and, for each row:")
    print("  1. Read the message and correct 'suggested_intent' into 'gold_intent'")
    print("     (the suggestion is just a rough keyword-based guess -- trust your own reading)")
    print("  2. Decide 'gold_action': auto_handle or escalate")
    print("  3. Optionally jot a one-line reason in 'notes' (helps later failure analysis)")
    print("When done, rename/save it as eval/golden_set.csv to replace the old one.")


if __name__ == "__main__":
    main()
