"""
Scans the real Kaggle dataset (data/raw/twcs.csv) and ranks candidate
"brand" accounts by inbound-tweet volume, so you can pick one brand to
build the pipeline against — as the assignment requires ("Pick one brand
from the dataset").

A "brand" here means an author_id that only ever appears as the RECIPIENT
of inbound customer tweets and the SENDER of outbound (non-inbound) reply
tweets — i.e. a company support handle, not a customer.

Usage:
    python data/scan_brands.py
    python data/scan_brands.py --top 30
"""

import argparse
import csv
from collections import Counter

DEFAULT_PATH = "data/raw/twcs.csv"


def scan(path: str, top_n: int = 20):
    brand_reply_counts = Counter()
    total_rows = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            # Outbound (agent) tweets: inbound == "False"
            # These are the brand accounts, e.g. AmazonHelp, AppleSupport
            if row["inbound"] == "False":
                brand_reply_counts[row["author_id"]] += 1

            if total_rows % 500000 == 0:
                print(f"...scanned {total_rows:,} rows so far")

    print(f"\nFinished scanning {total_rows:,} total rows.\n")
    print(f"Top {top_n} brand accounts by number of outbound replies sent:\n")
    print(f"{'Brand handle':30s} {'Reply count':>12s}")
    print("-" * 44)
    for brand, count in brand_reply_counts.most_common(top_n):
        print(f"{brand:30s} {count:>12,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    scan(args.path, args.top)
