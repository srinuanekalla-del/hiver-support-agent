"""
Filters the real Kaggle dataset (data/raw/twcs.csv, ~3M rows, all brands)
down to just the threads involving ONE chosen brand, and writes it out in
the exact same schema the pipeline already reads (see src/data_loader.py):

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

A "thread involving this brand" = any row where the brand is either the
author (an outbound reply) or the row is an inbound customer tweet that is
part of a chain leading to/from a reply by this brand. To keep this simple
and correct, we take the more inclusive approach used by most public
solutions to this dataset: keep any row that mentions the brand's handle
in the tweet text (inbound customer tweets always @-mention the brand) OR
is authored by the brand itself (outbound replies).

Usage:
    python data/filter_brand.py --brand SpotifyCares
    python data/filter_brand.py --brand AmazonHelp --out data/raw/amazon_help_tweets.csv
"""

import argparse
import csv


def filter_brand(input_path: str, brand: str, output_path: str):
    mention_tag = f"@{brand}".lower()

    fieldnames = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    kept = 0
    total = 0

    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            is_brand_reply = row["author_id"] == brand
            is_mention = mention_tag in row["text"].lower()

            if is_brand_reply or is_mention:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                kept += 1

            if total % 500000 == 0:
                print(f"...scanned {total:,} rows, kept {kept:,} so far")

    print(f"\nDone. Scanned {total:,} rows, wrote {kept:,} rows for brand "
          f"'{brand}' to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/twcs.csv")
    parser.add_argument("--brand", required=True, help="e.g. SpotifyCares")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or f"data/raw/{args.brand.lower()}_support_tweets.csv"
    filter_brand(args.input, args.brand, out_path)
