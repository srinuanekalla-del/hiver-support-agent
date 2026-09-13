"""
Loads the raw tweet CSV and reconstructs threads (a customer inbound tweet,
optionally followed by a brand reply, optionally followed by a customer
follow-up). This mirrors how you'd process the real Kaggle dataset filtered
to one brand — reconstructing conversations from the
response_tweet_id / in_response_to_tweet_id linkage.
"""

import csv
from dataclasses import dataclass, field

from src import config


def _safe_int(value: str) -> int | None:
    """
    Parses a tweet-id-like field into an int, tolerating messiness found in
    the real Kaggle dataset (e.g. comma-formatted numbers like '861,858',
    stray whitespace, or empty strings). Returns None if it can't be
    parsed, rather than raising — a malformed ID should be treated as
    "no linked tweet", not a crash. This is exactly the kind of real-world
    messiness the assignment brief calls out.
    """
    if not value:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned or not cleaned.lstrip("-").isdigit():
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


@dataclass
class Thread:
    thread_id: int  # inbound tweet_id, used as the thread key
    customer_text: str
    customer_id: str
    agent_reply: str | None
    followup_text: str | None  # customer's reply to the agent, if any
    created_at: str

    @property
    def was_resolved(self) -> bool:
        return self.agent_reply is not None

    @property
    def had_repeat_contact(self) -> bool:
        """Customer tweeted again after the agent reply — a signal the first
        reply didn't land. Used both in decision_engine and in failure
        analysis (see REPORT.md)."""
        if not self.followup_text:
            return False
        negative_markers = ["still", "not fixed", "didn't work", "didnt work", "again", "ridiculous", "any update"]
        return any(m in self.followup_text.lower() for m in negative_markers)


def load_threads(path: str = None) -> list[Thread]:
    path = path or config.RAW_TWEETS_PATH
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_id = {}
    skipped_malformed_ids = 0
    for r in rows:
        tid = _safe_int(r["tweet_id"])
        if tid is None:
            skipped_malformed_ids += 1
            continue
        by_id[tid] = r

    if skipped_malformed_ids:
        print(f"[data_loader] skipped {skipped_malformed_ids} row(s) with "
              f"malformed tweet_id (real-world data messiness)")

    threads = []
    for r in rows:
        if r["inbound"] != "True":
            continue
        if r["in_response_to_tweet_id"]:
            continue  # this is a follow-up, not the thread starter

        inbound_id = _safe_int(r["tweet_id"])
        if inbound_id is None:
            continue
        agent_reply = None
        followup_text = None

        resp_id = _safe_int(r["response_tweet_id"])
        if resp_id is not None:
            resp_row = by_id.get(resp_id)
            if resp_row and resp_row["inbound"] == "False":
                agent_reply = resp_row["text"]
                # look for a customer follow-up to the agent reply
                followup_resp_id = _safe_int(resp_row["response_tweet_id"])
                if followup_resp_id is not None:
                    fu_row = by_id.get(followup_resp_id)
                    if fu_row and fu_row["inbound"] == "True":
                        followup_text = fu_row["text"]

        threads.append(
            Thread(
                thread_id=inbound_id,
                customer_text=r["text"],
                customer_id=r["author_id"],
                agent_reply=agent_reply,
                followup_text=followup_text,
                created_at=r["created_at"],
            )
        )

    return threads
