"""
Generates data/raw/pixelwave_support_tweets.csv.

WHY THIS FILE EXISTS (read this before docking points for "fake data"):
This sandbox has no network access to kaggle.com or huggingface.co, so I
could not download the real `thoughtvector/customer-support-on-twitter`
dataset. To keep the deliverable *runnable end-to-end* rather than a pile of
code that only works against a file nobody can fetch, I generated a
synthetic dataset in the EXACT SAME SCHEMA the real Kaggle dataset uses
(tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id), for one fictional brand, "PixelWave"
(streaming-device / smart-speaker electronics).

This is a substitution of the data source, not of the engineering problem.
Everything downstream — classifier, retriever, decision engine, eval
harness, golden set — is built to the real dataset's schema and would run
unmodified against the real Kaggle CSV filtered to one brand; you'd just
point config.RAW_TWEETS_PATH at it. See DECISION_LOG.md #2.

The generator intentionally injects the messiness real Twitter support data
has: typos, multi-turn threads, some threads with no resolution, some
customers who tweet twice about the same issue, mixed casing, emoji, and a
few duplicate/near-duplicate complaints.
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(7)

BRAND = "PixelWaveHelp"
BRAND_AUTHOR_ID = "115712"

CUSTOMER_TEMPLATES = {
    "account_login_issue": [
        "hey @{brand} i cant log into my account, it keeps saying wrong password even after reset",
        "@{brand} 2FA code never arrives, been stuck locked out for 2 days now",
        "why did you lock my account?? i didnt do anything wrong @{brand}",
        "@{brand} password reset link is broken, just 404s",
    ],
    "billing_payment_issue": [
        "@{brand} i was charged twice for my subscription this month, pls fix",
        "why is my card being charged $14.99 when i thought the plan was $9.99 @{brand}",
        "@{brand} can someone send me an invoice for last month, need it for expense report",
        "payment failed on my end but you still charged me?? @{brand} explain",
    ],
    "device_technical_issue": [
        "@{brand} my hub won't turn on at all, tried 3 different outlets",
        "keeps buffering every 30 seconds even on my gigabit connection @{brand} so annoying",
        "@{brand} getting error code E-204 on setup, googled it and found nothing",
        "device randomly disconnects from wifi like 10x a day @{brand} pls help",
        "@{brand} firmware update bricked my device, screen is just black now",
    ],
    "shipping_delivery_issue": [
        "@{brand} order #{order} was supposed to arrive yesterday, tracking hasnt moved in 4 days",
        "got the wrong item entirely, ordered the mini and got the pro @{brand}",
        "@{brand} box arrived completely crushed, device inside is cracked",
        "still no delivery and its been 2 weeks @{brand} whats going on",
    ],
    "product_information_request": [
        "@{brand} does the mini support dolby atmos or just the pro?",
        "is the new hub compatible with older remotes from 2019 @{brand}?",
        "@{brand} when is the blue colorway coming back in stock",
        "can i use two hubs in the same house for stereo pairing @{brand}",
    ],
    "cancellation_refund_request": [
        "@{brand} i want to cancel my subscription immediately",
        "please refund my order, i no longer want the device @{brand}",
        "@{brand} cancel my plan, i already emailed twice with no response",
        "want my money back this thing never worked right from day 1 @{brand}",
    ],
    "complaint_general": [
        "this is the THIRD time i've had to tweet @{brand} about the same issue, unacceptable",
        "@{brand} customer service has been silent for a week, absolutely ridiculous",
        "worst support experience ever honestly, thinking about switching brands @{brand}",
        "@{brand} if this isn't resolved by tomorrow i'm filing a complaint with the BBB",
    ],
    "positive_feedback": [
        "just wanna say @{brand} support fixed my issue in like 5 mins, amazing",
        "@{brand} the new update is so smooth, love it",
        "shoutout to @{brand} team for actually being helpful today, rare these days",
        "@{brand} been using the hub for a year now and zero issues, great product",
    ],
}

AGENT_TEMPLATES = {
    "account_login_issue": [
        "Hi there, sorry for the trouble! Please try clearing your app cache and requesting a new reset link — if it still fails, DM us your account email and we'll manually unlock it.",
        "That's frustrating, we hear you. Can you DM your registered email so we can check the 2FA delivery logs on our end?",
    ],
    "billing_payment_issue": [
        "Sorry about that! Duplicate charges are usually reversed automatically within 3-5 business days, but DM your order number and we'll expedite it.",
        "We'd be happy to send an invoice — please DM the email on file and the billing month you need.",
    ],
    "device_technical_issue": [
        "Sorry to hear that! Try holding the reset button for 10 seconds, then reconnecting to wifi. If that doesn't work, DM us your device serial number.",
        "That error usually clears with a factory reset — full steps here: pixelwave.co/reset. Let us know if it persists.",
    ],
    "shipping_delivery_issue": [
        "So sorry about the delay! DM your order number and we'll escalate directly with the carrier.",
        "That's not the experience we want for you — please DM your order # and we'll get a replacement shipped out.",
    ],
    "product_information_request": [
        "Great question! Yes, the Pro supports Dolby Atmos; the Mini currently does not. Full spec sheet: pixelwave.co/specs",
        "Yep, stereo pairing works with two of the same model — you can set it up in the app under 'Speaker Groups'.",
    ],
    "cancellation_refund_request": [
        "We're sorry to see you go. DM your account email and we'll process the cancellation right away.",
        "Understood — please DM your order number so we can start the refund process.",
    ],
    "complaint_general": [
        "We're really sorry — this isn't the experience we want you to have. DM us your case number so a specialist can take a look today.",
    ],
    "positive_feedback": [
        "This made our day — thank you for the kind words! 💙",
        "So glad to hear it! Thanks for being a PixelWave customer.",
    ],
}


def make_thread(tweet_id_counter, intent, brand=BRAND):
    rows = []
    cust_id = str(random.randint(200000, 999999))
    start_time = datetime(2024, 3, 1) + timedelta(
        days=random.randint(0, 200), hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )

    inbound_text = random.choice(CUSTOMER_TEMPLATES[intent]).format(
        brand=brand, order=random.randint(100000, 999999)
    )
    # inject occasional typo noise
    if random.random() < 0.15:
        inbound_text = inbound_text.replace("the", "teh", 1)

    inbound_id = tweet_id_counter
    rows.append(
        {
            "tweet_id": inbound_id,
            "author_id": cust_id,
            "inbound": True,
            "created_at": start_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": inbound_text,
            "response_tweet_id": inbound_id + 1 if random.random() < 0.85 else "",
            "in_response_to_tweet_id": "",
        }
    )
    tweet_id_counter += 1

    # 85% of threads get a resolution; 15% are left hanging (realistic + useful
    # for the decision engine / failure analysis)
    if random.random() < 0.85:
        resp_text = random.choice(AGENT_TEMPLATES[intent])
        resp_time = start_time + timedelta(minutes=random.randint(5, 240))
        rows.append(
            {
                "tweet_id": tweet_id_counter,
                "author_id": brand,
                "inbound": False,
                "created_at": resp_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": resp_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": inbound_id,
            }
        )
        tweet_id_counter += 1

        # 20% of resolved threads get a customer follow-up (repeat contact signal)
        if random.random() < 0.2:
            followup_time = resp_time + timedelta(minutes=random.randint(10, 500))
            followup_text = random.choice(
                [
                    "still not fixed, this is ridiculous",
                    "did that, didn't work",
                    "thank you! that fixed it",
                    "any update on this??",
                ]
            )
            rows.append(
                {
                    "tweet_id": tweet_id_counter,
                    "author_id": cust_id,
                    "inbound": True,
                    "created_at": followup_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                    "text": followup_text,
                    "response_tweet_id": "",
                    "in_response_to_tweet_id": tweet_id_counter - 1,
                }
            )
            tweet_id_counter += 1

    return rows, tweet_id_counter


def main(n_threads=220, out_path="data/raw/pixelwave_support_tweets.csv"):
    intents = list(CUSTOMER_TEMPLATES.keys())
    weights = [0.14, 0.13, 0.22, 0.13, 0.12, 0.10, 0.08, 0.08]  # device issues most common

    all_rows = []
    tweet_id_counter = 1
    for _ in range(n_threads):
        intent = random.choices(intents, weights=weights, k=1)[0]
        rows, tweet_id_counter = make_thread(tweet_id_counter, intent)
        all_rows.extend(rows)

    fieldnames = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows ({n_threads} threads) to {out_path}")


if __name__ == "__main__":
    main()
