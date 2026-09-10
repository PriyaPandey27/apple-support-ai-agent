import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

INPUT_FILE = "twcs.csv"
SEED_FILE = "apple_support.csv"
OUTPUT_FILE = "apple_conversations.json"

# --------------------------------------------------
# 1. Get Apple-related tweet IDs
# --------------------------------------------------

seeds = pd.read_csv(
    SEED_FILE,
    usecols=["tweet_id"]
)

relevant_ids = set(seeds["tweet_id"].astype(str))

print("Initial Apple-related tweets:", len(relevant_ids))


# --------------------------------------------------
# 2. Expand through conversation relationships
# --------------------------------------------------

for iteration in range(5):

    old_count = len(relevant_ids)

    print(f"\nRelationship pass {iteration + 1}")
    print("IDs before:", old_count)

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=100000,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id"
        ]
    ):

        chunk["tweet_id"] = chunk["tweet_id"].astype(str)

        # Parent relationship
        parent_ids = chunk["in_response_to_tweet_id"].astype("string")

        mask_parent = parent_ids.isin(relevant_ids)

        for tweet_id in chunk.loc[mask_parent, "tweet_id"]:
            relevant_ids.add(tweet_id)

        # Child relationship
        for _, row in chunk.iterrows():

            response_ids = str(row["response_tweet_id"])

            if response_ids == "nan":
                continue

            ids = response_ids.split(",")

            if any(tweet_id.strip() in relevant_ids for tweet_id in ids):
                relevant_ids.add(str(row["tweet_id"]))

    new_count = len(relevant_ids)

    print("IDs after:", new_count)
    print("New IDs:", new_count - old_count)

    if new_count == old_count:
        break


# --------------------------------------------------
# 3. Extract all relevant tweets from original data
# --------------------------------------------------

print("\nExtracting relevant tweets...")

tweets = []

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=100000,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
):

    chunk["tweet_id"] = chunk["tweet_id"].astype(str)

    selected = chunk[chunk["tweet_id"].isin(relevant_ids)]

    if len(selected) > 0:
        tweets.append(selected)

df = pd.concat(tweets, ignore_index=True)

print("Relevant tweets recovered:", len(df))


# --------------------------------------------------
# 4. Build lookup
# --------------------------------------------------

tweet_map = df.set_index("tweet_id").to_dict("index")


# --------------------------------------------------
# 5. Reconstruct conversations
# --------------------------------------------------

conversations = []

for _, row in df.iterrows():

    if not row["inbound"]:
        continue

    conversation = []

    current_id = str(row["tweet_id"])
    visited = set()

    while current_id in tweet_map and current_id not in visited:

        visited.add(current_id)

        tweet = tweet_map[current_id]

        conversation.append({
            "tweet_id": current_id,
            "author_id": str(tweet["author_id"]),
            "inbound": bool(tweet["inbound"]),
            "created_at": tweet["created_at"],
            "text": str(tweet["text"])
        })

        next_id = tweet["response_tweet_id"]

        if pd.isna(next_id):
            break

        next_id = str(next_id).split(",")[0].strip()

        if next_id not in tweet_map:
            break

        current_id = next_id

    # Keep conversations containing at least
    # one customer message and one Apple response
    if len(conversation) >= 2:

        has_customer = any(
            t["inbound"] for t in conversation
        )

        has_support = any(
            not t["inbound"] for t in conversation
        )

        if has_customer and has_support:
            conversations.append(conversation)


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        conversations,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n================================")
print("FINAL RESULTS")
print("================================")

print("Conversations:", len(conversations))
print("Saved to:", OUTPUT_FILE)

if conversations:

    lengths = [len(c) for c in conversations]

    print("Average tweets:",
          round(sum(lengths) / len(lengths), 2))

    print("Shortest:", min(lengths))
    print("Longest:", max(lengths))