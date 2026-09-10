import pandas as pd
import json

df = pd.read_csv("apple_support.csv")

df["tweet_id"] = df["tweet_id"].astype(str)

tweet_map = df.set_index("tweet_id").to_dict("index")

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

        next_id = tweet.get("response_tweet_id")

        if pd.isna(next_id):
            break

        # response_tweet_id can contain multiple IDs
        next_id = str(next_id).split(",")[0].strip()

        if next_id not in tweet_map:
            break

        current_id = next_id

    if len(conversation) >= 2:
        conversations.append(conversation)

print("Conversations found:", len(conversations))

with open("apple_conversations.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print("Saved to apple_conversations.json")

lengths = [len(c) for c in conversations]

print("\nConversation statistics:")
print("Average tweets:", round(sum(lengths) / len(lengths), 2))
print("Shortest:", min(lengths))
print("Longest:", max(lengths))