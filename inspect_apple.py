import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("apple_conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type:", type(data))
print("Conversations:", len(data))

lengths = [len(c) for c in data]

print("\nConversation lengths:")
print("Average:", sum(lengths) / len(lengths))
print("Shortest:", min(lengths))
print("Longest:", max(lengths))

print("\nSample conversations:\n")

samples = random.sample(data, min(10, len(data)))

for i, conv in enumerate(samples, 1):
    print("=" * 80)
    print(f"CONVERSATION {i}")

    for tweet in conv:
        print(
            f"[{tweet.get('created_at')}] "
            f"inbound={tweet.get('inbound')} | "
            f"id={tweet.get('tweet_id')}"
        )
        print(tweet.get("text", ""))
        print()
