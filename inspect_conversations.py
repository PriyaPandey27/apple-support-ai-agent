import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("apple_conversations.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

print("Total conversations:", len(conversations))

for i, conversation in enumerate(conversations[:10], 1):

    print("\n" + "=" * 80)
    print(f"CONVERSATION {i}")
    print("=" * 80)

    for tweet in conversation:
        speaker = "CUSTOMER" if tweet["inbound"] else "APPLE"
        print(f"\n[{speaker}]")
        print(tweet["text"])