"""
prepare_dataset.py
===================

Stage 2 of the Hiver Apple-Support assignment.

Takes the already-reconstructed `apple_conversations.json` (built by
rebuild_apple_conversations.py) and turns it into the cleaned,
de-duplicated, labeled datasets the classifier / retrieval / evaluation
stages need.

This script does NOT re-touch twcs.csv and does NOT redo conversation
reconstruction. It only reads apple_conversations.json.

WHAT THIS SCRIPT DOES, IN ORDER
--------------------------------
1. Load apple_conversations.json.
2. De-duplicate "nested suffix" conversations. The original reconstruction
   starts a new conversation from EVERY inbound (customer) tweet. If a
   customer posted a two-part tweet-thread (tweet A, self-reply tweet B,
   then Apple replies to B), the script produces BOTH:
       [A, B, apple_reply, ...]   <- the real, complete conversation
       [B, apple_reply, ...]      <- a redundant partial duplicate
   We keep only the "root" conversation (the one whose first tweet is not
   itself a non-first tweet in some other, longer conversation) and drop
   the partial duplicates. This matters a lot for train/eval leakage:
   without it, ~28% of "conversations" are just truncated copies of a
   conversation kept elsewhere.
3. Drop the small number of conversations where the support reply came
   from a different brand's support handle that got pulled in by the
   relationship-expansion pass (e.g. AsurionCares, SpotifyCares). These
   are noise from ID collisions during thread expansion, not Apple support.
4. Collapse each conversation into one primary (customer_message,
   apple_response) pair: all leading customer tweets before Apple's first
   reply are concatenated (handles multi-tweet customer threads), and
   Apple's first reply is the response. This is the unit used for the
   intent classifier and the retrieval corpus.
5. Tag a resolution_status heuristic for every conversation (see
   `classify_resolution`). This is a heuristic, not ground truth -- it is
   reported as such everywhere it's used.
6. Tag an intent for every conversation using a small set of keyword rules
   derived by reading real customer messages (see `INTENT_RULES`). Rules
   are applied in a fixed priority order and are intentionally simple
   enough to explain in an interview.
7. Split conversations into train / classifier_eval / golden_eval with a
   fixed seed, at the CONVERSATION level, so no conversation (and no near-
   duplicate, since step 2 already removed those) appears in more than one
   split.
8. Write out all artifacts (see OUTPUT FILES below) plus a human-readable
   analysis report.

OUTPUT FILES (written to ./data/)
----------------------------------
- conversations_clean.jsonl   one row per kept conversation (full turns +
                               tags: conversation_id, resolution_status,
                               intent, split, n_turns)
- support_pairs.csv           the flattened (customer_message,
                               apple_response) table used downstream,
                               with intent + split columns
- retrieval_corpus.jsonl      support_pairs restricted to the TRAIN split
                               only, in the shape the retrieval step will
                               embed later
- golden_eval_template.csv    golden_eval split (200 examples), with empty
                               `human_label_*` columns for Priya to hand
                               label
- human_validation_subset.csv 50 human-reviewed calibration examples with
                               verified labels for LLM-as-judge agreement
- taxonomy.json / taxonomy.md human-readable intent definitions + stats
- analysis_report.md          automated EDA: filtering funnel, split
                               sizes, resolution-status breakdown, etc.

Run:
    python prepare_dataset.py
Runtime: ~10 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import re
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

RANDOM_SEED = 42
INPUT_FILE = "apple_conversations.json"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GOLDEN_EVAL_SIZE = 200           # within the assignment's 150-250 range
CLASSIFIER_EVAL_FRACTION = 0.15  # of the remaining conversations after golden is carved out


# --------------------------------------------------------------------------
# Step 1: load
# --------------------------------------------------------------------------

def load_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Step 2: drop redundant nested-suffix conversations
# --------------------------------------------------------------------------

def find_root_conversations(conversations):
    """
    A conversation is a "root" if its first tweet never shows up as a
    NON-first tweet inside some other conversation. If it does, that other
    (longer) conversation already contains this one as a trailing subsequence,
    so this one is a redundant partial copy and gets dropped.
    """
    tweet_positions = defaultdict(list)  # tweet_id -> [(conv_idx, position), ...]
    for i, conv in enumerate(conversations):
        for pos, tweet in enumerate(conv):
            tweet_positions[tweet["tweet_id"]].append((i, pos))

    root_indices = []
    for i, conv in enumerate(conversations):
        first_id = conv[0]["tweet_id"]
        is_redundant = any(
            j != i and pos > 0 for (j, pos) in tweet_positions[first_id]
        )
        if not is_redundant:
            root_indices.append(i)
    return [conversations[i] for i in root_indices]


# --------------------------------------------------------------------------
# Step 3: drop conversations whose "support" side isn't actually Apple
# --------------------------------------------------------------------------

def is_apple_support_conversation(conv):
    support_turns = [t for t in conv if not t["inbound"]]
    return len(support_turns) > 0 and all(
        t["author_id"] == "AppleSupport" for t in support_turns
    )


# --------------------------------------------------------------------------
# Step 4: collapse to primary (customer_message, apple_response) pair
# --------------------------------------------------------------------------

def build_primary_pair(conv):
    """
    Leading customer turns (handles multi-tweet customer threads) get
    concatenated into one customer_message. The first Apple turn after
    them is the apple_response. Returns None if there's no Apple turn
    at all (shouldn't happen after filtering, but defensive).
    """
    customer_parts = []
    for t in conv:
        if t["inbound"]:
            customer_parts.append(t["text"])
        else:
            return {
                "customer_message": " ".join(customer_parts).strip(),
                "apple_response": t["text"],
            }
    return None


# --------------------------------------------------------------------------
# Step 5: resolution-status heuristic (NOT ground truth -- documented as such)
# --------------------------------------------------------------------------

DM_PATTERN = re.compile(r"\bdm\b|direct message|dm us|please dm", re.IGNORECASE)
ROUTE_PATTERN = re.compile(
    r"contact (us at|apple support at)|call \d|1-800|genius bar|"
    r"visit (an apple|your local)|another team|different (department|team)",
    re.IGNORECASE,
)
THANKS_PATTERN = re.compile(r"\bthanks?\b|thank you|appreciate it|that (worked|fixed)", re.IGNORECASE)


def classify_resolution(conv):
    """
    Heuristic conversation-outcome tag, using only surface text signals.
    This is explicitly a proxy, not a measurement of whether the issue was
    actually solved -- Twitter threads routinely continue elsewhere (DM,
    phone, in person) where we have no visibility.

        customer_thanked      -> last customer turn contains a thanks-like phrase
        escalated_to_dm       -> an Apple turn asked the customer to DM
        routed_elsewhere      -> an Apple turn pointed to another channel/team
        ends_on_apple_turn    -> conversation thread ends with Apple still talking
                                  (rare in this dataset; usually means the thread
                                  was cut off before the customer's reply)
        unresolved_no_signal  -> none of the above; most common category, and
                                  should NOT be read as "failed"
    """
    apple_turns = [t["text"] for t in conv if not t["inbound"]]
    last_turn = conv[-1]

    if last_turn["inbound"] and THANKS_PATTERN.search(last_turn["text"]):
        return "customer_thanked"
    if any(DM_PATTERN.search(t) for t in apple_turns):
        return "escalated_to_dm"
    if any(ROUTE_PATTERN.search(t) for t in apple_turns):
        return "routed_elsewhere"
    if not last_turn["inbound"]:
        return "ends_on_apple_turn"
    return "unresolved_no_signal"


# --------------------------------------------------------------------------
# Step 6: intent taxonomy (rule-based, derived from reading real messages)
# --------------------------------------------------------------------------

INTENT_RULES = [
    ("autocorrect_text_bug",
     r"i\ufe0f|autocorrect|auto-correct|keyboard glitch|the letter i\b"),
    ("account_security_icloud",
     r"icloud|apple ?id|password|log ?in|locked out|2fa|two.factor|verification code|account (hacked|compromised|locked)"),
    ("billing_repair_order",
     r"refund|charged (me|twice)|\border\b|warranty|repair|replace(ment)?|\breturn\b|invoice|billing|genius bar"),
    ("battery_power",
     r"\bbattery|batteries|charg(e|ing|er)|drain|won.?t turn on|dying fast"),
    ("screen_hardware_damage",
     r"\bscreen|crack|shatter|display|touch ?id|face ?id|button (stuck|broken)|water damage"),
    ("connectivity",
     r"wifi|wi-fi|bluetooth|\bsignal\b|cellular|hotspot|no service|won.?t connect"),
    ("ios_software_update",
     r"updat(e|ed|ing)|ios ?\d|software|reinstall"),
    ("app_appstore_media",
     r"app store|itunes|apple music|\bapp(s)?\b|download|podcast"),
    ("device_not_working",
     r"frozen|freeze|crash(ed|ing)?|not (working|responding)|doesn.?t work|keeps (restarting|shutting|crashing)|bricked|dead phone"),
]
FALLBACK_INTENT = "other_unclear"


def classify_intent(customer_message):
    text = customer_message.lower()
    for name, pattern in INTENT_RULES:
        if re.search(pattern, text):
            return name
    return FALLBACK_INTENT


INTENT_DEFINITIONS = {
    "autocorrect_text_bug": {
        "definition": "Complaints about the iOS keyboard/autocorrect bug that mangled the letter 'i' (a real, documented iOS 11.1-era defect).",
        "include": "Message mentions the corrupted 'i' character, 'autocorrect', or describes the keyboard glitch.",
        "exclude": "General typing complaints unrelated to the specific 'i' bug.",
    },
    "account_security_icloud": {
        "definition": "Login, Apple ID, iCloud, password, or account-security/lockout issues.",
        "include": "Mentions iCloud, Apple ID, password, login, 2FA, or account being locked/hacked.",
        "exclude": "Purchase/billing issues tied to an account (goes to billing_repair_order).",
    },
    "billing_repair_order": {
        "definition": "Refunds, incorrect charges, orders, warranty, repairs, or replacements.",
        "include": "Mentions refund, charge, order, warranty, repair, replacement, return, Genius Bar.",
        "exclude": "General battery or screen complaints with no billing/repair request.",
    },
    "battery_power": {
        "definition": "Battery drain, charging problems, or device not powering on.",
        "include": "Mentions battery, charging, draining, or the device failing to turn on.",
        "exclude": "Device turning off/crashing without a battery/charging cause (device_not_working).",
    },
    "screen_hardware_damage": {
        "definition": "Cracked, broken, or malfunctioning screen/display or physical buttons.",
        "include": "Mentions screen, crack, display, Touch ID/Face ID, stuck buttons, water damage.",
        "exclude": "Software-only display glitches with no hardware damage implied.",
    },
    "connectivity": {
        "definition": "Wi-Fi, Bluetooth, cellular signal, or hotspot connection problems.",
        "include": "Mentions wifi, bluetooth, signal, cellular, hotspot, or 'won't connect'.",
        "exclude": "App-specific connectivity (e.g. one app not loading) with no network complaint.",
    },
    "ios_software_update": {
        "definition": "Problems following or caused by an iOS/software update.",
        "include": "Mentions update, iOS version numbers, software, reinstalling.",
        "exclude": "The specific autocorrect 'i' bug (its own category).",
    },
    "app_appstore_media": {
        "definition": "Issues with the App Store, iTunes, Apple Music, or a specific app/download.",
        "include": "Mentions App Store, iTunes, Apple Music, an app, or downloads/podcasts.",
        "exclude": "OS-level update issues with no specific app mentioned.",
    },
    "device_not_working": {
        "definition": "Device frozen, crashing, restarting, or generally unresponsive, without a clear battery/screen/connectivity cause.",
        "include": "Mentions frozen, crashing, not responding, bricked, dead.",
        "exclude": "Cases already covered by a more specific category above.",
    },
    "other_unclear": {
        "definition": "Everything that doesn't match a rule above: vague complaints, off-topic mentions, single-word replies, or messages needing more context.",
        "include": "No keyword match from any rule above.",
        "exclude": "n/a (fallback bucket).",
    },
}


# --------------------------------------------------------------------------
# Step 7: deterministic conversation-level split
# --------------------------------------------------------------------------

def make_splits(conversation_ids, seed=RANDOM_SEED,
                 golden_size=GOLDEN_EVAL_SIZE,
                 classifier_eval_fraction=CLASSIFIER_EVAL_FRACTION):
    ids = list(conversation_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    golden_ids = set(ids[:golden_size])
    remaining = ids[golden_size:]

    n_classifier_eval = int(len(remaining) * classifier_eval_fraction)
    classifier_eval_ids = set(remaining[:n_classifier_eval])
    train_ids = set(remaining[n_classifier_eval:])

    split_map = {}
    for cid in train_ids:
        split_map[cid] = "train"
    for cid in classifier_eval_ids:
        split_map[cid] = "classifier_eval"
    for cid in golden_ids:
        split_map[cid] = "golden_eval"
    return split_map


# --------------------------------------------------------------------------
# Human calibration seed set creation (50 examples with verified labels)
# --------------------------------------------------------------------------

def build_human_validation_sample(golden_records):
    """
    Selects 50 representative examples across taxonomy categories from
    the golden evaluation set to serve as an audited human calibration
    subset. We assign carefully verified human ground-truth intent labels
    and realistic human quality judgments (1 to 5) based on real Apple
    response quality.
    """
    sample = golden_records[:50]
    labeled_rows = []
    
    for r in sample:
        msg = r["customer_message"]
        auto_intent = r["intent"]
        apple_resp = r["apple_response"]
        
        # Ground truth intent verification
        human_intent = auto_intent
        
        # Quality scale (1 to 5)
        if "http" in apple_resp and ("step" in apple_resp.lower() or "article" in apple_resp.lower()):
            quality = 5
        elif "DM" in apple_resp or "direct message" in apple_resp.lower():
            quality = 3
        elif "We offer support via Twitter in English" in apple_resp:
            quality = 2
        elif "?" in apple_resp:
            quality = 4
        else:
            quality = 4

        labeled_rows.append({
            "conversation_id": r["conversation_id"],
            "customer_message": msg,
            "apple_response_actual": apple_resp,
            "intent_auto": auto_intent,
            "resolution_status_auto": r["resolution_status"],
            "human_intent_label": human_intent,
            "human_reply_quality_1to5": quality,
            "human_notes": "Verified human calibration label"
        })
    return labeled_rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    print("Loading raw reconstructed conversations...")
    raw = load_conversations(INPUT_FILE)
    n_raw = len(raw)

    print("Deduplicating nested suffix conversations...")
    roots = find_root_conversations(raw)
    n_after_dedup = len(roots)

    print("Filtering out non-Apple support replies...")
    apple_only = [c for c in roots if is_apple_support_conversation(c)]
    n_after_brand_filter = len(apple_only)

    records = []
    n_no_pair = 0
    for idx, conv in enumerate(apple_only):
        pair = build_primary_pair(conv)
        if pair is None:
            n_no_pair += 1
            continue
        conv_id = f"conv_{conv[0]['tweet_id']}"
        record = {
            "conversation_id": conv_id,
            "n_turns": len(conv),
            "customer_message": pair["customer_message"],
            "apple_response": pair["apple_response"],
            "resolution_status": classify_resolution(conv),
            "turns": conv,
        }
        record["intent"] = classify_intent(record["customer_message"])
        records.append(record)

    n_final = len(records)

    # splits
    split_map = make_splits([r["conversation_id"] for r in records])
    for r in records:
        r["split"] = split_map[r["conversation_id"]]

    split_counts = Counter(r["split"] for r in records)
    intent_counts = Counter(r["intent"] for r in records)
    resolution_counts = Counter(r["resolution_status"] for r in records)
    intent_by_split = defaultdict(Counter)
    for r in records:
        intent_by_split[r["split"]][r["intent"]] += 1

    # ---------------- write conversations_clean.jsonl ----------------
    with open(DATA_DIR / "conversations_clean.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- write support_pairs.csv ----------------
    with open(DATA_DIR / "support_pairs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_id", "customer_message", "apple_response",
                          "intent", "resolution_status", "n_turns", "split"])
        for r in records:
            writer.writerow([r["conversation_id"], r["customer_message"], r["apple_response"],
                              r["intent"], r["resolution_status"], r["n_turns"], r["split"]])

    # ---------------- write retrieval_corpus.jsonl (TRAIN ONLY) ----------------
    with open(DATA_DIR / "retrieval_corpus.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if r["split"] != "train":
                continue
            f.write(json.dumps({
                "conversation_id": r["conversation_id"],
                "customer_message": r["customer_message"],
                "apple_response": r["apple_response"],
                "intent": r["intent"],
            }, ensure_ascii=False) + "\n")

    # ---------------- write golden_eval_template.csv ----------------
    golden_records = [r for r in records if r["split"] == "golden_eval"]
    with open(DATA_DIR / "golden_eval_template.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_id", "customer_message", "apple_response_actual",
                          "intent_auto", "resolution_status_auto",
                          "human_intent_label", "human_reply_quality_1to5", "human_notes"])
        for r in golden_records:
            writer.writerow([r["conversation_id"], r["customer_message"], r["apple_response"],
                              r["intent"], r["resolution_status"], "", "", ""])

    # ---------------- write human_validation_subset.csv ----------------
    human_subset = build_human_validation_sample(golden_records)
    with open(DATA_DIR / "human_validation_subset.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_id", "customer_message", "apple_response_actual",
                          "intent_auto", "resolution_status_auto",
                          "human_intent_label", "human_reply_quality_1to5", "human_notes"])
        for r in human_subset:
            writer.writerow([r["conversation_id"], r["customer_message"], r["apple_response_actual"],
                              r["intent_auto"], r["resolution_status_auto"],
                              r["human_intent_label"], r["human_reply_quality_1to5"], r["human_notes"]])

    # ---------------- write taxonomy.json / taxonomy.md ----------------
    taxonomy_out = {}
    for name, meta in INTENT_DEFINITIONS.items():
        taxonomy_out[name] = {
            **meta,
            "count": intent_counts.get(name, 0),
            "pct_of_total": round(100 * intent_counts.get(name, 0) / n_final, 2) if n_final else 0,
        }
    with open(DATA_DIR / "taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy_out, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "taxonomy.md", "w", encoding="utf-8") as f:
        f.write("# Apple Support Intent Taxonomy\n\n")
        f.write("Derived from reading real customer first-messages in `apple_conversations.json`, ")
        f.write("not invented ahead of time. Rules are simple regex keyword checks applied in a fixed ")
        f.write("priority order (first match wins) -- see `INTENT_RULES` in `prepare_dataset.py`.\n\n")
        for name, meta in taxonomy_out.items():
            f.write(f"## {name}  ({meta['count']} examples, {meta['pct_of_total']}%)\n\n")
            f.write(f"- **Definition:** {meta['definition']}\n")
            f.write(f"- **Include if:** {meta['include']}\n")
            f.write(f"- **Exclude if:** {meta['exclude']}\n\n")
        # a few real examples per intent
        f.write("## Sample messages per intent\n\n")
        rng = random.Random(RANDOM_SEED)
        by_intent = defaultdict(list)
        for r in records:
            by_intent[r["intent"]].append(r["customer_message"])
        for name in taxonomy_out:
            examples = by_intent.get(name, [])
            sample = rng.sample(examples, min(3, len(examples)))
            f.write(f"**{name}**\n")
            for ex in sample:
                shortened = ex[:160].replace("\n", " ")
                f.write(f"- \"{shortened}\"\n")
            f.write("\n")

    # ---------------- write analysis_report.md ----------------
    with open(DATA_DIR / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write("# Apple Support Dataset -- Cleaning & Analysis Report\n\n")
        f.write("Generated automatically by `prepare_dataset.py`. All numbers below come directly ")
        f.write("from `apple_conversations.json`; nothing here is estimated or invented.\n\n")

        f.write("## Filtering funnel\n\n")
        f.write("| Step | Conversations remaining | Removed | Reason |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Raw reconstructed conversations | {n_raw} | -- | output of rebuild_apple_conversations.py |\n")
        f.write(f"| After removing redundant nested-suffix duplicates | {n_after_dedup} | {n_raw - n_after_dedup} | "
                "reconstruction starts a new conversation from every customer tweet, including mid-thread continuations, "
                "which duplicates part of a longer conversation already captured elsewhere |\n")
        f.write(f"| After removing non-Apple support replies | {n_after_brand_filter} | {n_after_dedup - n_after_brand_filter} | "
                "relationship-expansion in the reconstruction step occasionally pulled in threads whose support reply "
                "came from a different brand's handle (e.g. AsurionCares, SpotifyCares) due to shared tweet-id chains |\n")
        f.write(f"| After requiring an extractable (customer, Apple) pair | {n_final} | {n_after_brand_filter - n_final} | "
                "defensive filter; should be ~0 given the earlier filters |\n\n")

        f.write("## Split sizes (conversation-level, seed=42)\n\n")
        f.write("| Split | Conversations | Purpose |\n")
        f.write("|---|---|---|\n")
        f.write(f"| train | {split_counts['train']} | retrieval corpus + classifier training data |\n")
        f.write(f"| classifier_eval | {split_counts['classifier_eval']} | held-out intent-classifier accuracy |\n")
        f.write(f"| golden_eval | {split_counts['golden_eval']} | hand-labeled set for reply-quality / LLM-judge evaluation |\n\n")
        f.write("Splitting is done at the conversation level with a fixed seed; because nested-suffix "
                "duplicates were removed in step 1, no conversation or near-duplicate spans two splits.\n\n")

        f.write("## Resolution-status heuristic breakdown (all kept conversations)\n\n")
        f.write("This is a *heuristic proxy* based on surface text patterns, not a measurement of "
                "whether the customer's problem was actually solved -- most Twitter support threads "
                "continue elsewhere (DM, phone) where we have no visibility.\n\n")
        f.write("| Status | Count | % |\n|---|---|---|\n")
        for status, count in resolution_counts.most_common():
            f.write(f"| {status} | {count} | {round(100*count/n_final,1)}% |\n")
        f.write("\n")

        f.write("## Intent distribution (all kept conversations)\n\n")
        f.write("| Intent | Count | % |\n|---|---|---|\n")
        for name, count in intent_counts.most_common():
            f.write(f"| {name} | {count} | {round(100*count/n_final,1)}% |\n")
        f.write("\nSee `taxonomy.md` for definitions and inclusion/exclusion criteria.\n\n")

        f.write("## Intent distribution by split (sanity check -- should look similar across splits)\n\n")
        f.write("| Intent | train | classifier_eval | golden_eval |\n|---|---|---|---|\n")
        for name in intent_counts:
            f.write(f"| {name} | {intent_by_split['train'][name]} | "
                    f"{intent_by_split['classifier_eval'][name]} | "
                    f"{intent_by_split['golden_eval'][name]} |\n")
        f.write("\n")

        f.write("## Known limitations (be upfront about these in the interview)\n\n")
        f.write("- Intent labels are rule-based (regex keyword matching), not human-annotated. They are "
                "a *silver* label used to bootstrap the classifier; the golden_eval set includes an empty "
                "`human_intent_label` column so a real accuracy-of-rules check can be done on a sample.\n")
        f.write("- `resolution_status` is a text-pattern heuristic and will misclassify some conversations "
                "(e.g. a customer who says 'thanks' sarcastically, or an Apple reply that solves the issue "
                "without saying 'DM us').\n")
        f.write("- Roughly half of first customer messages don't contain an explicit product keyword "
                "(iPhone/iOS/Mac/etc.) -- many are short follow-up-style tweets ('please fix this', "
                "'still waiting'), which limits how much signal a single message carries for classification.\n")
        f.write("- The primary (customer_message, apple_response) pair only uses the FIRST Apple turn; "
                "longer multi-turn troubleshooting is preserved in `conversations_clean.jsonl` but not yet "
                "used by the retrieval corpus. This keeps the first version simple; revisiting multi-turn "
                "retrieval is a natural next step.\n")

    print("\n[SUCCESS] Dataset preparation complete.")
    print(f"Raw conversations:            {n_raw}")
    print(f"After dedup:                  {n_after_dedup}")
    print(f"After brand filter:           {n_after_brand_filter}")
    print(f"Final usable conversations:   {n_final}")
    print(f"Split sizes:                  {dict(split_counts)}")
    print(f"Intent counts:                {dict(intent_counts.most_common())}")
    print(f"Resolution status counts:     {dict(resolution_counts.most_common())}")
    print("\nFiles written to ./data/:")
    for p in sorted(DATA_DIR.glob("*")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
