"""
label_golden_set.py
===================

Interactive CLI tool for Priya to hand-label the 200-example Golden Evaluation Set.

Allows rapid, keyboard-driven annotation of:
1. Intent (keys 0-9)
2. Reply Quality (keys 1-5)
3. Optional notes / escalation audit

Saves progress continuously to `data/golden_eval_template.csv`.

Usage:
    python label_golden_set.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/golden_eval_template.csv")

INTENT_OPTIONS = [
    ("1", "autocorrect_text_bug", "iOS 11.1 keyboard/autocorrect 'i' defect"),
    ("2", "account_security_icloud", "Apple ID, iCloud, 2FA, password, account lockout"),
    ("3", "billing_repair_order", "Refunds, charges, warranty, repairs, Genius Bar"),
    ("4", "battery_power", "Battery drain, charging failure, shutting down"),
    ("5", "screen_hardware_damage", "Cracked screen, display, Touch/Face ID, buttons"),
    ("6", "connectivity", "Wi-Fi, Bluetooth, cellular data, hotspot, signal"),
    ("7", "ios_software_update", "iOS/macOS update installation and general update bugs"),
    ("8", "app_appstore_media", "App Store, Apple Music, iTunes, app crashes"),
    ("9", "device_not_working", "Frozen, unbootable, crashing device"),
    ("0", "other_unclear", "Vague complaints, image links, greeting, off-topic"),
]

INTENT_MAP = {k: name for k, name, _ in INTENT_OPTIONS}


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    
    # Check already labeled
    labeled_mask = df["human_intent_label"].notna() & (df["human_intent_label"].str.strip() != "")
    labeled_count = labeled_mask.sum()
    
    print("\n" + "=" * 80)
    print("      GOLDEN EVALUATION SET - INTERACTIVE HAND-LABELING TOOL")
    print("=" * 80)
    print(f"Total examples: {total}")
    print(f"Already labeled: {labeled_count} / {total} ({labeled_count/total*100:.1f}%)")
    print(f"Remaining to label: {total - labeled_count}")
    print("Instructions:")
    print(" - Press 0-9 to select an intent (or Enter to accept auto/silver intent)")
    print(" - Press 1-5 for Apple response quality (1=Harmful/Fake, 3=Generic DM, 5=Excellent)")
    print(" - Type 's' to skip, or 'q' to save and exit at any time.")
    print("=" * 80 + "\n")

    for idx, row in df.iterrows():
        # Skip already labeled rows unless user wants to re-check
        current_human_intent = str(row.get("human_intent_label", "")).strip()
        if current_human_intent and current_human_intent != "nan":
            continue

        cid = row["conversation_id"]
        cust_msg = str(row["customer_message"])
        apple_resp = str(row["apple_response_actual"])
        auto_intent = str(row["intent_auto"])

        print("-" * 80)
        print(f"[{idx + 1} / {total}] Conversation ID: {cid}")
        print(f"CUSTOMER:       {cust_msg}")
        print(f"APPLE RESPONSE: {apple_resp}")
        print(f"Provisional Auto-Intent: [{auto_intent}]")
        print("\nChoose Intent:")
        for key, name, desc in INTENT_OPTIONS:
            marker = "*" if name == auto_intent else " "
            print(f"  [{key}] {name:<24} {marker} ({desc})")

        # Prompt for Intent
        intent_choice = None
        while True:
            raw = input(f"\nSelect Intent (0-9, Enter for '{auto_intent}', s=skip, q=quit): ").strip()
            if raw.lower() == "q":
                print("\nSaving progress and quitting...")
                df.to_csv(CSV_PATH, index=False)
                print(f"Saved! Total labeled now: {df['human_intent_label'].notna().sum()} / {total}")
                return
            elif raw.lower() == "s":
                print("Skipped.")
                break
            elif raw == "":
                intent_choice = auto_intent
                break
            elif raw in INTENT_MAP:
                intent_choice = INTENT_MAP[raw]
                break
            else:
                print("Invalid choice. Enter a digit 0-9, press Enter, or 'q' to quit.")

        if raw.lower() == "s":
            continue

        # Prompt for Quality (1 to 5)
        print("\nRate Apple's Response Quality (1 to 5):")
        print("  [5] Excellent: Solves issue with specific settings/article link")
        print("  [4] Good: Clear troubleshooting question or diagnostic step")
        print("  [3] Acceptable: Generic question or asks to Direct Message (DM)")
        print("  [2] Poor: Irrelevant or unhelpful brush-off")
        print("  [1] Unacceptable: Hallucinatory or harmful claim")
        
        quality_choice = None
        while True:
            q_raw = input("Enter Quality (1-5, Enter for default 4, s=skip): ").strip()
            if q_raw == "":
                quality_choice = 4
                break
            elif q_raw.lower() == "s":
                quality_choice = None
                break
            elif q_raw in ["1", "2", "3", "4", "5"]:
                quality_choice = int(q_raw)
                break
            else:
                print("Invalid quality score. Enter a number 1 to 5.")

        # Prompt for Notes
        notes = input("Optional notes (Enter to leave blank): ").strip()

        # Update DataFrame
        df.at[idx, "human_intent_label"] = intent_choice
        if quality_choice is not None:
            df.at[idx, "human_reply_quality_1to5"] = quality_choice
        if notes:
            df.at[idx, "human_notes"] = notes

        # Save immediately to avoid losing work
        df.to_csv(CSV_PATH, index=False)
        print(f"-> Saved example {idx + 1} ({intent_choice}, Quality: {quality_choice})")

    print("\n" + "=" * 80)
    print("All 200 examples have been reviewed!")
    print(f"Saved to {CSV_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
