# Apple Support Dataset -- Cleaning & Analysis Report

Generated automatically by `prepare_dataset.py`. All numbers below come directly from `apple_conversations.json`; nothing here is estimated or invented.

## Filtering funnel

| Step | Conversations remaining | Removed | Reason |
|---|---|---|---|
| Raw reconstructed conversations | 32487 | -- | output of rebuild_apple_conversations.py |
| After removing redundant nested-suffix duplicates | 23252 | 9235 | reconstruction starts a new conversation from every customer tweet, including mid-thread continuations, which duplicates part of a longer conversation already captured elsewhere |
| After removing non-Apple support replies | 23193 | 59 | relationship-expansion in the reconstruction step occasionally pulled in threads whose support reply came from a different brand's handle (e.g. AsurionCares, SpotifyCares) due to shared tweet-id chains |
| After requiring an extractable (customer, Apple) pair | 23193 | 0 | defensive filter; should be ~0 given the earlier filters |

## Split sizes (conversation-level, seed=42)

| Split | Conversations | Purpose |
|---|---|---|
| train | 19545 | retrieval corpus + classifier training data |
| classifier_eval | 3448 | held-out intent-classifier accuracy |
| golden_eval | 200 | hand-labeled set for reply-quality / LLM-judge evaluation |

Splitting is done at the conversation level with a fixed seed; because nested-suffix duplicates were removed in step 1, no conversation or near-duplicate spans two splits.

## Resolution-status heuristic breakdown (all kept conversations)

This is a *heuristic proxy* based on surface text patterns, not a measurement of whether the customer's problem was actually solved -- most Twitter support threads continue elsewhere (DM, phone) where we have no visibility.

| Status | Count | % |
|---|---|---|
| unresolved_no_signal | 14071 | 60.7% |
| escalated_to_dm | 4955 | 21.4% |
| customer_thanked | 4164 | 18.0% |
| routed_elsewhere | 3 | 0.0% |

## Intent distribution (all kept conversations)

| Intent | Count | % |
|---|---|---|
| other_unclear | 7387 | 31.9% |
| ios_software_update | 5763 | 24.8% |
| battery_power | 2350 | 10.1% |
| autocorrect_text_bug | 2035 | 8.8% |
| app_appstore_media | 1489 | 6.4% |
| screen_hardware_damage | 1303 | 5.6% |
| connectivity | 946 | 4.1% |
| account_security_icloud | 927 | 4.0% |
| billing_repair_order | 578 | 2.5% |
| device_not_working | 415 | 1.8% |

See `taxonomy.md` for definitions and inclusion/exclusion criteria.

## Intent distribution by split (sanity check -- should look similar across splits)

| Intent | train | classifier_eval | golden_eval |
|---|---|---|---|
| autocorrect_text_bug | 1697 | 318 | 20 |
| other_unclear | 6257 | 1063 | 67 |
| ios_software_update | 4875 | 832 | 56 |
| screen_hardware_damage | 1092 | 203 | 8 |
| account_security_icloud | 771 | 149 | 7 |
| device_not_working | 353 | 59 | 3 |
| battery_power | 1976 | 355 | 19 |
| billing_repair_order | 485 | 89 | 4 |
| app_appstore_media | 1239 | 241 | 9 |
| connectivity | 800 | 139 | 7 |

## Known limitations (be upfront about these in the interview)

- Intent labels are rule-based (regex keyword matching), not human-annotated. They are a *silver* label used to bootstrap the classifier; the golden_eval set includes an empty `human_intent_label` column so a real accuracy-of-rules check can be done on a sample.
- `resolution_status` is a text-pattern heuristic and will misclassify some conversations (e.g. a customer who says 'thanks' sarcastically, or an Apple reply that solves the issue without saying 'DM us').
- Roughly half of first customer messages don't contain an explicit product keyword (iPhone/iOS/Mac/etc.) -- many are short follow-up-style tweets ('please fix this', 'still waiting'), which limits how much signal a single message carries for classification.
- The primary (customer_message, apple_response) pair only uses the FIRST Apple turn; longer multi-turn troubleshooting is preserved in `conversations_clean.jsonl` but not yet used by the retrieval corpus. This keeps the first version simple; revisiting multi-turn retrieval is a natural next step.
