# Engineering Report: AI Customer Support Agent for Apple Support

**Author:** Priya Pandey  
**Project:** AI Customer Support Agent (Hiver SDE Intern Assignment)  
**Dataset:** Kaggle "Customer Support on Twitter" by thoughtvector (`@AppleSupport` subset)  
**Evaluation Command:** `python run_evaluation.py` (Runtime: ~17 seconds on CPU)  

---

## 1. Problem Framing

Customer support on public social media (Twitter / X) presents unique challenges compared to traditional ticketing systems:
1. **Severe Character and Format Constraints:** Messages are concise, frequently split across multiple consecutive tweets, and often contain screenshots, emojis, or jargon without formal diagnostic headers.
2. **Asymmetric Brand Risk:** Public responses are visible to millions. Incorrect technical guidance, false warranty guarantees, or automated brush-offs cause immediate public relations damage.
3. **High Channel Deflection vs. Safety Tradeoff:** While routine troubleshooting (battery calibration, Wi-Fi resets, iOS update procedures) can be safely automated, sensitive inquiries (Apple ID lockouts, billing disputes, shattered glass) must be escalated to human agents or secure Direct Message channels immediately.

### Operational Scope
I framed the system as an **Autonomous Triage and Assisted Drafting Agent** for Apple Support. Rather than operating as an unconstrained chatbot, the agent executes a structured 4-stage pipeline:
$$\text{Customer Tweet} \longrightarrow \text{Intent Classification} \longrightarrow \text{Historical Retrieval} \longrightarrow \text{Explainable Escalation} \longrightarrow \text{Grounded Generation}$$

The unit of interaction is the customer’s opening complaint (with leading multi-tweet threads concatenated into a single message) and the agent’s drafted opening reply.

---

## 2. What "Good" Means

In production customer support, a high automated deflection rate is worthless if the responses are ungrounded or risky. I define "Good" across four objective dimensions:

1. **Groundedness Over Creativity:** An agent must not hallucinate policies, invent iOS version release dates, or promise hardware repairs not authorized by Apple. Good responses mirror the tone, diagnostic precision, and official documentation links historically used by Apple Support.
2. **Explainable, Conservative Escalation:** Every escalation decision must be accompanied by an auditable reason string. 100% of sensitive queries (account compromise, fraudulent billing, shattered screens) must be escalated to human specialists, even if retrieval similarity is high.
3. **High Intent Recall on Actionable Categories:** Misclassifying a battery drain query as an autocorrect bug ruins customer trust. High precision and recall on critical categories are non-negotiable.
4. **Reproducibility Under Resource Constraints:** A robust student engineering project must run cleanly on standard developer hardware (CPU-only, Windows/Linux/macOS), complete evaluation benchmarks in minutes, and avoid fragile dependencies.

---

## 3. What Was NOT Built (Intentional Non-Goals)

To deliver a reliable, interview-defensible system within assignment constraints, I deliberately excluded the following features:

1. **Complex Multi-Agent Orchestration / Swarms:** Frameworks like CrewAI or AutoGen introduce unpredictable non-deterministic loops, token costs, and difficult debugging. I prioritized a modular, deterministic pipeline.
2. **End-to-End Deep Learning Classifier (Fine-Tuned BERT/LLM):** Training a full transformer sequence classifier on CPU is slow and opaque. My hybrid system (domain regex rules + TF-IDF/Logistic Regression) achieves 95.5% accuracy with sub-millisecond inference and complete explainability.
3. **Multi-Turn Session State Machine:** Twitter conversations frequently fork or transition to private DMs. The current system models single-turn opening triage; tracking multi-turn conversation states across DMs was reserved for future work.
4. **Image / Screenshot OCR Understanding:** Many customer tweets attach error screenshots without text. Incorporating an OCR pipeline was deferred to the next iteration.
5. **Modern Apple Knowledge Base:** The dataset reflects late-2017 Twitter interactions (iOS 11 era). The system was intentionally not modified to invent modern iOS 18 knowledge.
6. **Live Twitter API Integration:** Twitter's commercial API requires enterprise licensing. Kaggle historical data was used as a self-contained offline corpus.

---

## 4. Headline Results vs. Trivial & Simple Baselines

All metrics below were computed on the held-out `classifier_eval` set (N=3,448), the FAISS retrieval index (N=19,545 train conversations), and the curated `golden_eval` set (N=200) using `python run_evaluation.py`.

### A. Intent Classification Benchmark

| Model | Accuracy | Macro F1 | Weighted F1 | Latency / Sample |
|---|---|---|---|---|
| **Trivial Baseline (Majority Class)** | 30.83% | 0.0471 | 0.1453 | < 0.01 ms |
| **Simple ML Baseline (TF-IDF + LogReg)** | 84.89% | 0.8418 | 0.8531 | 0.15 ms |
| **Rule-Based Taxonomy (Silver Labels)** | 100.00%* | 1.0000* | 1.0000* | 0.02 ms |
| **Production Hybrid (Rules + ML Fallback)** | **95.50%** | **0.9680** | **0.9566** | **0.05 ms** |

*\*Methodological Note on Intent Metrics: All four models above were evaluated against rule-derived silver labels on the held-out split (N=3,448). The Rule-Based score (100.0%) is circular because silver labels were bootstrapped from those exact rules. The Production Hybrid score (95.5%) measures taxonomy consistency and recovery from uninformative tweets, NOT independent human ground-truth accuracy. See Section 6 for full critique.*

#### Per-Intent Breakdown (Production Hybrid on Silver Held-Out Split)
| Intent | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| `autocorrect_text_bug` | 0.707 | 1.000 | 0.828 | 318 |
| `account_security_icloud` | 0.987 | 1.000 | 0.993 | 149 |
| `billing_repair_order` | 0.978 | 1.000 | 0.989 | 89 |
| `battery_power` | 0.997 | 1.000 | 0.999 | 355 |
| `screen_hardware_damage` | 0.990 | 1.000 | 0.995 | 203 |
| `connectivity` | 1.000 | 1.000 | 1.000 | 139 |
| `ios_software_update` | 0.990 | 1.000 | 0.995 | 832 |
| `app_appstore_media` | 0.984 | 1.000 | 0.992 | 241 |
| `device_not_working` | 0.937 | 1.000 | 0.967 | 59 |
| `other_unclear` | 1.000 | 0.854 | 0.921 | 1,063 |

---

### B. Historical Retrieval Benchmark (FAISS IndexFlatIP)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **Retrieval Corpus Size:** 19,545 conversations (Train split only — zero test leakage)
- **Evaluated Test Queries:** 500 held-out customer inquiries
- **Retrieval Hit@1 (Same Intent):** **67.80%**
- **Retrieval Hit@3 (Same Intent):** **86.80%**
- **Retrieval Hit@5 (Same Intent):** **91.60%**
- **Mean Top-1 Cosine Similarity:** **0.7695**

*Note on Retrieval Hit@K: Hit@K measures whether at least one retrieved interaction shares the customer's true INTENT topic. It is NOT proof that the retrieved response contains the exact device-specific solution.*

---

### C. Explainable Escalation Benchmark (Golden Set N=200)
- **Auto-Handled Rate:** **56.50%** (113 / 200)
- **Escalated to Human:** **43.50%** (87 / 200)
- **Brand Safety & Risk Routing:** **100.0%** (19 / 19 sensitive security, billing, and screen damage inquiries were correctly escalated to human specialists).

---

### D. Reply Quality (LLM-as-a-Judge Rubric, Scale 1 to 5)
Evaluated across 50 golden inquiries:
- **Relevance:** **3.96 / 5.0**
- **Groundedness:** **4.50 / 5.0**
- **Helpfulness:** **4.80 / 5.0**
- **Completeness:** **4.84 / 5.0**
- **Hallucination / Unsupported Claims Rate:** **0.0%**
- **Average Overall Score:** **4.46 / 5.0**

---

### E. LLM Judge vs. Human Agreement (N=50 Calibration Set)
To test whether the LLM Judge was aligned with human evaluation, ratings were compared on an audited calibration sample of 50 interactions:
- **Exact Rating Agreement:** **50.00%**
- **Agreement within $\pm 1$ Point:** **94.00%**
- **Mean Absolute Error (MAE):** **0.560**
- **Spearman Rank Correlation ($\rho$):** **0.1317**
- **Quadratic Weighted Cohen's Kappa ($\kappa$):** **0.1062**

#### Critical Analysis of Judge Agreement Metrics:
1. **Exact agreement is only 50.0%:** In half the cases, the judge differs from the human score.
2. **The 94.0% within $\pm 1$ point metric is inflated due to severe score compression:** Because almost all polite Apple responses receive ratings clustered in {4, 5}, the absolute difference is mathematically almost always $\le 1$.
3. **Low Spearman rank correlation ($\rho = 0.1317$):** The judge struggles to order fine differences in response quality.
4. **Low Kappa ($\kappa = 0.1062$):** Agreement beyond chance is slight.
5. **Takeaway:** The 4.46/5 score is a coarse diagnostic for polite tone and grounding, NOT definitive proof of perfect replies.

---

## 5. Golden Evaluation Set & Hand-Labeling Status

The assignment specifies a golden evaluation set of 150–250 hand-labelled examples.

### Current Status:
- `data/golden_eval_template.csv` contains **200 sampled conversations**.
- Columns `conversation_id`, `customer_message`, `apple_response_actual`, `intent_auto`, and `resolution_status_auto` are populated.
- Columns `human_intent_label`, `human_reply_quality_1to5`, and `human_notes` are currently blank.
- `data/human_validation_subset.csv` contains **50 calibration examples** that were provisionally reviewed during initial harness development.
- **Transparent Status:** 200 examples were sampled for the golden evaluation set; 50 were provisionally reviewed during initial calibration. The remaining examples require independent human annotation before this can be considered a fully hand-labelled golden set.

### Sampling Methodology:
1. **Unit of Sampling:** Complete root conversations from `apple_conversations.json` after nested-suffix deduplication.
2. **Partitioning:** A deterministic pseudo-random partition (`seed=42`) split conversation IDs into train, classifier_eval, and golden_eval.
3. **Intent Distribution:** Stratification was not applied programmatically; rather, uniform random sampling across the deduplicated corpus naturally mirrored the empirical intent distribution (`other_unclear` ~31%, `ios_software_update` ~25%, `battery_power` ~10%, etc.).
4. **Leakage Prevention:** Partitioning strictly at the conversation level ensured that no tweet from `golden_eval` appears in `train` or the FAISS index.

### Local Tooling for Completing Labels:
I created an interactive terminal labeling script:
```bash
python label_golden_set.py
```
This tool steps through unlabeled rows, displays the customer tweet and Apple response, accepts 1-key inputs for intent (0–9) and quality (1–5), and saves continuously.

---

## 6. Top 5 Meaningful Failure Modes

From automated error analysis across the evaluation run, I identified five recurring real failure modes:

### [FM-01] Ultra-Short / Image-Only Tweets Lacking Context
- **Real Example:** `@AppleSupport u too https://t.co/AubRbVtD4f`
- **Classified Intent:** `other_unclear` | **System Action:** `ESCALATE`
- **Why It Failed:** The customer posted a screenshot link without diagnostic text. The text-only classifier cannot parse image contents.
- **Hypothesis:** Customers on Twitter use media attachments as primary communication, assuming human agents will view the image.
- **Concrete Fix:** Integrate an image OCR pipeline to extract text from Twitter media URLs prior to classification.

### [FM-02] Vocabulary-Deficient Matching / Keyword Clashes
- **Real Example:** `"Can't edit Can't share Happens to various photos iOS 11.1.2 6s"`
- **Classified Intent:** `ios_software_update` | **System Action:** `AUTO-HANDLE`
- **Why It Failed:** Regex priority checked for `ios 11.1.2` before photos/media, routing a Photos app defect to general OS update triage.
- **Hypothesis:** Rigid priority ordering in rule-based systems penalizes multi-attribute complaints where software version co-occurs with application symptoms.
- **Concrete Fix:** Replace static regex priority with dense embedding classification where sentence semantics weigh app-level symptoms alongside OS tags.

### [FM-03] Historical Temporal Policy Drift
- **Real Example:** `"fix y'all damn software. 'A ?' I'm sick of this!"`
- **Classified Intent:** `autocorrect_text_bug` | **System Action:** `AUTO-HANDLE`
- **Why It Failed:** The agent retrieved an Apple response from November 2017 recommending a temporary text replacement workaround for the iOS 11.1 letter "I" glitch.
- **Hypothesis:** In static datasets, historical workarounds become obsolete when permanent OS patches are released.
- **Concrete Fix:** Add timestamp and knowledge cutoff metadata to retrieval vectors, prioritizing timeless advice over temporary patch workarounds.

### [FM-04] Complex Multi-Issue Symptoms With Competing Routing
- **Real Example:** `"Phone freezing, won't charge. Dad won't let me upgrade so chill please."`
- **Classified Intent:** `battery_power` | **System Action:** `AUTO-HANDLE`
- **Why It Failed:** The customer describes both a charging issue (`battery_power`) and system unresponsiveness (`device_not_working`). The system handled it as a battery issue, ignoring the freeze.
- **Hypothesis:** Single-label classification inherently discards secondary technical symptoms.
- **Concrete Fix:** Implement multi-label classification; trigger mandatory escalation whenever two distinct hardware/software intents co-occur.

### [FM-05] Sarcastic Gratitude Inverting Sentiment Filters
- **Real Example:** `"Gee thanks @AppleSupport for fixing NONE of the problems I've been experiencing"`
- **Classified Intent:** `other_unclear` | **System Action:** `ESCALATE`
- **Why It Failed:** Keyword filters detecting `"thanks"` can falsely flag the conversation as resolved (`customer_thanked`).
- **Hypothesis:** Surface heuristic resolution tagging is brittle against English sarcasm.
- **Concrete Fix:** Use a fine-tuned sentiment classifier to distinguish genuine gratitude from frustration before tagging resolution status.

---

## 7. "What is Misleading About My Headline Number?"

*(Mandatory Critical Reflection)*

1. **Circular Silver Labeling:** My hybrid classifier achieves 95.5% accuracy on `classifier_eval`. However, because ground truth labels in `classifier_eval` were generated via regex rules, this metric measures consistency with the taxonomy rules, not independent human accuracy.
2. **Retrieval Hit@K Measures Intent, Not Solution Quality:** A 91.6% Hit@5 means that an example with the same *topic* (e.g. `battery_power`) was retrieved, not necessarily that Apple's reply answered the customer's specific hardware/software issue.
3. **Retaining Unresolved Conversations:** Apple's historical reply was often just asking for the iOS version or requesting a DM. The agent occasionally mimics this clarifying step rather than providing an instant technical fix.
4. **Score Compression in LLM Judge:** As detailed in Section 4.E, the 94.0% agreement within $\pm 1$ point is largely an artifact of ratings clustering in {4, 5}. The low ranking correlation ($\rho = 0.1317$) proves that the judge cannot be trusted to order subtle quality gradations.
5. **Point-in-Time Dataset Bias (Late 2017):** The corpus captures late 2017 interactions (iOS 11 launch). Generalizing this model to modern devices requires ingesting modern Apple Support articles.

---

## 8. Next-Week Plan (Evaluation-First Roadmap)

If granted an additional week of development, my roadmap would focus on:

- **Day 1–2: Complete Full Golden Set Annotation:** Use `label_golden_set.py` to complete independent human annotation of all 200 golden evaluation items, expanding inter-rater agreement sample size from N=50 to N=200.
- **Day 3: Compound / Multi-Issue Intent Modeling:** Improve intent classification for multi-issue messages using multi-label heads or dense embedding classification.
- **Day 4: Temporal Decay in Retrieval:** Add temporal awareness to retrieval so old troubleshooting workarounds are down-weighted in favor of evergreen knowledge base articles.
- **Day 5: OCR & Multimodal Extraction:** Add image and screenshot OCR processing for media attachments in tweets.
- **Day 6–7: Multi-Turn Conversation State:** Add lightweight persistent conversation state for multi-turn customer dialogues.

---

## 9. Acknowledgments / Sources

- **Dataset:** Kaggle "Customer Support on Twitter" dataset by thoughtvector (Creative Commons).
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` by HuggingFace / UKPLab.
- **Vector Search:** FAISS (`faiss-cpu`) by Meta Research.
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF vectorizer and Logistic Regression).
- **Web UI:** Streamlit.
- **AI Coding Assistance:** AI coding assistants were used during development to assist with syntax, boilerplate, and rapid iteration; all architectural decisions, taxonomy designs, data filtering rules, and evaluation audits were designed and verified by the author.
