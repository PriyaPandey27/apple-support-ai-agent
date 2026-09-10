# Apple Support AI Agent

A multi-stage customer support triage and response generation system trained on Kaggle's "Customer Support on Twitter" dataset (@AppleSupport). The agent classifies customer complaints into a 10-intent taxonomy, retrieves historically verified Apple Support interactions via FAISS, applies explainable brand-safety escalation rules, and drafts grounded troubleshooting replies.

**Corpus:** Kaggle "Customer Support on Twitter" by thoughtvector (`@AppleSupport` subset)  
**Headline Benchmark Runtime:** ~17 seconds on CPU (Target: < 15 minutes)

---

## 1. Problem Framing / What "Good" Means

Customer support on Twitter presents unique operational risks. Unlike private email or chat tickets, tweets are publicly visible, character-constrained (280 characters), and frequently contain incomplete context, emojis, or external links.

### What "Good" Means for Apple Support:
1. **Accurate Problem Identification:** Correctly categorize the customer's technical domain (e.g. distinguishing a physical screen crack from an iOS update bug).
2. **Historical Grounding:** Retrieve historically relevant @AppleSupport interactions to mirror Apple's proven troubleshooting tone, diagnostic questions, and official knowledge base links.
3. **No Unsupported Claims:** Strictly avoid hallucinating refund guarantees, free hardware replacements, or unannounced software release dates.
4. **Conservative, Explainable Escalation:** Inquiries involving Apple ID account takeover, fraudulent billing, shattered screens, or low retrieval evidence must be escalated to human agents with a human-understandable reason string.
5. **Auditable Reason Strings:** Every routing decision must explain *why* an action was taken, enabling human agents to triage tickets immediately.

### What Was Intentionally NOT Built:
To maintain interview defensibility, strict correctness, and reproducible local execution within assignment constraints, the following were intentionally excluded:
- **Image / Screenshot OCR Understanding:** Twitter inquiries often include screenshot images. The current text-only pipeline flags image-only tweets for human escalation (`other_unclear`).
- **Persistent Multi-Turn Session Memory:** The agent models the opening triage turn. Session tracking across subsequent tweets or private DMs was excluded.
- **Modern Apple Policy Knowledge:** The underlying Kaggle dataset reflects late-2017 interactions (iOS 11 era). The model does not pretend to know modern iOS 18 features or contemporary warranty terms.
- **Multilingual Support:** Tweets in Spanish, French, or Portuguese are detected and escalated or directed to Apple's regional portals.
- **Unconstrained LLM Chatbots:** Using an open-ended LLM without deterministic safety guardrails risks severe brand damage.

---

## 2. Quickstart: Reproducing Headline Results

Clone the repository and run the benchmark:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the headline evaluation benchmark (< 20 seconds)
python run_evaluation.py

# 3. Launch interactive Streamlit demo UI
streamlit run app.py
```

> **Local Execution Note:** The system runs **100% locally on CPU without external API keys or internet connection**, using a built-in offline grounded generator and rubric judge. If you wish to test live cloud generation with Google Gemini, copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

---

## 3. System Architecture

```
                       [ Customer Tweet / Inquiry ]
                                     │
                                     ▼
                     [ Stage 1: Intent Classifier ]
                       ├─ Trivial Baseline: Majority Class (30.8% Acc)
                       ├─ Simple ML Baseline: TF-IDF + LogReg (84.9% Acc)
                       └─ Production Hybrid: Rules + ML Fallback (95.5% Acc*)
                                     │
                                     ▼
                   [ Stage 2: Historical Retrieval ]
                       ├─ Sentence-Transformers ('all-MiniLM-L6-v2')
                       ├─ FAISS IndexFlatIP (Train split only, N=19,545)
                       └─ Hits: 67.8% (Hit@1) | 86.8% (Hit@3) | 91.6% (Hit@5)
                                     │
                                     ▼
                   [ Stage 3: Explainable Escalation ]
                       ├─ Safety Risk Check: Security, Billing, Screen Damage
                       ├─ Evidence Thresholding: Cosine Similarity >= 0.55
                       └─ Output: AUTO-HANDLE (56.5%) vs ESCALATE (43.5%) + Reason
                                     │
                                     ▼
                   [ Stage 4: Grounded Reply Generation ]
                       ├─ Grounds advice strictly in Top-3 historical Apple tweets
                       ├─ Enforces empathetic, concise @AppleSupport tone
                       └─ Restricts fake warranty promises & unsupported claims
                                     │
                                     ▼
                   [ Stage 5: Automated & Human Evaluation ]
                       ├─ LLM-as-a-Judge Rubric (4.46 / 5.0 Overall Quality)
                       └─ Human Agreement Calibration: 94.0% within +/- 1 point
```

---

## 4. Headline Results vs. Baselines

All metrics below were computed on the held-out `classifier_eval` split (N=3,448) and the `golden_eval` split (N=200) via `python run_evaluation.py`:

### Intent Classification Benchmark
| Model | Accuracy | Macro F1 | Weighted F1 | Inference Speed |
|---|---|---|---|---|
| **Trivial Baseline (Majority Class)** | 30.83% | 0.0471 | 0.1453 | < 0.01 ms |
| **Simple ML Baseline (TF-IDF + LogReg)** | 84.89% | 0.8418 | 0.8531 | 0.15 ms |
| **Rule-Based Taxonomy (Silver Labels)** | 100.00%* | 1.0000* | 1.0000* | 0.02 ms |
| **Production Hybrid (Rules + ML Fallback)** | **95.50%** | **0.9680** | **0.9566** | **0.05 ms** |

*\*Important Methodological Note on Intent Metrics: All four models above were evaluated against rule-derived silver labels on the held-out split (N=3,448). The Rule-Based score (100.0%) is circular because silver labels were bootstrapped from those exact rules. The Production Hybrid score (95.5%) measures taxonomy consistency and recovery from uninformative tweets, NOT independent human ground-truth accuracy. See Section 6 for full critique.*

### Historical Retrieval Benchmark (FAISS IndexFlatIP, N=19,545 Train Corpus)
- **Hit@1 (True Intent Topic Retrieved):** **67.80%**
- **Hit@3 (True Intent Topic in Top 3):** **86.80%**
- **Hit@5 (True Intent Topic in Top 5):** **91.60%**
- **Mean Top-1 Cosine Similarity:** **0.7695**

*Note on Retrieval Hit@K: Hit@K measures whether at least one retrieved interaction shares the customer's true INTENT topic. It is NOT proof that the retrieved response contains the exact device-specific solution.*

### Explainable Escalation Benchmark (Golden Set N=200)
- **Auto-Handled Rate:** **56.50%** (113 / 200)
- **Escalated to Human:** **43.50%** (87 / 200)
- **Brand Safety & Risk Routing:** **100.0%** (19 / 19 high-risk inquiries—Apple ID security, billing fraud, shattered screens—were routed to human specialists).

### Reply Quality (LLM-as-a-Judge Rubric, 1–5 Scale)
- **Relevance:** 3.96 / 5.0
- **Groundedness:** 4.50 / 5.0
- **Helpfulness:** 4.80 / 5.0
- **Completeness:** 4.84 / 5.0
- **Hallucination / Unsupported Claims Rate:** **0.0%**
- **Average Overall Score:** **4.46 / 5.0**

### LLM Judge vs. Human Agreement (N=50 Calibration Set)
- **Exact Score Agreement:** **50.00%**
- **Agreement within $\pm 1$ Point:** **94.00%**
- **Mean Absolute Error (MAE):** **0.560**
- **Spearman Rank Correlation ($\rho$):** **0.1317**
- **Quadratic Weighted Cohen's Kappa ($\kappa$):** **0.1062**

---

## 5. Evaluation Methodology & Silver-Label Circularity

In `prepare_dataset.py`, conversations were partitioned at the conversation level into:
- `train` (19,545 conversations): Used exclusively for training models and populating the FAISS index.
- `classifier_eval` (3,448 conversations): Held-out split for measuring classification consistency.
- `golden_eval` (200 conversations): Sampled split for human evaluation and end-to-end agent triage auditing.

### Intent Label Verification:
- **Silver Labels:** The `intent` column in `classifier_eval` was generated by regex taxonomy rules. Evaluating rule-based models against rule-derived labels produces an artificially high 100% agreement score.
- **Why Simple ML Baseline Matters:** TF-IDF + Logistic Regression achieves 84.9% accuracy and 0.8418 Macro F1 independently of regex matching, proving that the 10 intent categories are linguistically separable and learnable.
- **Production Hybrid:** Checks regex rules first; if unmatched or falling into `other_unclear`, queries the trained ML model. This resolves ambiguous queries and achieves 95.5% consistency.

---

## 6. Golden Evaluation Set & Hand-Labeling Status

The assignment specifies a golden evaluation set of 150–250 hand-labelled examples.

### Current Golden Set Status:
- `data/golden_eval_template.csv` contains **200 sampled conversations**.
- Columns `conversation_id`, `customer_message`, `apple_response_actual`, `intent_auto`, and `resolution_status_auto` are populated.
- Columns `human_intent_label`, `human_reply_quality_1to5`, and `human_notes` are provided as blank annotation columns.
- `data/human_validation_subset.csv` contains **50 calibration examples** that were provisionally reviewed during initial harness development.
- **Transparent Status:** The 200 examples were sampled for the golden evaluation set; 50 were provisionally reviewed during initial calibration. The remaining examples require independent human annotation before this can be considered a fully hand-labelled golden set.

### Sampling Methodology:
1. **Unit of Sampling:** Complete root conversations from `apple_conversations.json` after nested-suffix deduplication.
2. **Partitioning:** A deterministic pseudo-random partition (`seed=42`) split the conversation IDs into train, classifier_eval, and golden_eval.
3. **Intent Distribution:** Stratification was not applied programmatically; rather, uniform random sampling across the deduplicated corpus naturally mirrored the empirical intent distribution (`other_unclear` ~31%, `ios_software_update` ~25%, `battery_power` ~10%, etc.).
4. **Leakage Prevention:** Partitioning strictly at the conversation level ensured that no tweet from `golden_eval` appears in `train` or the FAISS index.

### How to Hand-Label the Remaining Items:
An interactive labeling tool is included:
```bash
python label_golden_set.py
```
This CLI tool presents each tweet, allows 1-key intent selection (0–9) and quality rating (1–5), and saves directly to `data/golden_eval_template.csv`.

---

## 7. Top 5 Meaningful Failure Modes

1. **[FM-01] Ultra-Short / Image-Only Tweets:** Customers frequently post screenshots without text (e.g. *"@AppleSupport u too https://t.co/AubRbVtD4f"*). The text-only classifier cannot parse images and defaults to `other_unclear` $\to$ `ESCALATE`.
   - *Fix:* Add an OCR preprocessor to extract text from Twitter media URLs prior to classification.
2. **[FM-02] Vocabulary Clashes & Regex Priority:** `"Can't edit Can't share Happens to various photos iOS 11.1.2 6s"` was classified under `ios_software_update` because of `"iOS 11.1.2"`, missing the Photos app issue.
   - *Fix:* Replace rigid regex precedence with dense embedding classification where app symptoms weigh equally with OS tags.
3. **[FM-03] Historical Temporal Policy Drift:** `"fix y'all damn software. 'A ?'"` retrieved an obsolete 2017 text-replacement workaround for the iOS 11.1 "I" autocorrect bug.
   - *Fix:* Add timestamp metadata to retrieval items and prioritize permanent knowledge base links over temporary patches.
4. **[FM-04] Multi-Issue Inquiries:** `"Phone freezing, won't charge. Dad won't let me upgrade so chill please."` contains both battery and unresponsiveness symptoms. The system auto-handled the battery issue but missed the freezing symptom.
   - *Fix:* Multi-label intent classification with mandatory escalation when co-occurring issues appear.
5. **[FM-05] Sarcastic Gratitude:** `"Gee thanks @AppleSupport for fixing NONE of the problems"` contains the word `"thanks"`, which surface heuristics misinterpret as resolution.
   - *Fix:* Sentiment polarity analysis to detect frustration and prioritize senior human escalation.

---

## 8. "What is Misleading About My Headline Number?"

*(Mandatory Critical Reflection)*

1. **Circular Silver Labeling:** My hybrid classifier achieves 95.5% accuracy on `classifier_eval`. However, because ground truth labels in `classifier_eval` were generated via regex rules, this metric measures consistency with the taxonomy rules, not independent human accuracy.
2. **Retrieval Hit@K Measures Intent, Not Solution Quality:** A 91.6% Hit@5 means that an example with the same *topic* (e.g. `battery_power`) was retrieved, not necessarily that Apple's reply answered the customer's specific hardware/software issue.
3. **Retaining Unresolved Conversations:** Apple's historical reply was often just asking for the iOS version or requesting a DM. The agent occasionally mimics this clarifying step rather than providing an instant technical fix.
4. **Severe Score Compression in LLM Judge:** Exact agreement between the LLM Judge and human ratings is only **50.00%**. The **94.00% within $\pm 1$ point** metric is inflated because ratings heavily cluster in {4, 5}. Low Spearman rank correlation ($\rho = 0.1317$) and low Cohen's Kappa ($\kappa = 0.1062$) prove that the judge cannot reliably rank subtle quality differences. The 4.46/5 score is a coarse signal for tone, not proof of high-quality replies.
5. **Point-in-Time Dataset Bias (Late 2017):** The corpus captures late 2017 interactions (iOS 11 launch). Deploying to modern devices requires ingesting modern Apple Support articles.

---

## 9. Next-Week Plan (Evaluation-First Roadmap)

- **Day 1–2:** Complete independent human annotation across all 200 golden evaluation items using `label_golden_set.py` and expand human agreement analysis from N=50 to N=200.
- **Day 3:** Improve intent classification for compound/multi-issue messages using multi-label or dense embedding classification.
- **Day 4:** Add temporal awareness to retrieval so old troubleshooting workarounds are down-weighted in favor of evergreen knowledge base articles.
- **Day 5:** Add OCR/multimodal extraction for screenshot and image-only complaints.
- **Day 6–7:** Add lightweight persistent conversation state for multi-turn customer dialogues.

---

## 10. Decision Log Summary

A full record of 14 non-obvious engineering decisions is available in [`DECISION_LOG.md`](file:///c:/Users/PRIYA/Downloads/hiver_ass/DECISION_LOG.md):
- **Why Apple Support:** Cleanest, largest brand corpus (~140k tweets) with well-defined technical domains.
- **Why Conversation Reconstruction:** Apple does not mention `@AppleSupport` in its own replies; recursive parent-child graph expansion was required to recover 41,800 missing turns.
- **Why Nested-Suffix Deduplication:** Eliminating partial duplicate sub-threads prevented a 28% data leakage between train and eval sets.
- **Why FAISS on Train Only:** Completely isolated retrieval corpus (19,545 train items) to prevent evaluation data leakage.
- **Why Explainable Escalation:** Hard safety routing for security, billing, and hardware damage ensures brand safety.

---

## 11. Project Structure

```
├── README.md                      <- Project overview & reproduction guide
├── REPORT.md                      <- Formal 6-page submission report
├── DECISION_LOG.md                <- 14 detailed engineering decisions
├── LABELING_GUIDE.md              <- Annotation instructions for golden set
├── requirements.txt               <- Minimal Python dependencies
├── .gitignore                     <- Ignores raw 3M-row twcs.csv & caches
├── .env.example                   <- Template for optional Gemini API key
├── prepare_dataset.py             <- Deduplication, taxonomy, and splitting
├── run_evaluation.py              <- One-command evaluation harness (< 20s)
├── label_golden_set.py            <- Interactive CLI tool for golden set hand-labeling
├── app.py                         <- Interactive Streamlit demo
├── apple_conversations.json       <- 32,487 reconstructed conversations
├── data/
│   ├── support_pairs.csv          <- 23,193 flattened (customer, Apple) pairs
│   ├── conversations_clean.jsonl  <- Cleaned conversations with full turns
│   ├── retrieval_corpus.jsonl     <- 19,545 train conversations for FAISS
│   ├── golden_eval_template.csv   <- 200 sampled golden conversations
│   ├── human_validation_subset.csv<- 50 provisionally reviewed calibration rows
│   ├── faiss_index.bin            <- Precomputed FAISS IndexFlatIP binary
│   ├── retrieval_metadata.json    <- Metadata index for sub-second lookup
│   ├── hybrid_classifier.pkl      <- Trained production hybrid classifier
│   ├── taxonomy.json / .md        <- Intent taxonomy definitions & stats
│   └── eval_results.json          <- Output benchmark results from evaluation
└── src/
    ├── classifier.py              <- Trivial, Simple ML & Hybrid classifiers
    ├── retrieval.py               <- Sentence-Transformers + FAISS engine
    ├── escalation.py              <- Multi-factor explainable escalation logic
    ├── generator.py               <- Grounded reply generator (API + Offline)
    ├── judge.py                   <- LLM-as-a-judge & agreement statistics
    └── pipeline.py                <- Unified SupportAgent pipeline
```

---

## Screenshots - 
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/1ba5111f-ec92-4075-ade1-d90de10b4728" />



<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/ec1037f7-c696-430c-9858-ded53ca8b888" />



<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/2430528f-4a6f-4869-afc1-6afcabe6ff39" />

## 12. Key Engineering Q&A

### Architecture Overview
> *"I designed this as a structured 4-stage pipeline rather than an unconstrained chatbot. When an incoming customer tweet arrives, it is first classified into a 10-intent taxonomy using a high-precision hybrid classifier (regex rules backed by TF-IDF and Logistic Regression). Next, the agent queries a FAISS index containing 19,545 historical Apple Support interactions embedded with `all-MiniLM-L6-v2` to retrieve historically proven troubleshooting responses. Then, an explainable escalation engine audits the query: high-risk domains like Apple ID security, billing fraud, or shattered screens are strictly escalated to human agents, while routine issues with strong retrieval evidence are auto-handled. Finally, a grounded generator drafts a concise, polite Apple-style tweet strictly reflecting the retrieved historical evidence without hallucinating fake policies."*

### Key Technical Questions
1. **Why FAISS on Train Split Only?**  
   *Answer:* To avoid catastrophic data leakage. If test conversations are in the retrieval index, the retriever retrieves the exact duplicate, faking 100% retrieval performance.
2. **Why Nested-Suffix Deduplication?**  
   *Answer:* Multi-part customer threads caused the reconstruction algorithm to create redundant sub-threads (`[A, B, Apple]` and `[B, Apple]`). Deduplicating by keeping only root conversations removed 9,235 redundant rows and prevented train/eval contamination.
3. **Why Not Just Use an LLM for Everything?**  
   *Answer:* Unconstrained LLMs hallucinate policies, have unpredictable inference costs, and cannot provide deterministic guarantees on safety-critical routing (e.g. password resets or credit card refunds). Hard-coded explainable escalation ensures regulatory compliance and brand safety.

---

## 13. Acknowledgments / Sources

- **Dataset:** Kaggle "Customer Support on Twitter" dataset by thoughtvector (Creative Commons).
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` by HuggingFace / UKPLab.
- **Vector Search:** FAISS (`faiss-cpu`) by Meta Research.
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF vectorizer and Logistic Regression).
- **Web UI:** Streamlit.
- **AI Coding Assistance:** AI coding assistants were used to assist with syntax, documentation formatting, and rapid iteration during development; all architectural decisions, taxonomy designs, data filtering rules, and evaluation audits were designed and verified by the author.
