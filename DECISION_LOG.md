# Engineering Decision Log

**Author:** Priya Pandey  
**Project:** AI Customer Support Agent (Hiver Assignment)  
**Corpus:** Kaggle "Customer Support on Twitter" (`twcs.csv` / Apple Support)

---

### Decision 1: Targeting Apple Support Instead of Multi-Brand
- **Decision:** I chose to focus exclusively on `@AppleSupport` rather than building a generic multi-brand support agent across all 3 million tweets.
- **Alternatives Considered:**
  1. Training a generic multi-brand agent across Amazon, Uber, Delta, and Apple.
  2. Telecom customer care (Sprint, T-Mobile).
- **Reason:** Exploring the dataset showed Apple Support is the cleanest, largest brand corpus (~140K related tweets) with well-defined technical sub-domains (iOS versions, hardware, battery, App Store, iCloud). Groundedness evaluation is far more rigorous in a specific domain where factual accuracy can be verified.
- **Tradeoff:** The agent cannot answer questions about other companies without loading a different retrieval corpus.

---

### Decision 2: Graph-Based Recursive Conversation Reconstruction
- **Decision:** I used iterative parent-child graph expansion across `in_response_to_tweet_id` and `response_tweet_id` over the full 3M-row `twcs.csv` to reconstruct complete multi-turn threads.
- **Alternatives Considered:** Simply filtering for tweets with the text `@AppleSupport`.
- **Reason:** A simple text filter for `@AppleSupport` returned 97,913 tweets but missed ~41,800 Apple responses because Apple's support agents do not mention their own handle in replies. The recursive expansion recovered 139,770 tweets across 32,487 genuine conversation threads.
- **Tradeoff:** Graph traversal took substantial initial runtime and occasionally pulled in tweets from other brands mentioned in the same thread.

---

### Decision 3: Nested-Suffix Conversation Deduplication
- **Decision:** I implemented an algorithm to prune "nested suffix" conversations, retaining only root conversations whose first tweet is not a downstream reply in another conversation.
- **Alternatives Considered:** Keeping every conversation reconstructed from each inbound tweet.
- **Reason:** When a customer wrote a multi-part tweet thread (Tweet A $\to$ self-reply Tweet B $\to$ Apple reply), the naive reconstruction produced both `[A, B, Apple]` and `[B, Apple]`. This resulted in a ~28% duplicate rate and severe train/test data leakage. Pruning nested suffixes removed 9,235 redundant conversations.
- **Tradeoff:** Reduced total conversation count from 32,487 to 23,252, but guaranteed clean data hygiene.

---

### Decision 4: Non-Apple Support Handle Filtering
- **Decision:** I removed threads where the support agent handle was not `AppleSupport` (e.g. `SpotifyCares`, `AsurionCares`).
- **Alternatives Considered:** Keeping all recovered threads.
- **Reason:** Because customers sometimes tagged multiple companies in one tweet (e.g. asking both Spotify and Apple why an offline song won't play), non-Apple replies got pulled into the thread chain. Dropping them removed 59 noisy conversations.
- **Tradeoff:** Discarded a few multi-brand edge cases, but kept the support corpus strictly authentic to Apple.

---

### Decision 5: Concatenating Multi-Tweet Customer Inquiries
- **Decision:** I collapsed all consecutive inbound customer tweets preceding Apple's first response into a single `customer_message`.
- **Alternatives Considered:**
  1. Using only the customer's first tweet.
  2. Using only the last tweet before Apple's response.
- **Reason:** Twitter's 280-character limit forces customers to split their thoughts across 2 or 3 tweets. Using only the first tweet often missed crucial context (e.g. Tweet 1: *"My phone broke"*, Tweet 2: *"After updating to iOS 11.0.3"*). Concatenation provides the complete diagnostic picture.
- **Tradeoff:** Inquiry text lengths vary more widely, requiring text normalization.

---

### Decision 6: 10-Class Intent Taxonomy Derived from Real Data
- **Decision:** I designed a 10-category taxonomy (9 domain-specific intents + 1 `other_unclear` fallback) based on reading actual Apple customer complaints.
- **Alternatives Considered:**
  1. A sprawling 40-50 class taxonomy.
  2. A tiny 3-class taxonomy (Billing, Technical, General).
- **Reason:** A 10-class taxonomy is small enough to explain and defend in an interview, aligns with how Apple routes service tickets (iCloud security, hardware repair, battery, connectivity, OS update), and provides sufficient granularity for automated triage.
- **Tradeoff:** Merges related sub-issues (e.g. Wi-Fi and Bluetooth under `connectivity`).

---

### Decision 7: Keeping Unresolved Conversations in the Retrieval Corpus
- **Decision:** I retained conversations categorized as `unresolved_no_signal` in the corpus rather than filtering only for explicit customer thank-yous.
- **Alternatives Considered:** Filtering exclusively for threads where the customer said "thank you" or "fixed".
- **Reason:** Only 17.9% (4,164 / 23,193) of Apple Twitter threads end with explicit gratitude. On Twitter, customers routinely abandon threads once solved or continue troubleshooting in DMs or phone calls. Discarding unresolved threads would throw away over 80% of useful troubleshooting interactions.
- **Tradeoff:** The retrieval corpus includes diagnostic questions that did not lead to a recorded public resolution.

---

### Decision 8: Isolating the Retrieval Corpus Strictly to the Train Split
- **Decision:** I indexed only the `train` split (N=19,545) in the FAISS retrieval index, completely excluding `classifier_eval` (3,448) and `golden_eval` (200).
- **Alternatives Considered:** Building a single retrieval index over all 23,193 conversations.
- **Reason:** Indexing the entire dataset causes severe data leakage—the retriever would simply retrieve the exact identical conversation during evaluation, artificially inflating Hit@K to near 100%.
- **Tradeoff:** The retrieval corpus has ~3,600 fewer examples, but evaluation numbers reflect true generalization to unseen customer issues.

---

### Decision 9: FAISS IndexFlatIP with L2-Normalized Embeddings
- **Decision:** I used FAISS `IndexFlatIP` (inner product on unit-normalized vectors) with `all-MiniLM-L6-v2`.
- **Alternatives Considered:**
  1. Approximate nearest neighbor (e.g. `IndexIVFFlat` or `HNSW`).
  2. TF-IDF cosine similarity.
- **Reason:** For 19,545 vectors of dimension 384, exact exhaustive search runs in ~1.2 milliseconds per query on CPU. Approximate indexing adds recall loss with no perceptible speed benefit at this scale.
- **Tradeoff:** Linear memory scaling with corpus size, though 19,545 vectors take only ~30 MB of RAM.

---

### Decision 10: Hybrid Intent Classifier (Regex Rules + ML Fallback)
- **Decision:** I combined domain regex taxonomy rules with a TF-IDF + Logistic Regression fallback.
- **Alternatives Considered:**
  1. Pure TF-IDF + Logistic Regression.
  2. Fine-tuning a heavy BERT sequence classification model.
- **Reason:** High-precision domain keywords (e.g. `autocorrect`, `icloud`, `refund`) are unambiguous and achieve near 100% precision with zero latency. The ML fallback handles vocabulary variations. It trains in ~3 seconds and runs efficiently on CPU.
- **Tradeoff:** Requires maintaining regex patterns alongside the ML model.

---

### Decision 11: Multi-Factor Explainable Escalation Logic
- **Decision:** I implemented deterministic multi-factor escalation based on intent risk domain, query ambiguity, and retrieval similarity score, returning an explicit reason string.
- **Alternatives Considered:**
  1. Prompting an LLM to decide escalation without constraints.
  2. Static similarity thresholding alone.
- **Reason:** Customer support automation in consumer tech must be auditable. High-risk intents (account security, unauthorized billing, shattered screens) must NEVER be auto-handled regardless of retrieval score.
- **Tradeoff:** Conservative hard-coded risk policies escalate ~43% of inquiries, prioritizing brand safety over maximum deflection rate.

---

### Decision 12: Dual-Mode Reply Generator (Gemini REST API + Offline Grounded Fallback)
- **Decision:** I built reply generation using direct REST requests to Google Gemini with an automatic fallback to an offline retrieval-grounded generator.
- **Alternatives Considered:** Requiring the `google-generativeai` SDK.
- **Reason:** Python 3.13 environments often face dependency conflicts with Google SDKs. Using standard `requests` avoids installation breakages. The offline grounded fallback guarantees the assignment evaluates reproducibly in < 15 minutes without requiring API keys.
- **Tradeoff:** The offline generator produces slightly more templated responses than dynamic LLM generation.

---

### Decision 13: Ground-Truth Human Calibration Subset (N=50)
- **Decision:** I curated a verified N=50 human-labeled calibration set with independently rated reply quality scores (1-5) and computed formal agreement statistics (Spearman $\rho$, Cohen's $\kappa$).
- **Alternatives Considered:**
  1. Fabricating synthetic "human" labels using an LLM.
  2. Omitting the human agreement evaluation requirement.
- **Reason:** The assignment explicitly forbids fabricating human agreement. Providing an audited calibration sample allows real mathematical measurement of inter-rater reliability (94% agreement within $\pm 1$ point).
- **Tradeoff:** Sample size (N=50) is smaller than the full 200-row golden set, representing a calibration sample.

---

### Decision 14: Sub-Minute Reproducibility Target (< 5 Minutes)
- **Decision:** I pre-computed and cached the FAISS index (`data/faiss_index.bin`) and classifier (`data/hybrid_classifier.pkl`) so `run_evaluation.py` executes in under 20 seconds.
- **Alternatives Considered:** Re-embedding 19,545 queries from scratch on every evaluation run.
- **Reason:** Evaluators grading assignments test dozens of repositories; requiring a 45-minute re-indexing loop creates friction. Caching provides instantaneous reproduction while allowing a one-command rebuild via `prepare_dataset.py`.
- **Tradeoff:** Generates ~30MB binary index file on disk.
