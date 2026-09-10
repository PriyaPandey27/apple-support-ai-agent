"""
src/retrieval.py
================

Historical Retrieval System using Sentence-Transformers and FAISS.

Indexes only the TRAIN split from `data/retrieval_corpus.jsonl` to ensure
zero data leakage. Computes L2-normalized embeddings with 'all-MiniLM-L6-v2'
and searches using FAISS IndexFlatIP (cosine similarity).

Features:
- Fast pre-indexing and caching to disk (`data/faiss_index.bin`, `data/retrieval_metadata.json`).
- Sub-second retrieval (<20ms per query).
- Hit@K and Mean Similarity evaluation.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = Path("data/faiss_index.bin")
METADATA_FILE = Path("data/retrieval_metadata.json")
CORPUS_FILE = Path("data/retrieval_corpus.jsonl")


class HistoricalRetriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, corpus_path=CORPUS_FILE, batch_size=256):
        """Builds FAISS index from retrieval corpus (TRAIN split only)."""
        self._load_model()
        print(f"Loading corpus from {corpus_path}...")
        
        records = []
        texts = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                records.append(item)
                texts.append(item["customer_message"])
                
        print(f"Embedding {len(texts)} historical customer queries with {self.model_name}...")
        t0 = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float32")
        print(f"Embeddings computed in {time.time()-t0:.2f}s. Shape: {embeddings.shape}")
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = records
        
        # Save to disk
        self.save()
        return self

    def save(self, index_path=INDEX_FILE, metadata_path=METADATA_FILE):
        index_path.parent.mkdir(exist_ok=True, parents=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"Saved index ({self.index.ntotal} vectors) to {index_path}")

    def load(self, index_path=INDEX_FILE, metadata_path=METADATA_FILE):
        self._load_model()
        if not index_path.exists() or not metadata_path.exists():
            print("Index or metadata file not found. Building fresh index...")
            return self.build_index()
            
        print(f"Loading cached FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Loaded {self.index.ntotal} historical records.")
        return self

    def retrieve(self, query_text, top_k=3):
        """
        Retrieves top_k historically similar Apple support interactions.
        Returns list of dicts with:
          - conversation_id
          - customer_message
          - apple_response
          - intent
          - similarity_score
        """
        self._load_model()
        if self.index is None:
            self.load()
            
        q_emb = self.model.encode([query_text], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["similarity_score"] = round(float(dist), 4)
            results.append(item)
            
        return results

    def evaluate_retrieval(self, query_list, ground_truth_intents, k_values=[1, 3, 5]):
        """
        Evaluates retrieval quality:
        - Hit@K: Percentage of queries where at least one retrieved example shares the true intent.
        - Mean Top-1 Similarity: Average cosine similarity of the nearest historical match.
        """
        self._load_model()
        if self.index is None:
            self.load()
            
        hits = {k: 0 for k in k_values}
        top1_sims = []
        max_k = max(k_values)
        
        q_embs = self.model.encode(
            query_list,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        distances, indices = self.index.search(q_embs, max_k)
        
        total = len(query_list)
        for i in range(total):
            true_intent = ground_truth_intents[i]
            retrieved_intents = [self.metadata[idx]["intent"] for idx in indices[i] if idx >= 0]
            top1_sims.append(float(distances[i][0]))
            
            for k in k_values:
                k_intents = retrieved_intents[:k]
                if true_intent in k_intents:
                    hits[k] += 1
                    
        return {
            f"Hit@{k}": round(hits[k] / total, 4) for k in k_values
        } | {
            "MeanTop1Similarity": round(float(np.mean(top1_sims)), 4),
            "TotalEvaluated": total
        }


if __name__ == "__main__":
    retriever = HistoricalRetriever()
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        retriever.load()
    else:
        retriever.build_index()
        
    test_query = "My battery is draining so fast since I updated to iOS 11"
    matches = retriever.retrieve(test_query, top_k=2)
    print(f"\nQuery: {test_query}\n")
    for i, m in enumerate(matches, 1):
        print(f"Match {i} (Similarity: {m['similarity_score']}):")
        print(f"  Customer: {m['customer_message']}")
        print(f"  Apple:    {m['apple_response']}")
        print(f"  Intent:   {m['intent']}")
        print()
