"""
src/pipeline.py
===============

End-to-End AI Customer Support Agent Pipeline for Apple Support.

Ties together:
1. Intent Classification (Hybrid Regex + TF-IDF/LogisticRegression)
2. Historical Retrieval (sentence-transformers + FAISS index)
3. Explainable Escalation (Intent Risk + Evidence Strength + Clarification)
4. Grounded Reply Generation (Gemini API or Offline Grounded Fallback)

Simple, clean, and directly defensible in an interview:
    agent = SupportAgent()
    response = agent.handle("My battery drains in 2 hours since iOS 11")
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from src.classifier import HybridIntentClassifier
from src.retrieval import HistoricalRetriever
from src.escalation import EscalationEngine
from src.generator import ReplyGenerator

MODEL_PATH = Path("data/hybrid_classifier.pkl")


class SupportAgent:
    def __init__(self, classifier=None, retriever=None, escalation=None, generator=None):
        # 1. Classifier
        if classifier:
            self.classifier = classifier
        elif MODEL_PATH.exists():
            self.classifier = HybridIntentClassifier.load(str(MODEL_PATH))
        else:
            self.classifier = HybridIntentClassifier()
            # Train if support_pairs exists
            import pandas as pd
            df_path = Path("data/support_pairs.csv")
            if df_path.exists():
                df = pd.read_csv(df_path)
                train_df = df[df["split"] == "train"]
                self.classifier.fit(train_df["customer_message"], train_df["intent"])
                self.classifier.save(str(MODEL_PATH))

        # 2. Retriever
        self.retriever = retriever or HistoricalRetriever()
        if not self.retriever.index:
            self.retriever.load()

        # 3. Escalation Engine
        self.escalation = escalation or EscalationEngine()

        # 4. Generator
        self.generator = generator or ReplyGenerator()

    def handle(self, customer_message, top_k=3):
        """
        Processes a single customer message through the 4-stage pipeline.
        Returns a dictionary with intent, confidence, retrieved evidence,
        escalation decision & reason, and drafted reply.
        """
        msg = str(customer_message).strip()

        # Stage 1: Classify Intent
        intent, confidence = self.classifier.predict_one(msg)

        # Stage 2: Historical Retrieval
        retrieved_examples = self.retriever.retrieve(msg, top_k=top_k)

        # Stage 3: Auto-Handle vs Escalation Decision
        esc_result = self.escalation.evaluate(
            customer_message=msg,
            predicted_intent=intent,
            intent_confidence=confidence,
            retrieved_examples=retrieved_examples
        )

        # Stage 4: Grounded Reply Generation
        reply = self.generator.generate_reply(
            customer_message=msg,
            predicted_intent=intent,
            retrieved_examples=retrieved_examples
        )

        return {
            "customer_message": msg,
            "intent": intent,
            "confidence": round(float(confidence), 2),
            "decision": esc_result["decision"],
            "reason": esc_result["reason"],
            "risk_level": esc_result["risk_level"],
            "drafted_reply": reply,
            "retrieved_examples": retrieved_examples
        }


if __name__ == "__main__":
    agent = SupportAgent()
    queries = [
        "My battery is draining really fast since I updated to iOS 11 on my iPhone 7",
        "I'm locked out of my Apple ID and cannot get the verification code",
        "Screen cracked after dropping my phone, need repair appointment",
        "Why is my wifi disconnecting randomly?"
    ]
    
    print("\n" + "="*80)
    print("SUPPORT AGENT PIPELINE DEMO")
    print("="*80 + "\n")
    
    for q in queries:
        res = agent.handle(q)
        print(f"Customer:    {res['customer_message']}")
        print(f"Intent:      {res['intent']} (conf: {res['confidence']})")
        print(f"Decision:    [{res['decision']}] - {res['reason']}")
        print(f"Apple Reply: {res['drafted_reply']}")
        top_sim = res['retrieved_examples'][0]['similarity_score'] if res['retrieved_examples'] else 0.0
        print(f"Evidence:    Top match sim = {top_sim:.3f}")
        print("-" * 80)
