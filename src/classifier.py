"""
src/classifier.py
=================

Intent Classification module for the Apple Support Agent.

Implements three approaches:
1. Trivial Baseline: Always predicts the majority class in training data.
2. Simple ML Baseline: TF-IDF (unigram + bigram) + Logistic Regression.
3. Production Hybrid Classifier: Rule-based regex matcher with ML fallback for
   unmatched / ambiguous queries.

Evaluates:
- Accuracy
- Macro F1 / Weighted F1
- Per-intent Precision, Recall, and F1
- Confusion Matrix
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import pickle
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

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

ALL_INTENTS = [
    "autocorrect_text_bug",
    "account_security_icloud",
    "billing_repair_order",
    "battery_power",
    "screen_hardware_damage",
    "connectivity",
    "ios_software_update",
    "app_appstore_media",
    "device_not_working",
    "other_unclear"
]


class TrivialBaselineClassifier:
    """Always predicts the most frequent class seen during fit()."""
    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.majority_class] * len(X)

    def predict_one(self, text):
        return self.majority_class, 1.0


class SimpleMLClassifier:
    """TF-IDF (1-2 ngrams) + Logistic Regression with balanced weights."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=12000,
            sublinear_tf=True,
            strip_accents="unicode"
        )
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.classes_ = None

    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec)

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict_proba(X_vec)

    def predict_one(self, text):
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        best_idx = np.argmax(probs)
        return self.classes_[best_idx], float(probs[best_idx])


class RuleBasedClassifier:
    """Regex pattern matching based on real Apple support interactions."""
    def __init__(self):
        self.rules = INTENT_RULES
        self.fallback = FALLBACK_INTENT

    def fit(self, X, y):
        return self  # Rule-based requires no training

    def classify_text(self, text):
        t = str(text).lower()
        for name, pat in self.rules:
            if re.search(pat, t):
                return name, 0.95
        return self.fallback, 0.50

    def predict(self, X):
        return [self.classify_text(x)[0] for x in X]

    def predict_one(self, text):
        return self.classify_text(text)


class HybridIntentClassifier:
    """
    Production Intent Classifier.
    Checks regex taxonomy rules first. If a specific intent matches, returns it
    with high confidence. If the rule yields 'other_unclear', queries the trained
    TF-IDF + LogisticRegression model. If ML model has confidence > 0.40, uses ML,
    otherwise retains 'other_unclear'.
    """
    def __init__(self):
        self.rule_clf = RuleBasedClassifier()
        self.ml_clf = SimpleMLClassifier()

    def fit(self, X, y):
        self.ml_clf.fit(X, y)
        return self

    def predict_one(self, text):
        rule_pred, rule_conf = self.rule_clf.predict_one(text)
        if rule_pred != FALLBACK_INTENT:
            return rule_pred, rule_conf
        
        # Fall back to ML classifier
        ml_pred, ml_conf = self.ml_clf.predict_one(text)
        if ml_conf >= 0.35 and ml_pred != FALLBACK_INTENT:
            return ml_pred, round(ml_conf * 0.9, 2)
        return FALLBACK_INTENT, round(ml_conf, 2)

    def predict(self, X):
        preds = []
        for x in X:
            p, _ = self.predict_one(x)
            preds.append(p)
        return preds

    def save(self, filepath):
        state = {
            "vectorizer": self.ml_clf.vectorizer,
            "clf": self.ml_clf.clf,
            "classes_": self.ml_clf.classes_
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        instance = cls()
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
            
        if isinstance(obj, dict) and "clf" in obj:
            instance.ml_clf.vectorizer = obj["vectorizer"]
            instance.ml_clf.clf = obj["clf"]
            instance.ml_clf.classes_ = obj["classes_"]
            return instance
        elif hasattr(obj, "ml_clf"):
            return obj
        return instance


def evaluate_classifier(clf, X_test, y_test, labels=ALL_INTENTS):
    """Computes comprehensive classification metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    
    per_intent = {}
    for idx, label in enumerate(labels):
        per_intent[label] = {
            "precision": round(float(p[idx]), 3),
            "recall": round(float(r[idx]), 3),
            "f1": round(float(f1[idx]), 3),
            "support": int(support[idx])
        }
        
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": per_intent,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }


def format_confusion_matrix(cm, labels, max_len=14):
    """Formats confusion matrix as an aligned text table."""
    short_labels = [l[:max_len] for l in labels]
    header = f"{'True \\ Pred':<{max_len+2}} " + " ".join(f"{l[:5]:>5}" for l in short_labels)
    lines = [header, "-" * len(header)]
    for idx, row in enumerate(cm):
        row_str = f"{short_labels[idx]:<{max_len+2}} " + " ".join(f"{val:>5}" for val in row)
        lines.append(row_str)
    return "\n".join(lines)


if __name__ == "__main__":
    data_path = Path("data/support_pairs.csv")
    if not data_path.exists():
        print("[ERROR] data/support_pairs.csv not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == "classifier_eval"]
    
    print(f"Training on {len(train_df)} samples, evaluating on {len(eval_df)} samples...")
    
    # 1. Trivial Baseline
    trivial = TrivialBaselineClassifier().fit(train_df["customer_message"], train_df["intent"])
    trivial_metrics = evaluate_classifier(trivial, eval_df["customer_message"], eval_df["intent"])
    print("\n--- 1. TRIVIAL BASELINE (Majority Class) ---")
    print(f"Accuracy: {trivial_metrics['accuracy']:.4f} | Macro F1: {trivial_metrics['macro_f1']:.4f}")
    
    # 2. Simple ML Baseline
    ml_baseline = SimpleMLClassifier().fit(train_df["customer_message"], train_df["intent"])
    ml_metrics = evaluate_classifier(ml_baseline, eval_df["customer_message"], eval_df["intent"])
    print("\n--- 2. SIMPLE ML BASELINE (TF-IDF + Logistic Regression) ---")
    print(f"Accuracy: {ml_metrics['accuracy']:.4f} | Macro F1: {ml_metrics['macro_f1']:.4f}")
    
    # 3. Rule-Based Classifier
    rule_clf = RuleBasedClassifier()
    rule_metrics = evaluate_classifier(rule_clf, eval_df["customer_message"], eval_df["intent"])
    print("\n--- 3. RULE-BASED TAXONOMY ---")
    print(f"Accuracy: {rule_metrics['accuracy']:.4f} | Macro F1: {rule_metrics['macro_f1']:.4f}")
    
    # 4. Production Hybrid Classifier
    hybrid_clf = HybridIntentClassifier().fit(train_df["customer_message"], train_df["intent"])
    hybrid_metrics = evaluate_classifier(hybrid_clf, eval_df["customer_message"], eval_df["intent"])
    print("\n--- 4. PRODUCTION HYBRID CLASSIFIER (Rules + ML Fallback) ---")
    print(f"Accuracy: {hybrid_metrics['accuracy']:.4f} | Macro F1: {hybrid_metrics['macro_f1']:.4f}")
    
    # Save hybrid classifier
    hybrid_clf.save("data/hybrid_classifier.pkl")
    print("\nSaved trained model to data/hybrid_classifier.pkl")
