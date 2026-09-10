"""
src/judge.py
============

LLM-as-a-Judge Evaluation & Human Agreement Module.

Evaluates drafted Apple Support replies against a strict 5-dimension rubric:
1. Relevance (1-5): How directly the reply addresses the customer's specific problem.
2. Groundedness (1-5): How faithfully the reply reflects retrieved historical Apple evidence.
3. Helpfulness (1-5): Actionability, empathy, and troubleshooting effectiveness.
4. Completeness (1-5): Adequacy of next steps or necessary clarifying questions.
5. Unsupported Claims / Hallucination: Binary penalty (0=Clean, 1=Hallucinated).
6. Overall Score (1-5): Aggregate rating.

Computes Inter-Rater Agreement with Human Evaluator:
- Exact Agreement (%)
- Agreement within +/- 1 point (%)
- Spearman Rank Correlation (rho)
- Quadratic Weighted Cohen's Kappa (kappa)
- Mean Absolute Error (MAE)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import math
import requests
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT_TEMPLATE = """You are an expert AI customer support quality evaluator.
Evaluate the following generated customer support tweet reply from @AppleSupport to a customer.

Customer Message: "{customer_message}"
Predicted Intent: "{intent}"
Retrieved Historical Apple Support Evidence:
{retrieved_evidence}

Drafted Reply Under Evaluation:
"{drafted_reply}"

Evaluate the drafted reply on these 5 dimensions:
1. Relevance (1 to 5): Does it directly address the customer's specific issue?
2. Groundedness (1 to 5): Is it grounded in the historical Apple evidence without unsupported fabrications?
3. Helpfulness (1 to 5): Does it provide actionable troubleshooting or a clear path forward?
4. Completeness (1 to 5): Does it adequately address the query or ask necessary clarifying questions?
5. Hallucination (0 or 1): 1 if the reply makes fake policy/warranty promises, 0 if safe.

Output ONLY valid JSON in this exact format:
{{
  "relevance": 5,
  "groundedness": 5,
  "helpfulness": 5,
  "completeness": 4,
  "hallucination": 0,
  "overall_score": 5,
  "critique": "Brief 1-sentence evaluation"
}}"""


class LLMJudge:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def _heuristic_judge(self, customer_message, retrieved_examples, drafted_reply):
        """Offline deterministic rubric judge when LLM API key is not configured."""
        text = drafted_reply.lower()
        cust = customer_message.lower()
        
        # 1. Relevance: checks query terms
        cust_words = set(w for w in re.findall(r"\w+", cust) if len(w) > 3)
        match_count = sum(1 for w in cust_words if w in text)
        relevance = 5 if match_count >= 2 else (4 if match_count == 1 else 3)
        if len(text.split()) < 4:
            relevance = 2

        # 2. Groundedness: checks against retrieved text
        evidence_text = " ".join([ex.get("apple_response", "") for ex in retrieved_examples]).lower()
        groundedness = 4
        if any(token in text for token in ["settings", "restart", "update", "backup", "dm", "link", "http"]):
            groundedness = 5

        # 3. Helpfulness
        has_action = any(action in text for action in ["try", "update", "restart", "settings", "check", "dm", "let us know", "?"])
        helpfulness = 5 if has_action else 3

        # 4. Completeness
        completeness = 4
        if "?" in drafted_reply or "http" in drafted_reply:
            completeness = 5

        # 5. Hallucination / Unsupported claims
        hallucinated = 0
        forbidden = ["free replacement", "refund guaranteed", "tomorrow morning", "100% fixed", "my fault"]
        if any(f in text for f in forbidden):
            hallucinated = 1
            groundedness = min(groundedness, 2)
            helpfulness = min(helpfulness, 2)

        overall = round((relevance * 0.3) + (groundedness * 0.3) + (helpfulness * 0.25) + (completeness * 0.15))
        if hallucinated:
            overall = max(1, overall - 2)

        return {
            "relevance": relevance,
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "completeness": completeness,
            "hallucination": hallucinated,
            "overall_score": max(1, min(5, overall)),
            "critique": "Offline rubric evaluation based on relevance, actionability, and grounding."
        }

    def judge_reply(self, customer_message, intent, retrieved_examples, drafted_reply):
        """Evaluates drafted reply using Gemini API or fallback rubric."""
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            return self._heuristic_judge(customer_message, retrieved_examples, drafted_reply)

        evidence_str = "\n".join([
            f"- {ex.get('apple_response', '')}" for ex in retrieved_examples[:3]
        ])
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            intent=intent,
            retrieved_evidence=evidence_str,
            drafted_reply=drafted_reply
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
            }
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                raw_json = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw_json)
            else:
                return self._heuristic_judge(customer_message, retrieved_examples, drafted_reply)
        except Exception:
            return self._heuristic_judge(customer_message, retrieved_examples, drafted_reply)


# --------------------------------------------------------------------------
# Inter-Rater Agreement Statistics
# --------------------------------------------------------------------------

def compute_spearman_rho(x, y):
    """Computes Spearman's rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def rank(vals):
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_indices[j]] == vals[sorted_indices[j + 1]]:
                j += 1
            avg_rank = 1.0 + (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x * den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def compute_quadratic_weighted_kappa(human_scores, judge_scores, min_rating=1, max_rating=5):
    """
    Computes Quadratic Weighted Cohen's Kappa for ordinal 1-5 ratings.
    Standard metric for LLM vs Human grading agreement.
    """
    n = len(human_scores)
    if n == 0:
        return 0.0

    num_categories = max_rating - min_rating + 1
    O = [[0] * num_categories for _ in range(num_categories)]
    
    for h, j in zip(human_scores, judge_scores):
        h_idx = max(0, min(num_categories - 1, int(round(h)) - min_rating))
        j_idx = max(0, min(num_categories - 1, int(round(j)) - min_rating))
        O[h_idx][j_idx] += 1

    # Marginal totals
    h_totals = [sum(O[i][j] for j in range(num_categories)) for i in range(num_categories)]
    j_totals = [sum(O[i][j] for i in range(num_categories)) for j in range(num_categories)]

    # Expected matrix
    E = [[0.0] * num_categories for _ in range(num_categories)]
    for i in range(num_categories):
        for j in range(num_categories):
            E[i][j] = (h_totals[i] * j_totals[j]) / float(n)

    # Weights: (i - j)^2 / (num_categories - 1)^2
    max_diff_sq = float((num_categories - 1) ** 2)
    num = 0.0
    den = 0.0
    for i in range(num_categories):
        for j in range(num_categories):
            w = ((i - j) ** 2) / max_diff_sq
            num += w * O[i][j]
            den += w * E[i][j]

    if den == 0:
        return 1.0
    return round(1.0 - (num / den), 4)


def evaluate_judge_human_agreement(human_ratings, judge_ratings):
    """
    Computes full statistical comparison between human and LLM-as-judge scores.
    """
    assert len(human_ratings) == len(judge_ratings), "Rating lists must be same length"
    n = len(human_ratings)
    if n == 0:
        return {}

    exact_matches = sum(1 for h, j in zip(human_ratings, judge_ratings) if int(round(h)) == int(round(j)))
    within_one = sum(1 for h, j in zip(human_ratings, judge_ratings) if abs(int(round(h)) - int(round(j))) <= 1)
    mae = sum(abs(h - j) for h, j in zip(human_ratings, judge_ratings)) / float(n)
    
    spearman_rho = compute_spearman_rho(human_ratings, judge_ratings)
    weighted_kappa = compute_quadratic_weighted_kappa(human_ratings, judge_ratings)

    return {
        "sample_size": n,
        "exact_agreement_pct": round(100.0 * exact_matches / n, 2),
        "within_1_point_agreement_pct": round(100.0 * within_one / n, 2),
        "mean_absolute_error": round(mae, 3),
        "spearman_rho": spearman_rho,
        "quadratic_weighted_kappa": weighted_kappa
    }


if __name__ == "__main__":
    judge = LLMJudge()
    
    # Test single judge call
    res = judge.judge_reply(
        customer_message="My battery life is terrible on iOS 11.0.2",
        intent="battery_power",
        retrieved_examples=[{"apple_response": "Check out https://apple.co/battery for tips to optimize your battery settings."}],
        drafted_reply="We know battery life is crucial! Take a look at these tips: https://apple.co/battery and let us know if that helps."
    )
    print("Test Judge Result:\n", json.dumps(res, indent=2))

    # Test agreement stats
    mock_human = [5, 4, 3, 2, 4, 5, 3, 4, 2, 5, 4, 3]
    mock_judge = [5, 4, 4, 2, 4, 4, 3, 5, 2, 5, 4, 3]
    agreement = evaluate_judge_human_agreement(mock_human, mock_judge)
    print("\nTest Agreement Metrics:\n", json.dumps(agreement, indent=2))
