"""
src/escalation.py
=================

Explainable Auto-Handle vs. Escalation decision engine for Apple Support.

Decides whether an incoming customer inquiry can be answered automatically
or must be escalated to a human support specialist, providing an explicit,
auditable reason.

Evaluation dimensions:
1. Sensitive risk domain (Account security, Billing/Refunds, Physical hardware).
2. Query clarity and ambiguity ('other_unclear' / ultra-short tweets).
3. Retrieval evidence strength (FAISS cosine similarity vs threshold).
4. Intent classification confidence.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SENSITIVE_INTENTS = {
    "account_security_icloud": (
        "Sensitive security request: Apple ID / iCloud account issues require "
        "human verification or official account recovery to protect user privacy."
    ),
    "billing_repair_order": (
        "Financial/repair request: Billing disputes, refund requests, and repair orders "
        "require human agent access to internal customer billing systems."
    ),
    "screen_hardware_damage": (
        "Hardware damage: Cracked screens, physical buttons, or liquid damage require "
        "in-person evaluation at an Apple Store Genius Bar."
    )
}

ROUTINE_SUPPORT_INTENTS = {
    "battery_power",
    "connectivity",
    "ios_software_update",
    "app_appstore_media",
    "autocorrect_text_bug",
    "device_not_working"
}


class EscalationEngine:
    def __init__(self, sim_threshold=0.55, conf_threshold=0.60):
        self.sim_threshold = sim_threshold
        self.conf_threshold = conf_threshold

    def evaluate(self, customer_message, predicted_intent, intent_confidence, retrieved_examples):
        """
        Evaluates the support inquiry and returns:
          - decision: 'AUTO-HANDLE' or 'ESCALATE'
          - reason: explicit explanation string
          - risk_level: 'LOW', 'MEDIUM', 'HIGH'
        """
        text = customer_message.strip()
        top_sim = retrieved_examples[0]["similarity_score"] if retrieved_examples else 0.0

        # Check 1: Sensitive intent domain
        if predicted_intent in SENSITIVE_INTENTS:
            return {
                "decision": "ESCALATE",
                "reason": SENSITIVE_INTENTS[predicted_intent],
                "risk_level": "HIGH"
            }

        # Check 2: Ambiguous or uninformative request
        if predicted_intent == "other_unclear":
            return {
                "decision": "ESCALATE",
                "reason": "Ambiguous query: Message lacks specific technical symptoms, device model, or OS version needed for automated troubleshooting.",
                "risk_level": "MEDIUM"
            }

        if len(text.split()) < 3 and not any(k in text.lower() for k in ["ios", "iphone", "update", "battery", "wifi"]):
            return {
                "decision": "ESCALATE",
                "reason": "Vague or truncated message: Customer inquiry contains insufficient detail to resolve automatically.",
                "risk_level": "MEDIUM"
            }

        # Check 3: Retrieval evidence strength
        if top_sim < self.sim_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Insufficient historical retrieval evidence: Top historical match similarity ({top_sim:.2f}) is below threshold ({self.sim_threshold:.2f}).",
                "risk_level": "MEDIUM"
            }

        # Check 4: Intent confidence
        if intent_confidence < self.conf_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f} < {self.conf_threshold:.2f}); routing to human specialist for triage.",
                "risk_level": "MEDIUM"
            }

        # Check 5: Routine troubleshooting with high confidence & evidence
        if predicted_intent in ROUTINE_SUPPORT_INTENTS:
            return {
                "decision": "AUTO-HANDLE",
                "reason": (
                    f"High-confidence intent '{predicted_intent}' (conf={intent_confidence:.2f}) with strong historical "
                    f"evidence (sim={top_sim:.2f}). Routine technical troubleshooting can be safely automated."
                ),
                "risk_level": "LOW"
            }

        # Default fallback
        return {
            "decision": "ESCALATE",
            "reason": "Inquiry requires human triage due to unverified support criteria.",
            "risk_level": "MEDIUM"
        }


if __name__ == "__main__":
    engine = EscalationEngine()
    
    # Test case 1: routine battery
    res1 = engine.evaluate(
        customer_message="My iPhone battery drains really fast after iOS 11 update",
        predicted_intent="battery_power",
        intent_confidence=0.95,
        retrieved_examples=[{"similarity_score": 0.78}]
    )
    print("Test 1 (Battery):", res1)

    # Test case 2: account security
    res2 = engine.evaluate(
        customer_message="I'm locked out of my Apple ID and cannot reset my password",
        predicted_intent="account_security_icloud",
        intent_confidence=0.95,
        retrieved_examples=[{"similarity_score": 0.81}]
    )
    print("Test 2 (Security):", res2)

    # Test case 3: low retrieval similarity
    res3 = engine.evaluate(
        customer_message="The widget thingy on my gadget is doing something weird",
        predicted_intent="device_not_working",
        intent_confidence=0.80,
        retrieved_examples=[{"similarity_score": 0.42}]
    )
    print("Test 3 (Low similarity):", res3)
