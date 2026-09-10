"""
src/generator.py
================

Grounded Reply Generation module for Apple Support Agent.

Generates concise, helpful customer support responses grounded strictly
in historical Apple Support responses retrieved from the FAISS corpus.

Features:
- Primary: Google Gemini API (gemini-1.5-flash / gemini-2.5-flash via REST API)
  using GEMINI_API_KEY from environment / .env file.
- Fallback: Offline Grounded Generator that synthesizes high-quality Apple-style
  replies from the top historical retrieved matches without external API dependencies.
- Strict anti-hallucination guidelines (no fake warranty promises, no unsupported claims).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_INSTRUCTION = """You are an official Apple Support specialist on Twitter (@AppleSupport).
Your goal is to draft a helpful, professional, and concise customer support tweet (1 to 3 sentences maximum).

Follow these strict guidelines:
1. GROUNDING: Base your advice strictly on the provided historical Apple Support responses. Do not invent troubleshooting steps or policy claims that are not supported by the examples.
2. TONE: Be empathetic, polite, calm, and action-oriented (e.g., "We'd love to help with this", "Let's take a look together").
3. CLARIFICATION: If the customer's message lacks necessary details (e.g. device model, exact iOS version, error message), ask a targeted clarifying question.
4. SENSITIVE/COMPLEX ISSUES: If the issue requires personal account details (Apple ID, billing, orders) or complex troubleshooting, invite them to send a Direct Message (DM).
5. NO UNSUPPORTED CLAIMS: Never promise free hardware replacements, refunds, or software release dates unless explicitly stated in the historical evidence.
6. NO TWITTER HANDLES: Do not include placeholder user handles like @123456 or @user in your reply."""


def clean_tweet_handles(text):
    """Removes numerical user handles like @115858 from historical text."""
    cleaned = re.sub(r"@\d+", "", text)
    cleaned = re.sub(r"@AppleSupport", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


class OfflineGroundedGenerator:
    """Deterministic, high-quality fallback generator grounded in retrieved evidence."""
    def generate(self, customer_message, predicted_intent, retrieved_examples):
        if not retrieved_examples:
            return "We'd be glad to help with this! To get started, could you let us know which device model and iOS version you're currently using?"

        top_match = retrieved_examples[0]
        top_apple_resp = clean_tweet_handles(top_match.get("apple_response", ""))
        top_sim = top_match.get("similarity_score", 0.0)

        # If retrieval similarity is solid, adapt the proven historical Apple response
        if top_sim >= 0.65 and len(top_apple_resp) > 10:
            return top_apple_resp

        # Category-specific grounded fallbacks based on real Apple support templates
        if predicted_intent == "battery_power":
            return "We want to help ensure you get the best battery life from your device. Take a look at these tips: https://apple.co/battery and let us know if the issue persists."
        elif predicted_intent == "autocorrect_text_bug":
            return "We're here to help! Please update your device to the latest iOS version in Settings > General > Software Update, which includes a fix for this keyboard issue."
        elif predicted_intent == "connectivity":
            return "We'd love to help get you connected. Have you tried restarting your device and resetting network settings in Settings > General > Reset?"
        elif predicted_intent == "account_security_icloud":
            return "We know how important your account security is. For help resetting your password or accessing your Apple ID, please visit https://iforgot.apple.com or DM us."
        elif predicted_intent == "billing_repair_order":
            return "We'd be happy to look into this with you. Please connect with us in a Direct Message so we can securely check on your order or appointment."
        elif predicted_intent == "screen_hardware_damage":
            return "We understand how important your display is. We recommend scheduling an appointment at an Apple Store Genius Bar: https://getsupport.apple.com"
        elif predicted_intent == "ios_software_update":
            return "We're here to help. What happens when you attempt to update? Please let us know your current iOS version in Settings > General > About."
        elif predicted_intent == "app_appstore_media":
            return "Let's help get your apps working again. Have you tried force-closing the app and restarting your device?"
        else:
            return "We're here to help! To get started, could you tell us more about what you're experiencing, including your device model and iOS version?"


class ReplyGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.offline_generator = OfflineGroundedGenerator()

    def generate_reply(self, customer_message, predicted_intent, retrieved_examples):
        """
        Drafts a response using Gemini API if key is available,
        otherwise falls back to offline grounded generator.
        """
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            return self.offline_generator.generate(customer_message, predicted_intent, retrieved_examples)

        # Build prompt with retrieved evidence
        evidence_text = ""
        for i, ex in enumerate(retrieved_examples[:3], 1):
            clean_resp = clean_tweet_handles(ex.get("apple_response", ""))
            clean_cust = ex.get("customer_message", "").replace("\n", " ")
            evidence_text += (
                f"Example {i}:\n"
                f"Customer: {clean_cust}\n"
                f"Apple Response: {clean_resp}\n\n"
            )

        prompt = (
            f"Incoming Customer Tweet: \"{customer_message}\"\n"
            f"Classified Intent: {predicted_intent}\n\n"
            f"Historical Apple Support Evidence:\n{evidence_text}\n"
            f"Draft the official @AppleSupport reply tweet:"
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{SYSTEM_INSTRUCTION}\n\n{prompt}"}]}
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 100
                }
            }
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text
            else:
                # Fallback on API error (rate limit, invalid key, etc.)
                return self.offline_generator.generate(customer_message, predicted_intent, retrieved_examples)
        except Exception:
            return self.offline_generator.generate(customer_message, predicted_intent, retrieved_examples)


if __name__ == "__main__":
    generator = ReplyGenerator()
    test_msg = "My battery is dying in 2 hours since I updated to iOS 11"
    test_intent = "battery_power"
    test_retrieved = [{
        "similarity_score": 0.76,
        "customer_message": "battery draining fast after 11.0.2",
        "apple_response": "@12345 We want to help! Check out: https://apple.co/battery for setting recommendations."
    }]
    
    reply = generator.generate_reply(test_msg, test_intent, test_retrieved)
    print("Generated Reply:\n", reply)
