"""
app.py
======
Interactive Streamlit Demo for Apple Support Agent
Built by Priya Pandey
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
from src.classifier import HybridIntentClassifier
from src.pipeline import SupportAgent

st.set_page_config(
    page_title="Apple Support Agent | Priya Pandey",
    page_icon="🍏",
    layout="wide"
)

SAMPLE_QUERIES = {
    "Battery Drain (Routine)": "My iPhone 7 battery is draining super fast after updating to iOS 11. Can't even last 3 hours!",
    "Account Lockout (Sensitive)": "I am locked out of my Apple ID and iCloud. It says account disabled for security reasons, please help!",
    "Keyboard Autocorrect Bug": "Why is my iPhone changing the letter 'I' into an exclamation mark with a box? Please fix this autocorrect bug!",
    "Cracked Display (Hardware)": "Dropped my iPhone and the screen shattered completely and touch screen is unresponsive. How much to fix?",
    "Wi-Fi Connectivity": "Ever since the new update my phone won't connect to my home Wi-Fi network at all. The toggle is greyed out.",
    "Billing Dispute (Sensitive)": "I got charged twice on my credit card for my Apple Music subscription this month. I want an immediate refund.",
    "Ambiguous / Low Context": "@AppleSupport please fix this immediately, it's driving me crazy!"
}


@st.cache_resource
def load_agent():
    # Cache the agent instance across user sessions to prevent reload overhead
    return SupportAgent()


def main():
    st.title("Apple Customer Support AI Agent")
    st.caption("Grounded support triage and reply generation on Kaggle's Apple Twitter dataset | Built by Priya Pandey")

    # Sidebar: System configuration and preloaded test queries
    with st.sidebar:
        st.subheader("System Overview")
        st.markdown("""
        **Pipeline stages:**
        1. **Intent Classification:** 10-class Apple taxonomy (Rules + TF-IDF/Logistic Regression).
        2. **Historical Retrieval:** FAISS `IndexFlatIP` over 19,545 training conversations.
        3. **Explainable Escalation:** Risk assessment + similarity thresholding.
        4. **Grounded Reply:** Grounded strictly in retrieved historical Apple tweets.
        """)
        st.divider()
        st.subheader("Sample Test Inquiries")
        selected_sample = st.selectbox("Select a test customer tweet:", list(SAMPLE_QUERIES.keys()))
        sample_text = SAMPLE_QUERIES[selected_sample]

    agent = load_agent()

    # User Input area
    user_query = st.text_area(
        "Customer message / tweet:",
        value=sample_text,
        height=95,
        help="Enter any support inquiry or select a scenario from the sidebar."
    )

    if st.button("Submit Inquiry", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a customer message to proceed.")
            return

        with st.spinner("Processing through 4-stage pipeline..."):
            result = agent.handle(user_query)

        st.divider()
        st.subheader("Triage & Response Output")

        # Routing decision display
        decision = result["decision"]
        reason = result["reason"]
        risk = result["risk_level"]

        if decision == "AUTO-HANDLE":
            st.success(f"**Decision: AUTO-HANDLE**  \n*{reason}*")
        else:
            st.error(f"**Decision: ESCALATE TO HUMAN**  \n*{reason}*")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Intent Classification")
            st.metric("Predicted Intent", result["intent"].replace("_", " ").title())
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            st.markdown(f"**Assessed Risk Level:** `{risk}`")

        with col2:
            st.markdown("##### Drafted Response (@AppleSupport)")
            st.info(f"\"{result['drafted_reply']}\"")

        # Retrieved Historical Context Inspection
        st.divider()
        with st.expander("Inspect Retrieved Historical Context (Top-3 FAISS Matches)", expanded=True):
            for i, match in enumerate(result["retrieved_examples"], 1):
                sim = match["similarity_score"]
                st.markdown(f"**Match #{i} — Cosine Similarity: `{sim:.4f}` | Intent: `{match.get('intent', 'N/A')}` | Conversation ID: `{match.get('conversation_id', 'N/A')}`**")
                st.markdown(f"> **Customer:** {match['customer_message']}")
                st.markdown(f"> **Historical Apple Reply:** {match['apple_response']}")
                st.markdown("")


if __name__ == "__main__":
    main()
