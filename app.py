import streamlit as st
import joblib

# 1. Page Configuration (Friendly & Professional)
st.set_page_config(page_title="Customer Review Analyzer", page_icon="🌟", layout="centered")

# 2. Load the AI Brain
@st.cache_resource
def load_model():
    return joblib.load('review_rescue_model.pkl')

model = load_model()

# 3. Build the User Interface (Simple & Business-Focused)
st.title("🌟 Customer Review Analyzer")
st.markdown("### *Keep your customers happy and prevent churn.*")
st.write("Paste a recent customer review below to instantly check if they need a follow-up or a discount offer.")

# Text Input
user_review = st.text_area("Customer Review:", height=150, placeholder="Type or paste the customer's feedback here...")

# Analyze Button
if st.button("Analyze Review", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # 4. The AI does the math silently in the background
        prediction = model.predict([user_review])[0]
        probabilities = model.predict_proba([user_review])[0]
        confidence = max(probabilities) * 100
        
        st.markdown("---")
        st.subheader("Analysis Result:")
        
        # 5. Simple, Actionable Recommendations
        if prediction == 1:
            st.success("🟢 **HAPPY CUSTOMER**")
            st.write("**Action:** No urgent action needed. Consider replying to thank them, or ask them to leave a public review on Google!")
            st.caption(f"AI Certainty: {confidence:.0f}%")
        else:
            st.error("🔴 **AT-RISK CUSTOMER (CHURN ALERT)**")
            st.write("**Action:** Reach out immediately. Send a polite apology along with a **20% discount code** for their next visit to win them back.")
            st.caption(f"AI Certainty: {confidence:.0f}%")
