import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="💡",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('churn_prediction_model.pkl')

model = load_model()

# --- UI ---
st.title("💡 Customer Churn Predictor")
st.caption("Analyze customer feedback and detect churn risk instantly.")

st.markdown("#### ✍️ Enter Customer Feedback")
user_review = st.text_area(
    "",
    height=150,
    placeholder="e.g. I'm not happy with the service..."
)

# Button
if st.button("Analyze", use_container_width=True):

    if user_review.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = model.predict([user_review])[0]
        probabilities = model.predict_proba([user_review])[0]
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("🔍 Result")

        col1, col2 = st.columns([1, 3])

        if prediction == 1:
            with col1:
                st.image(
                    "https://cdn-icons-png.flaticon.com/512/190/190411.png",
                    width=80
                )
            with col2:
                st.success("Customer is likely to stay 😊")
                st.write(f"Confidence: **{confidence:.2f}%**")

        else:
            with col1:
                st.image(
                    "https://cdn-icons-png.flaticon.com/512/190/190406.png",
                    width=80
                )
            with col2:
                st.error("Customer may churn ⚠️")
                st.write(f"Confidence: **{confidence:.2f}%**")
