import streamlit as st
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="💡",
    layout="centered"
)

# -----------------------------
# Hide Streamlit Default UI (GitHub, Menu, Footer)
# -----------------------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stAppDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load('churn_prediction_model.pkl')

model = load_model()

# -----------------------------
# UI Header
# -----------------------------
st.title("💡 Customer Churn Predictor")
st.caption("Analyze customer feedback and generate retention insights instantly.")

st.markdown("#### ✍️ Enter Customer Feedback")
user_review = st.text_area(
    "",
    height=150,
    placeholder="e.g. I'm not happy with the service..."
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Analyze", use_container_width=True):

    if user_review.strip() == "":
        st.warning("Please enter some text to analyze.")

    else:
        # Model prediction
        prediction = model.predict([user_review])[0]
        probabilities = model.predict_proba([user_review])[0]
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("🔍 Result")

        # The columns now ONLY hold the icon and the Safe/Churn status
        col1, col2 = st.columns([1, 4])

        # -----------------------------
        # SAFE CASE (Class 1 = Positive)
        # -----------------------------
        if prediction == 1:
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=80)

            with col2:
                st.success("Customer is likely to stay 😊")
                st.write(f"Confidence: **{confidence:.2f}%**")

            # This is OUTSIDE the columns, so it spans the full width of the app!
            st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room
            st.markdown("### 💬 Suggested Response")
            st.info(
                "Thank you for your valuable feedback! We're glad you're having a great experience. "
                "We truly appreciate your support and look forward to serving you even better."
            )

        # -----------------------------
        # CHURN CASE (Class 0 = Negative)
        # -----------------------------
        else:
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/190/190406.png", width=80)

            with col2:
                st.error("Customer may churn ⚠️")
                st.write(f"Confidence: **{confidence:.2f}%**")

            # This is OUTSIDE the columns, so it spans the full width of the app!
            st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room
            st.markdown("### 💬 Suggested Response")
            st.warning(
                "We're really sorry for your experience. Your feedback is important to us. "
                "As a token of apology, we'd like to offer you a discount on your next purchase. "
                "We are committed to improving our service."
            )
