import streamlit as st
import joblib

# 1. Page Configuration (Academic/Technical Vibe)
st.set_page_config(page_title="NLP Churn Predictor", page_icon="⚙️", layout="centered")

# 2. Load the AI Brain
@st.cache_resource
def load_model():
    return joblib.load('review_rescue_model.pkl')

model = load_model()

# 3. Build the User Interface
st.title("⚙️ NLP-Driven Churn Prediction Model")
st.markdown("### *Lexical Analysis for Customer Attrition Warning*")
st.write("Enter text data below to run inference. The model utilizes a TF-IDF vectorization pipeline and Logistic Regression to classify semantic intent and predict churn probability.")

# Text box for the user
user_review = st.text_area("Input Text Data:", height=150, placeholder="Enter textual feedback for classification...")

# Analyze Button
if st.button("Initialize NLP Pipeline", type="primary"):
    if user_review.strip() == "":
        st.warning("Error: Input array cannot be empty.")
    else:
        # 4. The AI does its magic
        prediction = model.predict([user_review])[0]
        probabilities = model.predict_proba([user_review])[0]
        confidence = max(probabilities) * 100
        
        st.markdown("---")
        st.subheader("Classification Output:")
        
        # 5. UI Upgrade: Create two columns (Image on left, Text on right)
        col1, col2 = st.columns([1, 4]) 
        
        # 6. Technical Logic Translation with Images
        if prediction == 1:
            with col1:
                # Displays a happy 3D icon
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
            with col2:
                st.success("🟢 **CLASS 1: SAFE (POSITIVE SENTIMENT)**")
                st.write("**Analysis:** Lexical weights indicate a high probability of customer retention.")
                st.write(f"*Pipeline Confidence: {confidence:.2f}%*")
        else:
            with col1:
                # Displays a sad/angry 3D icon
                st.image("https://cdn-icons-png.flaticon.com/512/190/190406.png", width=100)
            with col2:
                st.error("🔴 **CLASS 0: CHURN RISK (NEGATIVE SENTIMENT)**")
                st.write("**Analysis:** Lexical weights indicate a critical risk of customer attrition. Immediate retention protocol recommended.")
                st.write(f"*Pipeline Confidence: {confidence:.2f}%*")
 
