# ⚙️ NLP-Driven Churn Prediction Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Accuracy](https://img.shields.io/badge/Model_Accuracy-90.41%25-brightgreen)

## 📌 Overview
An automated Natural Language Processing (NLP) pipeline and web application designed to predict customer churn.

In the highly competitive retail and digital service sectors, businesses generate thousands of textual data points daily. This project utilizes computational linguistics to read raw English feedback, interpret the semantic intent, and mathematically classify the user as "Safe" or an "Immediate Churn Risk," allowing business operators to initiate retention protocols before a customer is lost.

## ✨ Key Features
* **Automated Text Triage:** Instantly classifies raw customer reviews, support tickets, and survey responses.
* **Contextual N-Gram Analysis:** Utilizes Bigram vectorization to understand complex negations (e.g., catching "not good" instead of mathematically isolating "good").
* **Real-Time Enterprise Dashboard:** A sleek, user-friendly Streamlit web interface for live, offline inference by non-technical operators.
* **High-Recall Detection:** Tuned specifically to catch critical churn risks with a 91% recall rate.

## 📸 Application Demo

![Screenshot 1](link-to-your-safe-screenshot.png) 
> *Figure 1: The model correctly classifying a positive review and logging a safe status.*

![Screenshot 2](link-to-your-critical-screenshot.png)
> *Figure 2: The system flagging a churn risk and recommending an immediate retention protocol.*

## ⚙️ Technical Architecture & Methodology
### 1. Data Processing
* Trained on a balanced subset of **100,000 records** from a secondary customer review dataset.
* Ambiguous (3-star) reviews were dropped to force strict binary classification (0 = Churn Risk, 1 = Safe).

### 2. The NLP Pipeline
* **Text Vectorization:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency).
* **Dimensionality:** Capped at `max_features=10,000` to optimize RAM usage.
* **Tokenization:** `ngram_range=(1, 2)` to capture unigrams and bigrams.

### 3. Algorithm Selection
Benchmarking was performed on a 20,000-record sample comparing Random Forest, Naive Bayes, and Logistic Regression. **Logistic Regression** was selected as the production algorithm due to its superior accuracy (86.55% on the sample) and millisecond inference speed (0.07s) on sparse matrices.

## 📊 Model Performance
The final production model was tested on 20,000 completely unseen records:
* **True Testing Accuracy:** 90.41%
* **Precision (Negative Class):** 0.90
* **Recall (Negative Class):** 0.91
* **F1-Score:** Perfectly balanced at 0.90 across both classes.

## 🚀 Installation & Setup
To run this project locally on your machine:

**1. Clone the repository:**
```bash
https://github.com/ibnuumar448/Churn-Prediction-Model.git
