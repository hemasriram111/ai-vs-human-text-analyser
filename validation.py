import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('ai_vs_human_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI
st.title("AI vs Human Text Classifier")
st.write("Enter a piece of text below to determine whether it was written by AI or a human.")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        # Transform and predict
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        result = "AI-generated text" if prediction == 1 else "Human-written text"
        
        # Display result
        st.subheader("Prediction:")
        st.success(result)
    else:
        st.warning("Please enter some text before predicting.")
