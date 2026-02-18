import streamlit as st
import numpy as np

# Load the models and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('svm_model(1).pkl')
naive_bayes_model = joblib.load('naive_bayes_model(2).pkl')
scaler = joblib.load('scaler.pkl')

# Define function for sentiment prediction
def predict_sentiment(text, model, vectorizer, scaler):
    # Preprocess the text
    text_tfidf = vectorizer.transform([text])
    text_scaled = scaler.transform(text_tfidf)
    
    # Predict sentiment with SVM model (or Naive Bayes model)
    prediction = model.predict(text_scaled)
    sentiment = 'Positive' if prediction[0] == 'Positive' else ('Negative' if prediction[0] == 'Negative' else 'Neutral')
    
    return sentiment

# Streamlit UI
st.title('Sentiment Analysis with SVM and Naive Bayes')

st.markdown("""
    This app analyzes the sentiment of the given text.
    You can choose between two models: **Support Vector Machine (SVM)** or **Naive Bayes**.
""")

# Get user input
text_data = st.text_area("Enter the text you want to analyze:")

model_option = st.selectbox(
    "Choose the model for sentiment analysis:",
    ["SVM", "Naive Bayes"]
)

# When the user clicks the "Analyze" button
if st.button('Analyze Sentiment'):
    if text_data:
        # Choose the selected model
        if model_option == 'SVM':
            sentiment = predict_sentiment(text_data, svm_model, vectorizer, scaler)
        else:
            sentiment = predict_sentiment(text_data, naive_bayes_model, vectorizer, scaler)
        
        # Display the result
        st.write(f"**Predicted Sentiment**: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
