import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from rake_nltk import Rake  # For key phrase extraction

# Ensure NLTK resources are available
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load TFIDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Reload the pre-trained BERT model and tokenizer
pretrained_model_dir = "pretrained_bert_model"
tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)
model = BertForSequenceClassification.from_pretrained(pretrained_model_dir)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize RAKE for key phrase extraction
rake = Rake()

# Preprocessing function
def preprocess_text(text):
    import string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Convert to lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    sentiment_category = "neutral"
    if sentiment_scores["compound"] > 0.05:
        sentiment_category = "positive"
    elif sentiment_scores["compound"] < -0.05:
        sentiment_category = "negative"
    return sentiment_scores, sentiment_category

# BERT prediction function
def predict_category(text):
    # Tokenize and prepare input for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

# Key phrase extraction function
def extract_key_phrases(text):
    rake.extract_keywords_from_text(text)
    key_phrases = rake.get_ranked_phrases()[:10]  # Get top 10 key phrases
    return key_phrases

# Streamlit App
st.title("News Article Classification and Sentiment Analysis Dashboard")
st.write("Enter a news article below, and the model will predict its category, analyze the emotional tone, and extract key phrases.")

# Input field for the news article
user_input = st.text_area("Enter the news article text here:")

# Predict button
if st.button("Predict"):
    if user_input.strip():
        # Preprocess user input
        preprocessed_text = preprocess_text(user_input)
        
        # Sentiment Analysis
        sentiment_scores, sentiment_category = analyze_sentiment(user_input)
        
        # Predict Category using BERT
        prediction = predict_category(preprocessed_text)
        
        # Extract Key Phrases
        key_phrases = extract_key_phrases(user_input)
        
        # Map prediction to category names
        category_mapping = {0: "Politics", 1: "Technology", 2: "Entertainment", 3: "Business"}
        predicted_category = category_mapping[prediction]
        
        # Display results
        st.success(f"The predicted category is: **{predicted_category}**")
        st.write(f"**Sentiment Analysis:** The article has a **{sentiment_category}** tone.")
        st.write("**Sentiment Scores:**")
        st.json(sentiment_scores)

        # Display key phrases
        st.write("**Top Key Phrases from the Article:**")
        for i, phrase in enumerate(key_phrases, 1):
            st.write(f"{i}. {phrase}")
    else:
        st.error("Please enter some text for prediction.")
