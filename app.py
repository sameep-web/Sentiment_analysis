import streamlit as st
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer

import nltk

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load model and vectorizer
best_model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Optional: If you saved model info in a text file, you can load it
try:
    with open("model_info.txt", "r") as f:
        model_info = f.read()
except:
    model_info = "Best model loaded successfully!"

# Preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis App")
st.write("Enter any sentence or review and check whether it's **Positive** or **Negative**.")

# Show model info
st.info(model_info)

# User input
user_input = st.text_area("Enter your sentence here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid sentence.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = best_model.predict(vectorized_input)[0]

        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")

