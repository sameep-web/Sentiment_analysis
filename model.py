import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import load_dataset  # Hugging Face dataset

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load IMDb dataset
print("Downloading IMDb dataset...")
dataset = load_dataset("imdb")

# Convert to DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
df = pd.concat([train_df, test_df], ignore_index=True)
df.rename(columns={"text": "text", "label": "label"}, inplace=True)
print(f"Dataset loaded with shape: {df.shape}")

# 2. Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)    # keep only letters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]
    return " ".join(tokens)

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess_text)

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,   # IMDb is large; more features help
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Define models
models = {
    "LogisticRegression": LogisticRegression(C=2.0, max_iter=2000, solver='liblinear'),
    "SVM": LinearSVC(C=1.0, max_iter=3000)
}

# 5. Train and evaluate
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_score:
        best_model = model
        best_score = acc
        best_name = name

# 6. Save best model and vectorizer
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(f"\n✅ Best model: {best_name} saved with accuracy: {best_score:.4f}")
