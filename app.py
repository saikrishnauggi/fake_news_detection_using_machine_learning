import os
import re
import string
import logging
import joblib
import nltk
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer

# Setup logging and NLTK
logging.basicConfig(level=logging.INFO)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# --- HTML Template as String ---
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            text-align: center;
            padding-top: 50px;
        }
        h2 {
            color: #fff;
        }
        form {
            background: rgba(0, 0, 0, 0.6);
            padding: 30px;
            margin: auto;
            width: 60%;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
            border-radius: 15px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 10px;
            resize: vertical;
            border: none;
        }
        select, input[type="submit"] {
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        select {
            background: #fff;
            color: #000;
        }
        input[type="submit"] {
            background: #00c6ff;
            color: white;
        }
        input[type="submit"]:hover {
            background: #0072ff;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
            color: #00ffea;
        }
    </style>
</head>
<body>
    <h2>Fake News Detection</h2>
    <form method="POST" action="/predict">
        <textarea name="news_text" rows="10" placeholder="Paste your news article here..." required></textarea><br><br>
        <select name="model">
            <option value="nb">Naive Bayes</option>
            <option value="lr">Logistic Regression</option>
            <option value="svm">SVM</option>
        </select><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <div class="result">Prediction: {{ prediction }}</div>
    {% endif %}
</body>
</html>
"""

# --- Text Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# --- Data Loading and Training ---
def load_and_prepare_data(fake_path, true_path):
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        logging.error("Fake.csv or True.csv not found.")
        return None
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 1
    df_true['label'] = 0
    min_len = min(len(df_fake), len(df_true))
    df = pd.concat([df_fake.sample(min_len, random_state=42), df_true.sample(min_len, random_state=42)])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['text'] = df['text'].apply(clean_text)
    return df

def train_and_save_models():
    data = load_and_prepare_data("Fake.csv", "True.csv")
    if data is None:
        return

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'naive_bayes_model.pkl': MultinomialNB(),
        'logistic_regression_model.pkl': LogisticRegression(max_iter=1000),
        'svm_model.pkl': SVC(kernel='linear', probability=True)
    }

    for filename, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logging.info(f"{filename} Accuracy: {accuracy_score(y_test, y_pred)}")
        joblib.dump(model, filename)

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    logging.info("All models and vectorizer saved successfully.")

# --- Load Models ---
required_files = ["naive_bayes_model.pkl", "logistic_regression_model.pkl", "svm_model.pkl", "tfidf_vectorizer.pkl"]
if not all(os.path.exists(f) for f in required_files):
    logging.info("Training models as pickle files not found.")
    train_and_save_models()

nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Routes ---
@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    model_choice = request.form.get('model')
    cleaned = clean_text(news_text)
    vect_text = vectorizer.transform([cleaned])

    if model_choice == 'lr':
        prediction = lr_model.predict(vect_text)[0]
    elif model_choice == 'svm':
        prediction = svm_model.predict(vect_text)[0]
    else:
        prediction = nb_model.predict(vect_text)[0]

    result = "Fake News" if prediction == 1 else "Real News"
    return render_template_string(html_template, prediction=result)

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
