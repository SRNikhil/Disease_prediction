from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure to download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

stop = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop]
    return ' '.join(words)

app = Flask(__name__)

# Load the saved KNN model and TF-IDF vectorizer
knn_model = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the Treatments CSV into a DataFrame
treatments_df = pd.read_csv('Treatments.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    symptom = data['symptom']  # Symptom string from front end

    # Preprocess and vectorize the symptom input
    preprocessed_symptom = preprocess_text(symptom)
    symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])

    # Predict the disease
    predicted_disease = knn_model.predict(symptom_tfidf)[0]

    # Fetch the treatment for the predicted disease
    treatment_row = treatments_df[treatments_df['disease'].str.lower() == predicted_disease.lower()]
    
    if not treatment_row.empty:
        treatment = treatment_row['prescription'].values[0]
    else:
        treatment = "No treatment information available"

    # Send the prediction and treatment as a JSON response
    return jsonify({
        'predicted_disease': predicted_disease,
        'treatment': treatment
    })

if __name__ == '__main__':
    app.run(debug=True)
