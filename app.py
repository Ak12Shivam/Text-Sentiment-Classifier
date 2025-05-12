from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pytesseract
from PIL import Image
import docx
import PyPDF2
import os
import io

app = Flask(__name__)

# Load sentiment & zero-shot classification models
sentiment_analyzer = pipeline("sentiment-analysis")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels for classification
candidate_labels = ["technology", "finance", "health", "sports", "politics", "education", "travel"]

def extract_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_storage.stream)
        return pytesseract.image_to_string(image)

    elif filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file_storage.stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif filename.endswith(('.docx', '.doc')):
        doc = docx.Document(file_storage.stream)
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith('.txt'):
        return file_storage.stream.read().decode("utf-8")

    return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form.get('text', '')
    file = request.files.get('file')

    if file:
        extracted_text = extract_text(file)
        input_text += "\n" + extracted_text

    if not input_text.strip():
        return jsonify({"error": "No input text found."}), 400

    # Perform sentiment analysis
    sentiment = sentiment_analyzer(input_text[:512])[0]['label']  # Limit to 512 tokens

    # Perform text classification
    classification = classifier(input_text[:512], candidate_labels)
    best_label = classification['labels'][0]

    return jsonify({
        "classification": best_label,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
