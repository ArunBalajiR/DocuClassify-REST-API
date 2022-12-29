from flask import Flask, request,jsonify
import fasttext
import re

app = Flask(__name__)

# Load the FastText model from a pickle file using joblib
loaded_model = fasttext.load_model("fasttext_model.bin");

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    preprocessed_text = preprocess(text)
    prediction = loaded_model.predict(preprocessed_text)
    # prediction_label = str(prediction[0][0]).replace('__label__', '')
    prediction_label, prediction_probability = prediction[0][0], prediction[1][0]
    label = prediction_label.replace('__label__', '')
    return jsonify({
        "prediction": label,
        "probability": prediction_probability
    })


def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

if __name__ == '__main__':
    app.run()
