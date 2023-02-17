from flask import Flask, request,jsonify
import fasttext
import re
import joblib
# Load trained model
clf = joblib.load('./models/lob_predictor/lobmodel.joblib')
tfidf = joblib.load('./models/lob_predictor/lob_tfidf_model.joblib')
mlb = joblib.load('./models/lob_predictor/lob_mlb_model.joblib')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "NLP DOCUMENT CLASSIFICATION API"

@app.route('/gettags', methods=['GET'])
def predictLOBsandTags():
    try:
        text = request.json['text']
        preprocessed_text = preprocess(text)
        print(preprocessed_text)
        return "hi"
    except Exception as e:
        return jsonify({"error": str(e)})

# Load the FastText model from a pickle file using joblib
loaded_model = fasttext.load_model("./models/fasttext_model.bin");
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        preprocessed_text = preprocess(text)
        prediction = loaded_model.predict(preprocessed_text)
        prediction_label, prediction_probability = prediction[0][0], prediction[1][0]
        label = prediction_label.replace('__label__', '')
        return jsonify({
            "prediction": label,
            "probability": round(prediction_probability * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

if __name__ == "__main__":
    app.run()
