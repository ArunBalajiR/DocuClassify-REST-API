from flask import Flask, request,jsonify
import fasttext
import re
import joblib



app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "NLP DOCUMENT CLASSIFICATION API"

# Load the FastText model from a pickle file using joblib
loaded_model = fasttext.load_model("./models/fasttext_model.bin")

# Load trained model
clf = joblib.load('./models/lob_predictor/lobmodel.joblib')
tfidf = joblib.load('./models/lob_predictor/lob_tfidf_model.joblib')
mlb = joblib.load('./models/lob_predictor/lob_mlb_model.joblib')

tclf = joblib.load('./models/tags_predictor/tagsmodel.joblib')
ttfidf = joblib.load('./models/tags_predictor/tags_tfidf_model.joblib')
tmlb = joblib.load('./models/tags_predictor/tags_mlb_model.joblib')

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


@app.route('/predictTags', methods=['POST'])
def predictTags():
    try:
        text = [request.json['text']]
        # LOB PREDICTION
        xt = tfidf.transform(text)
        y_hat = clf.predict(xt)
        lobtags = mlb.inverse_transform(y_hat)
        # tags prediction
        xxt = ttfidf.transform(text)
        yy_hat = tclf.predict(xxt)
        tags = tmlb.inverse_transform(yy_hat)
        return jsonify({
            "tags": list(tags[0]),
            "lobs": list(lobtags[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)})

def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

if __name__ == "__main__":
    app.run()
