from flask import Flask, request, jsonify
import joblib

# Load trained model
model = joblib.load("model.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return "SwasthyaWeave Backend is Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symptoms = data.get("symptoms")

    if symptoms is None:
        return jsonify({"error": "No symptoms provided"}), 400

    prediction = model.predict([symptoms])[0]

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
