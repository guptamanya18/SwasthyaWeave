from flask import Flask, request, jsonify
import joblib
import json

model = joblib.load("model.joblib")

with open("disease_data.json", "r") as f:
    disease_data = json.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "SwasthyaWeave Medical Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symptoms = data.get("symptoms")

    if symptoms is None:
        return jsonify({"error": "No symptoms provided"}), 400

    prediction = model.predict([symptoms])[0]

    info = disease_data.get(prediction, {
        "description": "No data available",
        "treatment": "Consult doctor",
        "precautions": [],
        "severity": "Unknown"
    })

    return jsonify({
        "disease": prediction,
        "description": info["description"],
        "treatment": info["treatment"],
        "precautions": info["precautions"],
        "severity": info["severity"]
    })

if __name__ == "__main__":
    app.run(debug=True)
