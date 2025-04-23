from flask import Blueprint, request, jsonify
from .model import get_model, get_feature_ranges
from .validation import validate_input
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Pest Detection API is running. Use POST /predict for predictions."
    })

@main.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    validation_result = validate_input(data)
    if not validation_result["valid"]:
        return jsonify({
            "error": "Invalid input data",
            "details": validation_result["message"]
        }), 400

    df = pd.DataFrame([data])
    model = get_model()
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "prediction_label": "Pest Detected" if prediction == 1 else "No Pest Detected",
        "confidence": float(probability)
    })

@main.route('/batch-predict', methods=['POST'])
def batch_predict():
    request_data = request.get_json()
    if 'data' not in request_data or not isinstance(request_data['data'], list):
        return jsonify({
            "error": "Invalid input format",
            "message": "Request must contain a 'data' field with a list of sensor readings"
        }), 400

    results = []
    for i, data_point in enumerate(request_data['data']):
        validation = validate_input(data_point)
        if not validation['valid']:
            return jsonify({
                "error": f"Invalid data at index {i}",
                "message": validation['message']
            }), 400

    df = pd.DataFrame(request_data['data'])
    model = get_model()
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "index": i,
            "prediction": int(pred),
            "prediction_label": "Pest Detected" if pred == 1 else "No Pest Detected",
            "confidence": float(prob)
        })

    return jsonify({
        "results": results,
        "count": len(results)
    })

@main.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "RandomForestClassifier",
        "features": list(get_feature_ranges().keys()),
        "feature_ranges": get_feature_ranges(),
        "output": {
            "0": "No Pest Detected",
            "1": "Pest Detected"
        },
        "version": "1.0.0"
    })
